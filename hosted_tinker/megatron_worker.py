"""Megatron worker process for distributed training.

Uses PyTorch DDP (not FSDP) with HuggingFace model + PEFT LoRA.
Each GPU holds a full copy of the model with manual gradient all_reduce.

Launched via mp.Process by MegatronBackend (queue-based IPC), or via torchrun
(file-based IPC fallback).

Supports both H100 and B200 GPUs:
- H100: Uses NCCL for all collectives (broadcast_object_list, etc.)
- B200: Uses Gloo group for object collectives (NCCL hangs on Blackwell)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Loss functions (same as pytorch_backend.py)
_CL, _CH = 0.8, 1.2
def _slm(v, m): return (v * m).sum() / m.sum().clamp(min=1.0)
def ce_loss(lp, m, _s, _a, _c): return -_slm(lp, m)
def is_loss(lp, m, s, a, _c): return -_slm(torch.exp(lp - s) * a, m)
def ppo_loss(lp, m, s, a, c):
    cl = (c or {}).get("clip_low_threshold", _CL); ch = (c or {}).get("clip_high_threshold", _CH)
    r = torch.exp(lp - s); return -_slm(torch.min(r * a, torch.clamp(r, cl, ch) * a), m)
def cispo_loss(lp, m, s, a, c):
    cl = (c or {}).get("clip_low_threshold", _CL); ch = (c or {}).get("clip_high_threshold", _CH)
    r = torch.exp(lp - s); return -_slm(torch.clamp(r, cl, ch).detach() * lp * a, m)
LOSS_FN = {"cross_entropy": ce_loss, "importance_sampling": is_loss, "ppo": ppo_loss, "cispo": cispo_loss}


def _init_model(base_model, lora_rank, lora_alpha, gradient_checkpointing, device):
    """Load model, apply LoRA. Returns (model, optimizer, tokenizer, lora_params)."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    module_names = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [m for m in ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj",
                           "gate_proj", "up_proj", "down_proj"] if m in module_names]
    peft_config = PeftLoraConfig(task_type=TaskType.CAUSAL_LM, r=lora_rank,
                                 lora_alpha=lora_alpha, target_modules=targets, bias="none")
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=0.0, betas=(0.9, 0.95), eps=1e-8)

    return model, optimizer, tokenizer, lora_params


def _run_command_loop(rank, world_size, model, optimizer, tokenizer, lora_params,
                      device, obj_group, default_mbs, lora_sync_dir, base_model_name,
                      get_cmd_fn, send_result_fn):
    """Shared command loop for both queue-based and file-based IPC."""
    pad_id = tokenizer.pad_token_id or 0
    accum_count = 0

    while True:
        cmd_data = [None]
        if rank == 0:
            cmd_data[0] = get_cmd_fn()

        dist.broadcast_object_list(cmd_data, src=0, group=obj_group)
        cmd = cmd_data[0]

        if cmd["type"] == "shutdown":
            break

        elif cmd["type"] in ("forward_backward", "forward"):
            compute_grad = cmd["type"] == "forward_backward"
            batch = cmd["batch"]
            n_examples = len(batch["all_input_ids"])

            all_lp = [None] * n_examples
            all_loss = [None] * n_examples

            if compute_grad:
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()

            micro_batch_size = cmd.get("micro_batch_size", default_mbs)

            sorted_indices = sorted(range(n_examples), key=lambda i: len(batch["all_input_ids"][i]))
            my_indices = sorted_indices[rank::world_size]
            my_indices.sort(key=lambda i: len(batch["all_input_ids"][i]))

            for mb_start in range(0, len(my_indices), micro_batch_size):
                mb_indices = my_indices[mb_start:mb_start + micro_batch_size]
                seqs = [batch["all_input_ids"][i] for i in mb_indices]
                max_len = max(len(s) for s in seqs)

                input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
                attn_mask = torch.zeros(len(seqs), max_len, dtype=torch.long, device=device)
                for j, seq in enumerate(seqs):
                    input_ids[j, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                    attn_mask[j, :len(seq)] = 1

                if compute_grad:
                    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                else:
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

                logits = out.logits
                log_probs = F.log_softmax(logits, dim=-1)
                del logits, out

                total_loss = None
                gpu_lps = []
                for j, idx in enumerate(mb_indices):
                    seq_len = len(batch["all_input_ids"][idx])
                    tgt = torch.tensor(batch["all_targets"][idx][:seq_len], dtype=torch.long, device=device)
                    target_lp = log_probs[j, :seq_len].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

                    gpu_lps.append((idx, target_lp.detach().float(), seq_len))

                    if compute_grad:
                        wt = torch.tensor(batch["all_token_weights"][idx][:seq_len], dtype=torch.bfloat16, device=device)
                        slp = torch.tensor(batch["all_sampling_logprobs"][idx][:seq_len], dtype=torch.bfloat16, device=device)
                        adv = torch.tensor(batch["all_advantages"][idx][:seq_len], dtype=torch.bfloat16, device=device)
                        loss_fn = LOSS_FN.get(batch["all_loss_fns"][idx], ce_loss)
                        loss_j = loss_fn(target_lp, wt, slp, adv, batch["all_loss_fn_configs"][idx])
                        total_loss = loss_j if total_loss is None else total_loss + loss_j

                del log_probs

                # Batch transfer GPU → CPU
                for idx, t, sl in gpu_lps:
                    all_lp[idx] = t.cpu().tolist()
                    all_loss[idx] = [0.0] * sl

                if compute_grad and total_loss is not None:
                    total_loss.backward()
                    accum_count += len(mb_indices)

            # Gather results to rank 0
            my_result = {i: (all_lp[i], all_loss[i]) for i in my_indices if all_lp[i] is not None}
            gathered = [None] * world_size
            dist.all_gather_object(gathered, my_result, group=obj_group)

            if rank == 0:
                merged_lp = [None] * n_examples
                merged_loss = [None] * n_examples
                for rank_data in gathered:
                    if rank_data:
                        for i, (lp, ls) in rank_data.items():
                            merged_lp[i] = lp
                            merged_loss[i] = ls

                send_result_fn({"logprobs": merged_lp, "losses": merged_loss})

        elif cmd["type"] == "optim_step":
            adam = cmd["adam_params"]
            for pg in optimizer.param_groups:
                pg["lr"] = adam["learning_rate"]
                pg["betas"] = (adam["beta1"], adam["beta2"])
                pg["eps"] = adam["eps"]
                pg["weight_decay"] = adam["weight_decay"]

            # Coalesced gradient all_reduce
            trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if world_size > 1 and trainable:
                grads = [p.grad.data for p in trainable]
                flat = torch._utils._flatten_dense_tensors(grads)
                dist.all_reduce(flat, op=dist.ReduceOp.AVG)
                for g, uf in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
                    g.copy_(uf)

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item() if trainable else 0.0

            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

            if rank == 0:
                lora_state = {k: v.cpu().contiguous() for k, v in model.state_dict().items()
                              if "lora_" in k or "modules_to_save" in k}
                save_dir = os.path.join(lora_sync_dir, "adapter")
                os.makedirs(save_dir, exist_ok=True)
                _vllm_unsupported = {"in_proj_qkv", "out_proj"}
                vllm_state = {k: v for k, v in lora_state.items()
                              if not any(f".{m}." in k for m in _vllm_unsupported)}
                save_file(vllm_state, os.path.join(save_dir, "adapter_model.safetensors"))
                peft_cfg = list(model.peft_config.values())[0]
                vllm_targets = sorted(
                    m for m in (peft_cfg.target_modules or []) if m not in _vllm_unsupported
                )
                adapter_cfg = {
                    "base_model_name_or_path": base_model_name,
                    "bias": peft_cfg.bias,
                    "fan_in_fan_out": False,
                    "inference_mode": True,
                    "init_lora_weights": True,
                    "lora_alpha": peft_cfg.lora_alpha,
                    "lora_dropout": peft_cfg.lora_dropout,
                    "modules_to_save": None,
                    "peft_type": "LORA",
                    "r": peft_cfg.r,
                    "target_modules": vllm_targets,
                    "task_type": "CAUSAL_LM",
                }
                with open(os.path.join(save_dir, "adapter_config.json"), "w") as cf:
                    json.dump(adapter_cfg, cf)

                send_result_fn({"grad_norm": grad_norm, "lora_path": save_dir})

        elif cmd["type"] == "save_checkpoint":
            if rank == 0:
                save_dir = cmd["save_dir"]
                os.makedirs(save_dir, exist_ok=True)
                lora_state = {k: v.cpu().contiguous() for k, v in model.state_dict().items()
                              if "lora_" in k or "modules_to_save" in k}
                save_file(lora_state, os.path.join(save_dir, "adapter_model.safetensors"))
                send_result_fn({"saved": True})

    dist.destroy_process_group()


def worker_main(local_rank, world_size, master_port, args_dict, cmd_queue, result_queue,
                env_overrides=None, env_removals=None):
    """Entry point for mp.Process-based workers (queue IPC)."""
    logging.basicConfig(level=logging.INFO)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    if env_overrides:
        for k, v in env_overrides.items():
            os.environ[k] = v
    if env_removals:
        for k in env_removals:
            os.environ.pop(k, None)

    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(seconds=1800))
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    _use_gloo = os.environ.get("NCCL_P2P_DISABLE") == "1"
    obj_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=86400)) if _use_gloo else None

    if rank == 0:
        logger.info(f"Megatron worker (queue): {world_size} GPUs, model={args_dict['base_model']}")

    start = time.time()
    model, optimizer, tokenizer, lora_params = _init_model(
        args_dict["base_model"], args_dict["lora_rank"], args_dict.get("lora_alpha", 64),
        args_dict.get("gradient_checkpointing", True), device,
    )
    load_time = time.time() - start
    mem_gb = torch.cuda.memory_allocated() / (1024**3)
    if rank == 0:
        logger.info(f"Model loaded: {load_time:.1f}s, {mem_gb:.1f}GB/GPU, {len(lora_params)} LoRA param groups")

    def get_cmd():
        return cmd_queue.get()

    def send_result(result):
        result_queue.put(result)

    _run_command_loop(
        rank, world_size, model, optimizer, tokenizer, lora_params,
        device, obj_group, args_dict.get("micro_batch_size", 4),
        args_dict.get("lora_sync_dir", "/dev/shm/lora_adapters"),
        args_dict["base_model"],
        get_cmd, send_result,
    )


def main():
    """Legacy entry point for torchrun-based workers (file IPC)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--cmd-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--lora-sync-dir", default="/dev/shm/lora_adapters")
    parser.add_argument("--micro-batch-size", type=int, default=4,
                        help="Number of sequences per GPU forward pass (higher = better GPU utilization)")
    args = parser.parse_args()

    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(seconds=1800))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    _use_gloo = os.environ.get("NCCL_P2P_DISABLE") == "1"
    obj_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=86400)) if _use_gloo else None

    if rank == 0:
        logger.info(f"Megatron worker (torchrun): {world_size} GPUs, model={args.base_model}, lora_rank={args.lora_rank}")

    start = time.time()
    model, optimizer, tokenizer, lora_params = _init_model(
        args.base_model, args.lora_rank, args.lora_alpha,
        args.gradient_checkpointing, device,
    )
    load_time = time.time() - start
    mem_gb = torch.cuda.memory_allocated() / (1024**3)
    if rank == 0:
        logger.info(f"Model loaded: {load_time:.1f}s, {mem_gb:.1f}GB/GPU, {len(lora_params)} LoRA param groups")

    def get_cmd():
        while not os.path.exists(args.cmd_file):
            time.sleep(0.05)
        with open(args.cmd_file, "rb") as f:
            cmd = pickle.load(f)
        os.unlink(args.cmd_file)
        return cmd

    def send_result(result):
        with open(args.result_file, "wb") as f:
            pickle.dump(result, f)

    _run_command_loop(
        rank, world_size, model, optimizer, tokenizer, lora_params,
        device, obj_group, args.micro_batch_size,
        args.lora_sync_dir, args.base_model,
        get_cmd, send_result,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
