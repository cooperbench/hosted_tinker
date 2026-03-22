"""Megatron worker process for distributed training.

Uses PyTorch DDP (not FSDP) with HuggingFace model + PEFT LoRA.
DDP uses all_reduce for gradient sync which works on B200 GPUs
(unlike FSDP which hangs on gather_object).

Each GPU holds a full copy of the model. LoRA params are synced via DDP.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel as DDP
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--cmd-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--lora-sync-dir", default="/dev/shm/lora_adapters")
    args = parser.parse_args()

    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(seconds=1800))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    if rank == 0:
        logger.info(f"Megatron worker: {world_size} GPUs, model={args.base_model}, lora_rank={args.lora_rank}")

    start = time.time()

    # Each rank loads the full model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Apply LoRA
    module_names = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [m for m in ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj",
                           "gate_proj", "up_proj", "down_proj"] if m in module_names]
    peft_config = PeftLoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
                                 lora_alpha=args.lora_alpha, target_modules=targets, bias="none")
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # NOTE: DDP doesn't work with MoE (unused experts cause issues):
    # - find_unused_parameters=True: uses broadcast which hangs on B200
    # - find_unused_parameters=False: OOM trying to broadcast MoE expert buffers
    # Instead, use manual gradient all_reduce after backward pass.
    # model stays unwrapped (no DDP).

    # Optimizer
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=0.0, betas=(0.9, 0.95), eps=1e-8)

    load_time = time.time() - start
    mem_gb = torch.cuda.memory_allocated() / (1024**3)
    if rank == 0:
        logger.info(f"Model loaded: {load_time:.1f}s, {mem_gb:.1f}GB/GPU, {len(lora_params)} LoRA param groups")

    pad_id = tokenizer.pad_token_id or 0
    accum_count = 0

    # Command loop
    while True:
        cmd_data = [None]
        if rank == 0:
            while not os.path.exists(args.cmd_file):
                time.sleep(0.05)
            with open(args.cmd_file, "rb") as f:
                cmd_data[0] = pickle.load(f)
            os.unlink(args.cmd_file)

        dist.broadcast_object_list(cmd_data, src=0)
        cmd = cmd_data[0]

        if cmd["type"] == "shutdown":
            break

        elif cmd["type"] in ("forward_backward", "forward"):
            compute_grad = cmd["type"] == "forward_backward"
            batch = cmd["batch"]
            n_examples = len(batch["all_input_ids"])

            # Split examples across ranks (data parallelism)
            per_rank = (n_examples + world_size - 1) // world_size
            my_start = rank * per_rank
            my_end = min(my_start + per_rank, n_examples)

            all_lp = [None] * n_examples
            all_loss = [None] * n_examples

            if compute_grad:
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()

            for idx in range(my_start, my_end):
                ids = torch.tensor([batch["all_input_ids"][idx]], dtype=torch.long, device=device)
                tgt = torch.tensor([batch["all_targets"][idx][:len(batch["all_input_ids"][idx])]],
                                   dtype=torch.long, device=device)
                wt = torch.tensor([batch["all_token_weights"][idx][:len(batch["all_input_ids"][idx])]],
                                  dtype=torch.bfloat16, device=device)
                slp = torch.tensor([batch["all_sampling_logprobs"][idx][:len(batch["all_input_ids"][idx])]],
                                   dtype=torch.bfloat16, device=device)
                adv = torch.tensor([batch["all_advantages"][idx][:len(batch["all_input_ids"][idx])]],
                                   dtype=torch.bfloat16, device=device)

                if compute_grad:
                    out = model(input_ids=ids, use_cache=False)
                else:
                    with torch.no_grad():
                        out = model(input_ids=ids, use_cache=False)

                logits = out.logits
                log_probs = F.log_softmax(logits, dim=-1)
                target_lp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                del logits, log_probs

                all_lp[idx] = target_lp[0].detach().float().cpu().tolist()

                if compute_grad:
                    loss_fn = LOSS_FN.get(batch["all_loss_fns"][idx], ce_loss)
                    loss = loss_fn(target_lp[0], wt[0], slp[0], adv[0], batch["all_loss_fn_configs"][idx])
                    all_loss[idx] = [0.0] * len(batch["all_input_ids"][idx])
                    loss.backward()
                    accum_count += 1
                else:
                    all_loss[idx] = [0.0] * len(batch["all_input_ids"][idx])

                del out, target_lp

            # Gather results to rank 0.
            # NOTE: all_gather_object / gather_object HANG on B200.
            # Use tensor-based gather: serialize per-rank results to a byte tensor,
            # all_gather the byte tensors, then deserialize on rank 0.
            my_result_bytes = pickle.dumps(
                {i: (all_lp[i], all_loss[i]) for i in range(my_start, my_end) if all_lp[i] is not None}
            )
            # Pad all byte tensors to same length
            local_size = torch.tensor([len(my_result_bytes)], device=device, dtype=torch.long)
            all_sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)
            max_size = max(s.item() for s in all_sizes)

            padded = torch.zeros(max_size, device=device, dtype=torch.uint8)
            padded[:len(my_result_bytes)] = torch.tensor(list(my_result_bytes), dtype=torch.uint8, device=device)
            gathered_tensors = [torch.zeros(max_size, device=device, dtype=torch.uint8) for _ in range(world_size)]
            dist.all_gather(gathered_tensors, padded)

            if rank == 0:
                merged_lp = [None] * n_examples
                merged_loss = [None] * n_examples
                for r_idx in range(world_size):
                    sz = all_sizes[r_idx].item()
                    rank_data = pickle.loads(gathered_tensors[r_idx][:sz].cpu().numpy().tobytes())
                    for i, (lp, ls) in rank_data.items():
                        merged_lp[i] = lp
                        merged_loss[i] = ls

                with open(args.result_file, "wb") as f:
                    pickle.dump({"logprobs": merged_lp, "losses": merged_loss}, f)

        elif cmd["type"] == "optim_step":
            adam = cmd["adam_params"]
            for pg in optimizer.param_groups:
                pg["lr"] = adam["learning_rate"]
                pg["betas"] = (adam["beta1"], adam["beta2"])
                pg["eps"] = adam["eps"]
                pg["weight_decay"] = adam["weight_decay"]

            # Manual gradient all_reduce (since we don't use DDP)
            trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if world_size > 1:
                for p in trainable:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item() if trainable else 0.0

            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

            # Extract and save LoRA weights on rank 0
            if rank == 0:
                lora_state = {k: v.cpu().contiguous() for k, v in model.state_dict().items()
                              if "lora_" in k or "modules_to_save" in k}
                save_dir = os.path.join(args.lora_sync_dir, "adapter")
                os.makedirs(save_dir, exist_ok=True)
                save_file(lora_state, os.path.join(save_dir, "adapter_model.safetensors"))

                with open(args.result_file, "wb") as f:
                    pickle.dump({"grad_norm": grad_norm, "lora_path": save_dir}, f)

        elif cmd["type"] == "save_checkpoint":
            if rank == 0:
                save_dir = cmd["save_dir"]
                os.makedirs(save_dir, exist_ok=True)
                lora_state = {k: v.cpu().contiguous() for k, v in model.state_dict().items()
                              if "lora_" in k or "modules_to_save" in k}
                save_file(lora_state, os.path.join(save_dir, "adapter_model.safetensors"))
                with open(args.result_file, "wb") as f:
                    pickle.dump({"saved": True}, f)

        # NOTE: skip dist.barrier() — it intermittently hangs on B200.
        # The broadcast_object_list at the top of each loop iteration
        # provides sufficient synchronization.

    dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
