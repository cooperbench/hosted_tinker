"""FSDP2 worker process for distributed training.

Launched via mp.Process by FSDP2Backend (queue-based IPC), or via torchrun
(file-based IPC fallback). All ranks participate in collectives.
Rank 0 receives commands via queue or file and broadcasts to other ranks.

Supports H100 and B200 GPUs:
- H100: Uses NCCL for all collectives (broadcast_object_list, gather_object)
- B200: Uses Gloo group for object collectives (NCCL hangs on Blackwell)

Usage (internal, called by FSDP2Backend via mp.Process — preferred):
    worker_main(local_rank, world_size, master_port, args_dict, cmd_queue, result_queue, ...)

Legacy usage (torchrun fallback):
    torchrun --nproc_per_node=N hosted_tinker/fsdp2_worker.py \
        --base-model MODEL --lora-rank R --cmd-file /tmp/cmd.pkl --result-file /tmp/result.pkl
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Loss functions (same as pytorch_backend.py)
_DEFAULT_CLIP_LOW = 0.8
_DEFAULT_CLIP_HIGH = 1.2


def _safe_loss_mask(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def cross_entropy_loss(lp, mask, _slp, _adv, _cfg):
    return -_safe_loss_mask(lp, mask)


def importance_sampling_loss(lp, mask, slp, adv, _cfg):
    return -_safe_loss_mask(torch.exp(lp - slp) * adv, mask)


def ppo_loss(lp, mask, slp, adv, cfg):
    cl = (cfg or {}).get("clip_low_threshold", _DEFAULT_CLIP_LOW)
    ch = (cfg or {}).get("clip_high_threshold", _DEFAULT_CLIP_HIGH)
    ratio = torch.exp(lp - slp)
    return -_safe_loss_mask(torch.min(ratio * adv, torch.clamp(ratio, cl, ch) * adv), mask)


def cispo_loss(lp, mask, slp, adv, cfg):
    cl = (cfg or {}).get("clip_low_threshold", _DEFAULT_CLIP_LOW)
    ch = (cfg or {}).get("clip_high_threshold", _DEFAULT_CLIP_HIGH)
    ratio = torch.exp(lp - slp)
    return -_safe_loss_mask(torch.clamp(ratio, cl, ch).detach() * lp * adv, mask)


LOSS_FN_MAP = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
    "ppo": ppo_loss,
    "cispo": cispo_loss,
}


def _get_transformer_layer_cls(model):
    """Auto-detect transformer layer class for FSDP wrapping."""
    layer_classes = set()
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if any(p in cls_name for p in ["DecoderLayer", "TransformerBlock", "EncoderLayer"]):
            layer_classes.add(type(module))
    return layer_classes


def _init_model(base_model, lora_rank, lora_alpha, lora_targets, gradient_checkpointing, device, remove_padding):
    """Load model, apply LoRA, wrap with FSDP. Returns (model, optimizer, tokenizer, lora_params)."""
    if remove_padding:
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            raise RuntimeError("remove_padding requires flash_attn to be installed")
        attn_impl = "flash_attention_2"
    else:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Apply LoRA
    module_names = {n.split(".")[-1] for n, _ in model.named_modules()}
    if lora_targets == "auto":
        targets = []
        for mod in [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "out_proj",
            "gate_proj", "up_proj", "down_proj",
        ]:
            if mod in module_names:
                targets.append(mod)
    else:
        targets = lora_targets.split(",")

    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = model.to(dtype=torch.bfloat16, device=device)

    # FSDP wrap
    from functools import partial

    layer_classes = _get_transformer_layer_cls(model)
    if layer_classes:
        auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_classes)
    else:
        auto_wrap = None

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        limit_all_gathers=True,
    )

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=0.0, betas=(0.9, 0.95), eps=1e-8)

    return model, optimizer, tokenizer, lora_params


def _run_command_loop(rank, world_size, model, optimizer, tokenizer, lora_params,
                      device, obj_group, remove_padding, default_mbs, lora_sync_dir,
                      get_cmd_fn, send_result_fn):
    """Shared command loop for both queue-based and file-based IPC."""
    accum_count = 0
    pad_id = tokenizer.pad_token_id or 0

    while True:
        # Rank 0 reads command
        cmd_data = [None]
        if rank == 0:
            cmd_data[0] = get_cmd_fn()

        # Broadcast command to all ranks
        dist.broadcast_object_list(cmd_data, src=0, group=obj_group)
        cmd = cmd_data[0]
        cmd_type = cmd["type"]

        if cmd_type == "shutdown":
            break

        elif cmd_type in ("forward_backward", "forward"):
            compute_grad = cmd_type == "forward_backward"
            batch = cmd["batch"]

            all_input_ids = batch["all_input_ids"]
            all_targets = batch["all_targets"]
            all_weights = batch["all_token_weights"]
            all_slp = batch["all_sampling_logprobs"]
            all_adv = batch["all_advantages"]
            all_loss_fns = batch["all_loss_fns"]
            all_loss_configs = batch["all_loss_fn_configs"]
            n_examples = len(all_input_ids)

            all_logprobs = [None] * n_examples
            all_losses = [None] * n_examples

            if compute_grad:
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()

            micro_batch_size = cmd.get("micro_batch_size", default_mbs)

            sorted_indices = sorted(range(n_examples), key=lambda i: len(all_input_ids[i]))
            my_indices = sorted_indices[rank::world_size]
            my_indices.sort(key=lambda i: len(all_input_ids[i]))

            # All ranks must have the same number of micro-batches for FSDP
            # collectives. Compute max deterministically (no collective needed
            # since all ranks see the same n_examples/world_size/mbs).
            max_per_rank = (n_examples + world_size - 1) // world_size
            max_n_mbs = max(1, (max_per_rank + micro_batch_size - 1) // micro_batch_size)

            mb_starts = list(range(0, len(my_indices), micro_batch_size))
            while len(mb_starts) < max_n_mbs:
                mb_starts.append(None)

            for mb_start in mb_starts:
                is_dummy = mb_start is None
                if is_dummy:
                    mb_indices = []
                    seqs = [[pad_id]]
                else:
                    mb_indices = my_indices[mb_start : mb_start + micro_batch_size]
                    seqs = [all_input_ids[i] for i in mb_indices]

                if remove_padding and not is_dummy:
                    # ── Packed path (flash_attention_2) ──
                    seq_lens_mb = [len(s) for s in seqs]
                    flat_ids = []
                    flat_pos = []
                    for s in seqs:
                        flat_ids.extend(s)
                        flat_pos.extend(range(len(s)))
                    input_ids = torch.tensor(flat_ids, dtype=torch.long, device=device).unsqueeze(0)
                    position_ids = torch.tensor(flat_pos, dtype=torch.long, device=device).unsqueeze(0)

                    if compute_grad:
                        out = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
                    else:
                        with torch.no_grad():
                            out = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)

                    log_probs = F.log_softmax(out.logits[0], dim=-1)
                    del out

                    total_loss = None
                    offset = 0
                    gpu_lps = []
                    gpu_losses = []
                    for j, idx in enumerate(mb_indices):
                        sl = seq_lens_mb[j]
                        tgt = torch.tensor(all_targets[idx][:sl], dtype=torch.long, device=device)
                        target_lp = log_probs[offset : offset + sl].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

                        gpu_lps.append((idx, target_lp.detach().float()))

                        if compute_grad:
                            wt = torch.tensor(all_weights[idx][:sl], dtype=torch.bfloat16, device=device)
                            slp_t = torch.tensor(all_slp[idx][:sl], dtype=torch.bfloat16, device=device)
                            adv_t = torch.tensor(all_adv[idx][:sl], dtype=torch.bfloat16, device=device)
                            loss_fn = LOSS_FN_MAP.get(all_loss_fns[idx], cross_entropy_loss)
                            loss_j = loss_fn(target_lp, wt, slp_t, adv_t, all_loss_configs[idx])
                            gpu_losses.append((idx, (-(target_lp * wt)).detach().float()))
                            total_loss = loss_j if total_loss is None else total_loss + loss_j
                        else:
                            all_losses[idx] = [0.0] * sl
                        offset += sl

                    for idx, t in gpu_lps:
                        all_logprobs[idx] = t.cpu().tolist()
                    for idx, t in gpu_losses:
                        all_losses[idx] = t.cpu().tolist()

                    del log_probs, input_ids, position_ids
                else:
                    # ── Padded path ──
                    max_len = max(len(s) for s in seqs)
                    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
                    attn_mask = torch.zeros(len(seqs), max_len, dtype=torch.long, device=device)
                    for j, seq in enumerate(seqs):
                        input_ids[j, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                        attn_mask[j, : len(seq)] = 1

                    if compute_grad:
                        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                    else:
                        with torch.no_grad():
                            out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

                    if is_dummy:
                        # Dummy micro-batch: still need backward for FSDP
                        # reduce-scatter if compute_grad is True
                        if compute_grad:
                            dummy_loss = out.logits.sum() * 0.0
                            dummy_loss.backward()
                        del out, input_ids
                        continue

                    log_probs = F.log_softmax(out.logits, dim=-1)
                    del out

                    total_loss = None
                    gpu_lps = []
                    gpu_losses = []
                    for j, idx in enumerate(mb_indices):
                        seq_len = len(all_input_ids[idx])
                        tgt = torch.tensor(all_targets[idx][:seq_len], dtype=torch.long, device=device)
                        target_lp = log_probs[j, :seq_len].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

                        gpu_lps.append((idx, target_lp.detach().float()))

                        if compute_grad:
                            wt = torch.tensor(all_weights[idx][:seq_len], dtype=torch.bfloat16, device=device)
                            slp_t = torch.tensor(all_slp[idx][:seq_len], dtype=torch.bfloat16, device=device)
                            adv_t = torch.tensor(all_adv[idx][:seq_len], dtype=torch.bfloat16, device=device)
                            loss_fn = LOSS_FN_MAP.get(all_loss_fns[idx], cross_entropy_loss)
                            loss_j = loss_fn(target_lp, wt, slp_t, adv_t, all_loss_configs[idx])
                            gpu_losses.append((idx, (-(target_lp * wt)).detach().float()))
                            total_loss = loss_j if total_loss is None else total_loss + loss_j
                        else:
                            all_losses[idx] = [0.0] * seq_len

                    for idx, t in gpu_lps:
                        all_logprobs[idx] = t.cpu().tolist()
                    for idx, t in gpu_losses:
                        all_losses[idx] = t.cpu().tolist()

                    del log_probs, input_ids

                if compute_grad and not is_dummy and total_loss is not None:
                    (total_loss / world_size).backward()

            if compute_grad:
                accum_count += 1

            # Gather all logprobs to rank 0
            gathered = [None] * world_size
            dist.gather_object(
                {i: (all_logprobs[i], all_losses[i]) for i in my_indices if all_logprobs[i] is not None},
                gathered if rank == 0 else None,
                dst=0,
                group=obj_group,
            )

            if rank == 0:
                merged_lp = [None] * n_examples
                merged_loss = [None] * n_examples
                for rank_data in gathered:
                    if rank_data:
                        for i, (lp, ls) in rank_data.items():
                            merged_lp[i] = lp
                            merged_loss[i] = ls

                send_result_fn({"logprobs": merged_lp, "losses": merged_loss})

        elif cmd_type == "optim_step":
            adam = cmd["adam_params"]

            for pg in optimizer.param_groups:
                pg["lr"] = adam["learning_rate"]
                pg["betas"] = (adam["beta1"], adam["beta2"])
                pg["eps"] = adam["eps"]
                pg["weight_decay"] = adam["weight_decay"]

            grad_norm = model.clip_grad_norm_(1.0).item()

            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
                state = model.state_dict()

            if rank == 0:
                lora_state = {k: v.contiguous() for k, v in state.items() if "lora_" in k or "modules_to_save" in k}

                save_dir = os.path.join(lora_sync_dir, "adapter")
                os.makedirs(save_dir, exist_ok=True)
                save_file(lora_state, os.path.join(save_dir, "adapter_model.safetensors"))

                send_result_fn({"grad_norm": grad_norm, "lora_path": save_dir})

            del state

        elif cmd_type == "save_checkpoint":
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
                state = model.state_dict()

            if rank == 0:
                lora_state = {k: v.contiguous() for k, v in state.items() if "lora_" in k or "modules_to_save" in k}
                save_dir = cmd["save_dir"]
                os.makedirs(save_dir, exist_ok=True)
                save_file(lora_state, os.path.join(save_dir, "adapter_model.safetensors"))
                send_result_fn({"saved": True})

        dist.barrier()

    dist.destroy_process_group()


def worker_main(local_rank, world_size, master_port, args_dict, cmd_queue, result_queue,
                env_overrides=None, env_removals=None):
    """Entry point for mp.Process-based workers (queue IPC)."""
    logging.basicConfig(level=logging.INFO)

    # Set environment for distributed
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

    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    os.environ.pop("TORCH_NCCL_ASYNC_ERROR_HANDLING", None)
    from datetime import timedelta

    dist.init_process_group("nccl", timeout=timedelta(seconds=1800))
    rank = dist.get_rank()

    # On B200, NCCL object collectives hang — use Gloo fallback.
    # On H100/A100, NCCL works fine — use default (None) for speed.
    _use_gloo = os.environ.get("NCCL_P2P_DISABLE") == "1"
    obj_group = dist.new_group(backend="gloo") if _use_gloo else None

    if rank == 0:
        logger.info(f"FSDP2 worker (queue): {world_size} GPUs, model={args_dict['base_model']}, gloo={'yes' if _use_gloo else 'no'}")

    remove_padding = args_dict.get("remove_padding", False)
    if remove_padding and rank == 0:
        logger.info("remove_padding=True: packing sequences with flash_attention_2")

    start = time.time()
    model, optimizer, tokenizer, lora_params = _init_model(
        args_dict["base_model"], args_dict["lora_rank"], args_dict["lora_alpha"],
        args_dict.get("lora_targets", "auto"), args_dict.get("gradient_checkpointing", True),
        device, remove_padding,
    )
    load_time = time.time() - start
    mem_gb = torch.cuda.memory_allocated() / (1024**3)
    if rank == 0:
        logger.info(f"Model loaded: {load_time:.1f}s, {mem_gb:.1f}GB/GPU, {len(lora_params)} LoRA params")

    # Queue-based IPC: rank 0 gets from queue, puts to result queue
    def get_cmd():
        return cmd_queue.get()

    def send_result(result):
        result_queue.put(result)

    _run_command_loop(
        rank, world_size, model, optimizer, tokenizer, lora_params,
        device, obj_group, remove_padding, args_dict.get("micro_batch_size", 1),
        args_dict.get("lora_sync_dir", "/dev/shm/lora_adapters"),
        get_cmd, send_result,
    )


def main():
    """Legacy entry point for torchrun-based workers (file IPC)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-targets", type=str, default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--cmd-file", required=True, help="Path to command pickle file")
    parser.add_argument("--result-file", required=True, help="Path to result pickle file")
    parser.add_argument("--lora-sync-dir", default="/dev/shm/lora_adapters")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Number of sequences per GPU forward pass")
    parser.add_argument(
        "--remove-padding",
        action="store_true",
        default=False,
        help="Pack sequences to remove padding waste (requires flash_attn)",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    os.environ.pop("TORCH_NCCL_ASYNC_ERROR_HANDLING", None)
    from datetime import timedelta

    dist.init_process_group("nccl", timeout=timedelta(seconds=1800))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    _use_gloo = os.environ.get("NCCL_P2P_DISABLE") == "1"
    obj_group = dist.new_group(backend="gloo") if _use_gloo else None

    if rank == 0:
        logger.info(f"FSDP2 worker (torchrun): {world_size} GPUs, model={args.base_model}, rank={args.lora_rank}")

    remove_padding = args.remove_padding
    if remove_padding and rank == 0:
        logger.info("remove_padding=True: packing sequences with flash_attention_2")

    start = time.time()
    model, optimizer, tokenizer, lora_params = _init_model(
        args.base_model, args.lora_rank, args.lora_alpha,
        args.lora_targets, args.gradient_checkpointing,
        device, remove_padding,
    )
    load_time = time.time() - start
    mem_gb = torch.cuda.memory_allocated() / (1024**3)
    if rank == 0:
        logger.info(f"Model loaded: {load_time:.1f}s, {mem_gb:.1f}GB/GPU, {len(lora_params)} LoRA params")

    # File-based IPC: rank 0 reads from file, writes to file
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
        device, obj_group, remove_padding, args.micro_batch_size,
        args.lora_sync_dir,
        get_cmd, send_result,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
