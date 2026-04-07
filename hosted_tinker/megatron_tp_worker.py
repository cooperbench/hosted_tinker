"""Megatron-Core TP worker using AutoBridge for HF model conversion.

Uses Megatron-Core's native tensor parallelism (not DDP) for training.
Supports Qwen3-30B-A3B via Megatron Bridge.

Key advantage: real TP splits model across GPUs (1/N weights per GPU)
instead of replicating the full model on each GPU.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import tempfile
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def _atomic_pickle(obj: object, path: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
        os.rename(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


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

    # On B200, NCCL object collectives hang (pytorch#165727). Use Gloo fallback.
    # On H100/A100, NCCL works fine — use default process group.
    _use_gloo = os.environ.get("NCCL_P2P_DISABLE") == "1"
    obj_group = dist.new_group(backend="gloo") if _use_gloo else None

    if rank == 0:
        logger.info(f"Megatron TP worker: {world_size} GPUs, model={args.base_model}")

    start = time.time()

    # Initialize Megatron parallel state
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
    model_parallel_cuda_manual_seed(42)

    # Load model via AutoBridge
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(args.base_model, trust_remote_code=True)
    provider = bridge.to_megatron_provider()

    # Check KV heads constraint: TP must be <= num_key_value_heads
    hf_config = provider._config if hasattr(provider, '_config') else None
    if hf_config and hasattr(hf_config, 'num_key_value_heads'):
        max_tp = hf_config.num_key_value_heads
        if world_size > max_tp:
            raise ValueError(
                f"TP={world_size} exceeds num_key_value_heads={max_tp} for {args.base_model}. "
                f"Use --nproc_per_node={max_tp} or fewer."
            )

    provider.tensor_model_parallel_size = world_size
    provider.pipeline_model_parallel_size = 1
    # For MoE models, set expert parallelism
    if hasattr(provider, 'expert_model_parallel_size'):
        provider.expert_model_parallel_size = 1
    provider.finalize()

    try:
        models = provider.provide_distributed_model(wrap_with_ddp=False)
        model = models[0] if isinstance(models, list) else models
        if model is None:
            raise RuntimeError("Model provider returned None — architecture not fully supported in Megatron-Core")
        model = model.to(torch.bfloat16)
    except Exception as e:
        if rank == 0:
            logger.error(f"Megatron model creation failed: {e}")
            logger.error(f"This model may not be supported by Megatron Bridge. "
                         f"Use --backend pytorch (PEFT) or --backend ddp instead.")
        raise

    if args.gradient_checkpointing:
        # Megatron's activation checkpointing
        from megatron.core.transformer.transformer_config import TransformerConfig
        # Enable through model config if available
        pass  # TODO: Enable recompute via config

    # Apply LoRA via Megatron Bridge PEFT
    try:
        from megatron.bridge.peft.lora import LoRA
        lora = LoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            dim=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=0.0,
        )
        model = lora(model)  # __call__ transforms the model
        if rank == 0:
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in model.parameters())
            logger.info(f"LoRA applied: {n_trainable}/{n_total} trainable params")
    except Exception as e:
        if rank == 0:
            logger.warning(f"LoRA not available ({e}), training all params")

    # Optimizer for trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=0.0, betas=(0.9, 0.95), eps=1e-8)

    load_time = time.time() - start
    mem_gb = torch.cuda.memory_allocated() / (1024**3)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded: {load_time:.1f}s, {mem_gb:.1f}GB/GPU, {n_params} total params")

    # Get tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or 0

    # Command loop
    while True:
        cmd_data = [None]
        if rank == 0:
            while not os.path.exists(args.cmd_file):
                time.sleep(0.05)
            with open(args.cmd_file, "rb") as f:
                cmd_data[0] = pickle.load(f)
            os.unlink(args.cmd_file)

        dist.broadcast_object_list(cmd_data, src=0, group=obj_group)
        cmd = cmd_data[0]

        if cmd["type"] == "shutdown":
            break

        elif cmd["type"] in ("forward_backward", "forward"):
            compute_grad = cmd["type"] == "forward_backward"
            batch = cmd["batch"]
            n_examples = len(batch["all_input_ids"])

            # In TP mode, ALL ranks process ALL examples (model is split, not data)
            all_lp = []
            all_loss = []

            if compute_grad:
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()

            for idx in range(n_examples):
                input_ids = torch.tensor([batch["all_input_ids"][idx]], dtype=torch.long, device=device)
                target_ids = torch.tensor([batch["all_targets"][idx][:len(batch["all_input_ids"][idx])]],
                                          dtype=torch.long, device=device)
                position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

                if compute_grad:
                    output = model(input_ids, position_ids, None)
                else:
                    with torch.no_grad():
                        output = model(input_ids, position_ids, None)

                # output shape: [batch, seq, vocab] (parallel_output=False gathers across TP)
                logits = output
                log_probs = F.log_softmax(logits.float(), dim=-1)
                target_lp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                del logits, log_probs

                lp_cpu = target_lp[0].detach().float().cpu().tolist()
                all_lp.append(lp_cpu)

                if compute_grad:
                    wt = torch.tensor(batch["all_token_weights"][idx][:len(batch["all_input_ids"][idx])],
                                      dtype=torch.bfloat16, device=device)
                    loss = -(target_lp[0] * wt).sum() / wt.sum().clamp(min=1.0)
                    all_loss.append([0.0] * len(batch["all_input_ids"][idx]))
                    loss.backward()
                else:
                    all_loss.append([0.0] * len(batch["all_input_ids"][idx]))

                del output, target_lp

            # Only rank 0 writes result (all ranks have same logprobs due to TP gather)
            if rank == 0:
                _atomic_pickle({"logprobs": all_lp, "losses": all_loss}, args.result_file)

        elif cmd["type"] == "optim_step":
            adam = cmd["adam_params"]
            for pg in optimizer.param_groups:
                pg["lr"] = adam["learning_rate"]
                pg["betas"] = (adam["beta1"], adam["beta2"])
                pg["eps"] = adam["eps"]
                pg["weight_decay"] = adam["weight_decay"]

            trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item() if trainable else 0.0

            optimizer.step()
            optimizer.zero_grad()

            # Save LoRA/adapter weights on rank 0
            if rank == 0:
                # Extract trainable param state dict
                adapter_state = {n: p.cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}
                save_dir = os.path.join(args.lora_sync_dir, "adapter")
                os.makedirs(save_dir, exist_ok=True)
                save_file(adapter_state, os.path.join(save_dir, "adapter_model.safetensors"))

                _atomic_pickle({"grad_norm": grad_norm, "lora_path": save_dir}, args.result_file)

        elif cmd["type"] == "save_checkpoint":
            if rank == 0:
                save_dir = cmd["save_dir"]
                os.makedirs(save_dir, exist_ok=True)
                adapter_state = {n: p.cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}
                save_file(adapter_state, os.path.join(save_dir, "adapter_model.safetensors"))
                _atomic_pickle({"saved": True}, args.result_file)

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
