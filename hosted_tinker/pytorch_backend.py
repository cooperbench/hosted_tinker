"""PyTorch backend for hosted-tinker.

Uses HuggingFace transformers + PEFT LoRA for training.
Supports Qwen3 MoE models with device_map="auto" across multiple GPUs.
No Ray dependency. Single process, all GPUs.

Usage:
    python -m hosted_tinker.api --base-model Qwen/Qwen3-30B-A3B --backend pytorch
"""
from __future__ import annotations

import logging
import os
import tarfile
import tempfile
import time
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from hosted_tinker.backend import AbstractBackend
from hosted_tinker import types

logger = logging.getLogger(__name__)

# Default PPO clip thresholds (matching JAX backend)
_DEFAULT_CLIP_LOW = 0.8
_DEFAULT_CLIP_HIGH = 1.2


class PyTorchBackendConfig(BaseModel, extra="forbid"):
    """Configuration for the PyTorch backend."""

    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing")
    torch_dtype: str = Field(default="bfloat16", description="Model dtype (bfloat16, float16, float32)")
    micro_batch_size: int = Field(default=1, description="Micro-batch size for gradient accumulation")
    loss_chunk_size: int = Field(default=1024, description="Chunk size for logprob computation (0=full)")
    # GPU restriction: limit which GPUs the model uses (for split-GPU mode)
    train_gpus: str = Field(default="", description="GPU IDs for training (e.g., '0,1,2,3'). Empty = all GPUs.")
    # vLLM LoRA sync: after each optim_step, save LoRA weights and reload on vLLM
    vllm_sync_url: str | None = Field(default=None, description="vLLM base URL for LoRA sync (e.g., http://localhost:8001)")
    lora_sync_dir: str = Field(default="/dev/shm/lora_adapters", description="Dir to save LoRA weights for vLLM")


# ---------------------------------------------------------------------------
# Loss functions (port from skyrl/tinker/loss_fns.py)
# ---------------------------------------------------------------------------

def _safe_loss_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute mean of values where mask > 0."""
    masked = values * mask
    total = mask.sum()
    return masked.sum() / total.clamp(min=1.0)


def cross_entropy_loss(
    target_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    _sampling_logprobs: torch.Tensor,
    _advantages: torch.Tensor,
    _config: dict | None,
) -> torch.Tensor:
    return -_safe_loss_mask(target_logprobs, loss_mask)


def importance_sampling_loss(
    target_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    _config: dict | None,
) -> torch.Tensor:
    ratio = torch.exp(target_logprobs - sampling_logprobs)
    return -_safe_loss_mask(ratio * advantages, loss_mask)


def ppo_loss(
    target_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    config: dict | None,
) -> torch.Tensor:
    clip_low = config.get("clip_low_threshold", _DEFAULT_CLIP_LOW) if config else _DEFAULT_CLIP_LOW
    clip_high = config.get("clip_high_threshold", _DEFAULT_CLIP_HIGH) if config else _DEFAULT_CLIP_HIGH
    ratio = torch.exp(target_logprobs - sampling_logprobs)
    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    return -_safe_loss_mask(torch.min(unclipped, clipped), loss_mask)


def cispo_loss(
    target_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    config: dict | None,
) -> torch.Tensor:
    clip_low = config.get("clip_low_threshold", _DEFAULT_CLIP_LOW) if config else _DEFAULT_CLIP_LOW
    clip_high = config.get("clip_high_threshold", _DEFAULT_CLIP_HIGH) if config else _DEFAULT_CLIP_HIGH
    ratio = torch.exp(target_logprobs - sampling_logprobs)
    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
    obj = clipped_ratio.detach() * target_logprobs * advantages
    return -_safe_loss_mask(obj, loss_mask)


LOSS_FN_MAP = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
    "ppo": ppo_loss,
    "cispo": cispo_loss,
}


# ---------------------------------------------------------------------------
# Accumulated gradients (like JAX AccumulatedGradients)
# ---------------------------------------------------------------------------

class AccumulatedGradients:
    """Accumulate gradients across multiple forward_backward calls."""

    def __init__(self, named_params: list[tuple[str, torch.nn.Parameter]]):
        self.grad_sum: dict[str, torch.Tensor] = {}
        self.param_ref: dict[str, torch.nn.Parameter] = {}
        for name, param in named_params:
            self.grad_sum[name] = torch.zeros_like(param.data)
            self.param_ref[name] = param
        self.count = 0

    def accumulate(self):
        """Add current .grad from parameters to accumulated sum."""
        for name, param in self.param_ref.items():
            if param.grad is not None:
                self.grad_sum[name].add_(param.grad.data)
        self.count += 1

    def apply_to_params(self) -> float:
        """Set param.grad = accumulated_mean. Returns grad norm."""
        if self.count == 0:
            return 0.0
        total_norm_sq = 0.0
        for name, param in self.param_ref.items():
            avg_grad = self.grad_sum[name] / self.count
            param.grad = avg_grad
            total_norm_sq += avg_grad.norm().item() ** 2
        return total_norm_sq ** 0.5

    def reset(self):
        for g in self.grad_sum.values():
            g.zero_()
        self.count = 0


# ---------------------------------------------------------------------------
# Logprob computation
# ---------------------------------------------------------------------------

def compute_target_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_ids: torch.Tensor,
    chunk_size: int = 0,
) -> torch.Tensor:
    """Compute log P(target_ids[i] | input_ids[:i+1]) for each position.

    Args:
        model: HuggingFace causal LM
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        target_ids: [batch, seq_len] — token to predict at each position
        chunk_size: If > 0, compute logprobs in chunks to save memory

    Returns:
        target_logprobs: [batch, seq_len] — log prob of each target token
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # [batch, seq_len, vocab]

    if chunk_size > 0 and logits.shape[1] > chunk_size:
        # Compute log_softmax in chunks to avoid materializing full [B, T, V]
        bs, seq_len, vocab = logits.shape
        all_logprobs = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_logits = logits[:, start:end, :]
            chunk_targets = target_ids[:, start:end]
            chunk_lp = F.log_softmax(chunk_logits, dim=-1)
            chunk_target_lp = chunk_lp.gather(-1, chunk_targets.unsqueeze(-1)).squeeze(-1)
            all_logprobs.append(chunk_target_lp)
            del chunk_logits, chunk_lp
        target_logprobs = torch.cat(all_logprobs, dim=1)
    else:
        log_probs = F.log_softmax(logits, dim=-1)
        target_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        del log_probs

    del logits
    return target_logprobs


# ---------------------------------------------------------------------------
# PyTorch Backend
# ---------------------------------------------------------------------------

class PyTorchBackend(AbstractBackend):
    """PyTorch backend using HuggingFace + PEFT LoRA.

    Supports Qwen3 MoE and other HF models.
    Uses device_map="auto" for multi-GPU model parallelism.
    """

    def __init__(self, base_model: str, config: PyTorchBackendConfig):
        self.base_model_name = base_model
        self.config = config
        self.metrics = types.EngineMetrics()

        # Resolve dtype
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

        # Load base model
        # Determine device_map: restrict to train_gpus if specified
        if config.train_gpus.strip():
            gpu_ids = [int(g) for g in config.train_gpus.split(",")]
            # Build max_memory dict to constrain device_map="auto"
            import torch as _torch
            max_memory = {}
            for i in range(_torch.cuda.device_count()):
                if i in gpu_ids:
                    max_memory[i] = f"{int(_torch.cuda.get_device_properties(i).total_memory * 0.9 / 1024**3)}GiB"
                else:
                    max_memory[i] = "0GiB"  # Don't use this GPU
            max_memory["cpu"] = "80GiB"
            logger.info(f"Restricting model to GPUs {gpu_ids}")
        else:
            max_memory = None

        logger.info(f"Loading base model: {base_model}")
        start = time.time()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        except (ValueError, ImportError):
            logger.warning("flash_attention_2 not available, falling back to eager")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        logger.info(f"Model loaded in {time.time() - start:.1f}s")

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Model registry
        self._models: dict[str, types.ModelMetadata] = {}
        self._optimizers: dict[str, torch.optim.AdamW] = {}
        self._accum_grads: dict[str, AccumulatedGradients] = {}
        self._adapter_counter = 0

    def has_model(self, model_id: str) -> bool:
        return model_id in self._models

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType

        # Determine target modules based on lora_config flags and model architecture
        # Auto-detect attention module names (different across model families)
        module_names = {n.split(".")[-1] for n, _ in self.model.named_modules()}

        target_modules = []
        if lora_config.train_attn:
            # Standard Qwen3/Llama: q_proj, k_proj, v_proj, o_proj
            # Qwen3.5 MoE: in_proj_qkv, out_proj (fused attention)
            for mod in ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]:
                if mod in module_names:
                    target_modules.append(mod)
        if lora_config.train_mlp:
            for mod in ["gate_proj", "up_proj", "down_proj"]:
                if mod in module_names:
                    target_modules.append(mod)
        if not target_modules:
            # Fallback: let PEFT auto-detect
            target_modules = "all-linear"

        peft_config = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.rank,
            lora_alpha=int(lora_config.alpha),
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="none",
        )

        # If model already has PEFT, remove it first
        if hasattr(self.model, "peft_config"):
            self.model = self.model.merge_and_unload()
            # Re-freeze
            for param in self.model.parameters():
                param.requires_grad = False

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # Enable input gradients for gradient checkpointing
        self.model.enable_input_require_grads()

        # Create optimizer (lr=0 placeholder, actual lr set in optim_step)
        trainable_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [p for _, p in trainable_params],
            lr=0.0, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
        )

        accum = AccumulatedGradients(trainable_params)

        self._models[model_id] = types.ModelMetadata(
            adapter_index=self._adapter_counter,
            lora_config=lora_config,
        )
        self._optimizers[model_id] = optimizer
        self._accum_grads[model_id] = accum
        self._adapter_counter += 1

        logger.info(f"Created model {model_id}: rank={lora_config.rank}, "
                     f"targets={target_modules}, trainable params={sum(p.numel() for _, p in trainable_params)}")

    def delete_model(self, model_id: str) -> None:
        self._models.pop(model_id, None)
        self._optimizers.pop(model_id, None)
        self._accum_grads.pop(model_id, None)

    def _run_model_pass(
        self,
        prepared_batch: types.PreparedModelPassBatch,
        compute_gradients: bool,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Core implementation for forward and forward_backward."""
        results: dict[str, types.ForwardBackwardOutput | types.ErrorResponse] = {}
        n_examples = len(prepared_batch.all_input_ids)

        if n_examples == 0:
            return results

        # Get the device for inputs (use the device of the first parameter)
        device = next(self.model.parameters()).device

        # Process in micro-batches
        all_logprobs_list: list[list[float]] = []
        all_losses_list: list[list[float]] = []

        micro_bs = self.config.micro_batch_size
        for mb_start in range(0, n_examples, micro_bs):
            mb_end = min(mb_start + micro_bs, n_examples)

            # Pad sequences in this micro-batch to same length
            mb_input_ids = prepared_batch.all_input_ids[mb_start:mb_end]
            mb_targets = prepared_batch.all_targets[mb_start:mb_end]
            mb_weights = prepared_batch.all_token_weights[mb_start:mb_end]
            mb_sampling_lps = prepared_batch.all_sampling_logprobs[mb_start:mb_end]
            mb_advantages = prepared_batch.all_advantages[mb_start:mb_end]
            mb_loss_fns = prepared_batch.all_loss_fns[mb_start:mb_end]
            mb_loss_configs = prepared_batch.all_loss_fn_configs[mb_start:mb_end]
            mb_model_ids = prepared_batch.all_model_ids[mb_start:mb_end]

            max_len = max(len(ids) for ids in mb_input_ids)
            bs = mb_end - mb_start
            pad_id = self.tokenizer.pad_token_id or 0

            # Build padded tensors (right-padded)
            input_ids_t = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
            attn_mask_t = torch.zeros((bs, max_len), dtype=torch.long, device=device)
            target_ids_t = torch.zeros((bs, max_len), dtype=torch.long, device=device)
            weights_t = torch.zeros((bs, max_len), dtype=self.torch_dtype, device=device)
            sampling_lps_t = torch.zeros((bs, max_len), dtype=self.torch_dtype, device=device)
            advantages_t = torch.zeros((bs, max_len), dtype=self.torch_dtype, device=device)

            for i in range(bs):
                seq_len = len(mb_input_ids[i])
                input_ids_t[i, :seq_len] = torch.tensor(mb_input_ids[i], dtype=torch.long)
                attn_mask_t[i, :seq_len] = 1
                target_ids_t[i, :seq_len] = torch.tensor(mb_targets[i][:seq_len], dtype=torch.long)
                weights_t[i, :seq_len] = torch.tensor(mb_weights[i][:seq_len], dtype=self.torch_dtype)
                sampling_lps_t[i, :seq_len] = torch.tensor(mb_sampling_lps[i][:seq_len], dtype=self.torch_dtype)
                advantages_t[i, :seq_len] = torch.tensor(mb_advantages[i][:seq_len], dtype=self.torch_dtype)

            # Forward pass
            if compute_gradients:
                target_logprobs = compute_target_logprobs(
                    self.model, input_ids_t, attn_mask_t, target_ids_t,
                    chunk_size=self.config.loss_chunk_size,
                )
            else:
                with torch.no_grad():
                    target_logprobs = compute_target_logprobs(
                        self.model, input_ids_t, attn_mask_t, target_ids_t,
                        chunk_size=self.config.loss_chunk_size,
                    )

            # Compute loss per example and accumulate gradients
            total_loss = None
            gpu_logprobs = []
            gpu_elem_losses = []
            seq_lens = []
            for i in range(bs):
                seq_len_i = len(mb_input_ids[i])
                seq_lens.append(seq_len_i)

                gpu_logprobs.append(target_logprobs[i, :seq_len_i].detach().float())

                if compute_gradients:
                    loss_fn = LOSS_FN_MAP.get(mb_loss_fns[i], cross_entropy_loss)
                    loss = loss_fn(
                        target_logprobs[i, :seq_len_i],
                        weights_t[i, :seq_len_i],
                        sampling_lps_t[i, :seq_len_i],
                        advantages_t[i, :seq_len_i],
                        mb_loss_configs[i],
                    )
                    total_loss = loss if total_loss is None else total_loss + loss
                    with torch.no_grad():
                        gpu_elem_losses.append(-(target_logprobs[i, :seq_len_i] * weights_t[i, :seq_len_i]).float())

            # Batch transfer GPU → CPU
            if gpu_logprobs:
                cpu_lps = [t.cpu().tolist() for t in gpu_logprobs]
                all_logprobs_list.extend(cpu_lps)
            if compute_gradients and gpu_elem_losses:
                cpu_losses = [t.cpu().tolist() for t in gpu_elem_losses]
                all_losses_list.extend(cpu_losses)
            elif not compute_gradients:
                for sl in seq_lens:
                    all_losses_list.append([0.0] * sl)

            if compute_gradients and total_loss is not None:
                total_loss.backward()
                # Accumulate gradients for all models in this micro-batch
                for i in range(bs):
                    model_id = mb_model_ids[i]
                    if model_id in self._accum_grads:
                        self._accum_grads[model_id].accumulate()
                self.model.zero_grad()

            del input_ids_t, attn_mask_t, target_ids_t, target_logprobs

        # Build per-request results
        for request_id, model_id, start_idx, end_idx in prepared_batch.request_batch_slices:
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                loss_fn_outputs.append({
                    "logprobs": {"data": all_logprobs_list[i], "dtype": "float32"},
                    "elementwise_loss": {"data": all_losses_list[i], "dtype": "float32"},
                })

            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar",
                loss_fn_outputs=loss_fn_outputs,
                metrics={},
            )

        return results

    def forward_backward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        self.model.train()
        return self._run_model_pass(prepared_batch, compute_gradients=True)

    def forward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        self.model.eval()
        return self._run_model_pass(prepared_batch, compute_gradients=False)

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        adam = request_data.adam_params
        optimizer = self._optimizers[model_id]
        accum = self._accum_grads[model_id]

        # Apply accumulated gradients to params
        grad_norm = accum.apply_to_params()

        # Update optimizer hyperparameters
        for pg in optimizer.param_groups:
            pg["lr"] = adam.learning_rate
            pg["betas"] = (adam.beta1, adam.beta2)
            pg["eps"] = adam.eps
            pg["weight_decay"] = adam.weight_decay

        # Clip gradients if needed
        trainable = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

        # Step
        optimizer.step()
        optimizer.zero_grad()
        accum.reset()

        # Sync LoRA weights to vLLM if configured
        if self.config.vllm_sync_url:
            self._sync_lora_to_vllm(model_id)

        return types.OptimStepOutput(metrics={
            "skyrl.ai/grad_norm": grad_norm,
            "skyrl.ai/learning_rate": adam.learning_rate,
        })

    def _sync_lora_to_vllm(self, model_id: str) -> None:
        """Save LoRA weights and reload on vLLM."""
        from hosted_tinker.vllm_manager import save_lora_for_vllm

        adapter_dir = self.config.lora_sync_dir
        save_path = save_lora_for_vllm(self.model, adapter_dir, adapter_name=model_id)

        try:
            # Reload on vLLM
            url = self.config.vllm_sync_url
            requests_mod = __import__("requests")

            # Unload old adapter
            try:
                requests_mod.post(
                    f"{url}/v1/unload_lora_adapter",
                    json={"lora_name": model_id}, timeout=30,
                )
            except Exception:
                pass

            # Load new adapter
            r = requests_mod.post(
                f"{url}/v1/load_lora_adapter",
                json={"lora_name": model_id, "lora_path": os.path.abspath(save_path)},
                timeout=60,
            )
            if r.status_code == 200:
                logger.info(f"Synced LoRA to vLLM: {model_id}")
            else:
                logger.warning(f"vLLM LoRA reload: {r.status_code} {r.text[:200]}")
        except Exception as e:
            logger.warning(f"vLLM LoRA sync failed: {e}")

    def sample(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        # Sampling is handled by external vLLM
        results = {}
        for request_id, model_id, start, end, _ in prepared_batch.request_batch_slices:
            results[request_id] = types.ErrorResponse(
                error="Sampling not supported in PyTorch backend. Use external inference engine.",
                status="error",
            )
        return results

    def save_checkpoint(self, output_path, model_id: str) -> None:
        """Save LoRA weights + optimizer state as tar.gz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save PEFT adapter — use safetensors to avoid torch.save inline_container.cc bug
            adapter_dir = os.path.join(tmpdir, "adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            lora_state = {k: v.cpu().contiguous() for k, v in self.model.state_dict().items()
                          if "lora_" in k or "modules_to_save" in k}
            from safetensors.torch import save_file
            save_file(lora_state, os.path.join(adapter_dir, "adapter_model.safetensors"))
            # Save adapter config
            if hasattr(self.model, "peft_config"):
                import json
                for adapter_name, peft_cfg in self.model.peft_config.items():
                    cfg_dict = peft_cfg.to_dict() if hasattr(peft_cfg, "to_dict") else {}
                    # Convert sets to lists for JSON serialization
                    def _make_serializable(obj):
                        if isinstance(obj, set):
                            return list(obj)
                        if isinstance(obj, dict):
                            return {k: _make_serializable(v) for k, v in obj.items()}
                        if isinstance(obj, (list, tuple)):
                            return [_make_serializable(v) for v in obj]
                        return obj
                    cfg_dict = _make_serializable(cfg_dict)
                    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
                        json.dump(cfg_dict, f)

            # Note: optimizer state is NOT saved (LoRA optimizer state is small but
            # serializing multi-device param groups is problematic). The optimizer
            # state will be re-initialized on load. For production, consider using
            # single-device model or FSDP with proper state dict handling.

            # Create tar in temp, then copy to output_path (which may be cloud)
            tar_path = os.path.join(tmpdir, "checkpoint.tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(adapter_dir, arcname="adapter")

            # Copy to output_path (supports cloud paths via cloudpathlib)
            import shutil
            if hasattr(output_path, "write_bytes"):
                output_path.write_bytes(open(tar_path, "rb").read())
            else:
                shutil.copy2(tar_path, str(output_path))

        logger.info(f"Checkpoint saved to {output_path}")

    def load_checkpoint(self, checkpoint_path, model_id: str) -> None:
        """Load LoRA weights + optimizer state from tar.gz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download from cloud path if needed
            tar_path = os.path.join(tmpdir, "checkpoint.tar.gz")
            if hasattr(checkpoint_path, "read_bytes"):
                with open(tar_path, "wb") as f:
                    f.write(checkpoint_path.read_bytes())
            else:
                import shutil
                shutil.copy2(str(checkpoint_path), tar_path)

            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(tmpdir)

            adapter_dir = os.path.join(tmpdir, "adapter")
            adapter_st = os.path.join(adapter_dir, "adapter_model.safetensors")
            adapter_bin = os.path.join(adapter_dir, "adapter_model.bin")
            if os.path.exists(adapter_st):
                from safetensors.torch import load_file
                lora_state = load_file(adapter_st)
                model_state = self.model.state_dict()
                for k, v in lora_state.items():
                    if k in model_state:
                        model_state[k].copy_(v.to(model_state[k].device))
                logger.info(f"Loaded {len(lora_state)} adapter params (safetensors)")
            elif os.path.exists(adapter_bin):
                lora_state = torch.load(adapter_bin, map_location="cpu", weights_only=True)
                model_state = self.model.state_dict()
                for k, v in lora_state.items():
                    if k in model_state:
                        model_state[k].copy_(v.to(model_state[k].device))
                logger.info(f"Loaded {len(lora_state)} adapter params (bin)")

            # Load optimizer state (try pickle first, then torch.save fallback)
            opt_pkl = os.path.join(tmpdir, "optimizer.pkl")
            opt_pt = os.path.join(tmpdir, "optimizer.pt")
            if os.path.exists(opt_pkl) and model_id in self._optimizers:
                import pickle
                with open(opt_pkl, "rb") as pf:
                    state = pickle.load(pf)
                self._optimizers[model_id].load_state_dict(state)
                logger.info("Loaded optimizer state (pickle)")
            elif os.path.exists(opt_pt) and model_id in self._optimizers:
                state = torch.load(opt_pt, map_location="cpu", weights_only=True)
                self._optimizers[model_id].load_state_dict(state)
                logger.info("Loaded optimizer state (torch)")

    def save_sampler_checkpoint(self, output_path, model_id: str, persist: bool = True) -> None:
        """Save LoRA adapter in PEFT format for vLLM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            if persist:
                adapter_dir = os.path.join(tmpdir, "adapter")
                os.makedirs(adapter_dir, exist_ok=True)
                lora_state = {k: v.cpu().contiguous() for k, v in self.model.state_dict().items()
                              if "lora_" in k or "modules_to_save" in k}
                from safetensors.torch import save_file
                save_file(lora_state, os.path.join(adapter_dir, "adapter_model.safetensors"))
                tar_path = os.path.join(tmpdir, "sampler.tar.gz")
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(adapter_dir, arcname="adapter")
            else:
                tar_path = os.path.join(tmpdir, "sampler.tar.gz")
                with tarfile.open(tar_path, "w:gz") as tar:
                    pass  # Empty marker

            if hasattr(output_path, "write_bytes"):
                output_path.write_bytes(open(tar_path, "rb").read())
            else:
                import shutil
                shutil.copy2(tar_path, str(output_path))

        logger.info(f"Sampler checkpoint {'saved' if persist else 'marker'} to {output_path}")
