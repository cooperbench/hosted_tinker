"""FSDP2 backend for hosted-tinker.

Uses mp.Process (spawn) to launch N worker processes with PyTorch FSDP2 for
data-parallel training. Communicates with workers via multiprocessing queues
(metadata + batch data through queue, results back through queue).

Supports Qwen3 MoE and Qwen3.5 models.
"""

from __future__ import annotations

import logging
import os
import random
import tempfile
import time

import subprocess

import torch.multiprocessing as mp

from pydantic import BaseModel, Field

from hosted_tinker.backend import AbstractBackend
from hosted_tinker import types


def _detect_gpu_type() -> str:
    """Detect GPU type via nvidia-smi. Returns 'H100', 'B200', 'A100', or 'unknown'."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            name = result.stdout.strip().split("\n")[0].upper()
            for family in ("H100", "B200", "A100"):
                if family in name:
                    return family
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"

logger = logging.getLogger(__name__)


class FSDP2BackendConfig(BaseModel, extra="forbid"):
    """Configuration for the FSDP2 backend."""

    n_train_gpus: int = Field(default=6, description="Number of GPUs for training")
    train_gpu_offset: int = Field(default=2, description="First GPU index for training (skip vLLM GPUs)")
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing")
    loss_chunk_size: int = Field(default=1024, description="Chunk size for logprob computation")
    micro_batch_size: int = Field(
        default=1, description="Sequences per GPU forward pass (higher = better utilization but more memory)"
    )
    remove_padding: bool = Field(
        default=False, description="Pack sequences to remove padding waste (requires flash_attn)"
    )
    # vLLM LoRA sync
    vllm_sync_url: str | None = Field(default=None, description="vLLM base URL for LoRA sync")
    lora_sync_dir: str = Field(default="/dev/shm/lora_adapters", description="Dir for LoRA weight sync")


class FSDP2Backend(AbstractBackend):
    """FSDP2 backend using mp.Process workers with queue-based IPC."""

    def __init__(self, base_model: str, config: FSDP2BackendConfig):
        self.base_model_name = base_model
        self.config = config
        self.metrics = types.EngineMetrics()

        self._models: dict[str, types.ModelMetadata] = {}
        self._worker_processes: list[mp.Process] = []
        self._adapter_counter = 0
        self._micro_batch_size = config.micro_batch_size
        self._mbs_file = "/dev/shm/hosted_tinker_mbs"

        # Queue-based IPC
        self._mp_ctx = mp.get_context("spawn")
        self._cmd_queue: mp.Queue = self._mp_ctx.Queue()
        self._result_queue: mp.Queue = self._mp_ctx.Queue()

        # Keep a log dir for worker output
        self._ipc_dir = tempfile.mkdtemp(prefix="fsdp2_ipc_")

    def has_model(self, model_id: str) -> bool:
        return model_id in self._models

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        if self._worker_processes:
            self._shutdown_workers()

        gpu_ids = list(range(self.config.train_gpu_offset, self.config.train_gpu_offset + self.config.n_train_gpus))
        master_port = random.randint(29500, 29999)

        env_overrides = {
            "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids),
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "HF_HUB_OFFLINE": "1",
        }
        env_removals = []

        # GCP VMs set NCCL_NET=gIB for multi-node clusters but
        # libibverbs.so is not installed on single-node VMs.
        # Also strip /usr/local/gib from LD_LIBRARY_PATH — the gIB shim
        # plugin loads via LD path even with NCCL_NET_PLUGIN="" and crashes
        # when libibverbs.so is missing.
        env_overrides["NCCL_NET_PLUGIN"] = ""
        env_removals.append("NCCL_NET")
        raw_ldpath = os.environ.get("LD_LIBRARY_PATH", "")
        fixed_ldpath = ":".join(p for p in raw_ldpath.split(":") if "gib" not in p.lower() and p)
        env_overrides["LD_LIBRARY_PATH"] = fixed_ldpath

        # GPU-specific NCCL config
        gpu_type = _detect_gpu_type()
        if gpu_type == "B200":
            env_overrides["NCCL_P2P_DISABLE"] = "1"
        else:
            env_removals.append("NCCL_P2P_DISABLE")

        worker_args = {
            "base_model": self.base_model_name,
            "lora_rank": lora_config.rank,
            "lora_alpha": int(lora_config.alpha),
            "lora_targets": "auto",
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "lora_sync_dir": self.config.lora_sync_dir,
            "micro_batch_size": self.config.micro_batch_size,
            "remove_padding": self.config.remove_padding,
        }

        from hosted_tinker.fsdp2_worker import worker_main

        logger.info(f"Launching FSDP2 workers on GPUs {gpu_ids} (port {master_port})")

        self._worker_processes = []
        for local_rank in range(self.config.n_train_gpus):
            p = self._mp_ctx.Process(
                target=worker_main,
                args=(
                    local_rank,
                    self.config.n_train_gpus,
                    master_port,
                    worker_args,
                    self._cmd_queue,
                    self._result_queue,
                    env_overrides,
                    env_removals,
                ),
                daemon=True,
            )
            p.start()
            self._worker_processes.append(p)

        # Wait a moment for workers to start, then check for crashes
        time.sleep(2)
        if all(not p.is_alive() for p in self._worker_processes):
            raise RuntimeError("FSDP2 workers crashed on startup")

        # Send a no-op forward to verify workers are ready
        logger.info("Waiting for FSDP2 workers to load model...")
        self._send_command(
            {
                "type": "forward",
                "batch": {
                    "all_input_ids": [[1, 2, 3]],
                    "all_targets": [[2, 3, 0]],
                    "all_token_weights": [[1.0, 1.0, 1.0]],
                    "all_sampling_logprobs": [[0.0, 0.0, 0.0]],
                    "all_advantages": [[0.0, 0.0, 0.0]],
                    "all_loss_fns": ["cross_entropy"],
                    "all_loss_fn_configs": [None],
                },
            }
        )
        self._read_result(timeout=900)
        logger.info("FSDP2 workers ready")

        self._models[model_id] = types.ModelMetadata(
            adapter_index=self._adapter_counter,
            lora_config=lora_config,
        )
        self._adapter_counter += 1

    def set_micro_batch_size(self, n: int) -> None:
        self._micro_batch_size = n
        try:
            with open(self._mbs_file, "w") as f:
                f.write(str(n))
        except Exception:
            pass

    def _refresh_micro_batch_size(self) -> None:
        """Pick up changes written by the admin endpoint."""
        try:
            if os.path.exists(self._mbs_file):
                with open(self._mbs_file) as f:
                    self._micro_batch_size = int(f.read().strip())
        except Exception:
            pass

    def _send_command(self, cmd: dict) -> None:
        """Put command on the queue for rank 0 to read."""
        self._cmd_queue.put(cmd)

    def _read_result(self, timeout: float = 300) -> dict:
        """Read result from the result queue."""
        try:
            result = self._result_queue.get(timeout=timeout)
        except Exception:
            # Check if workers crashed
            if all(not p.is_alive() for p in self._worker_processes):
                raise RuntimeError("FSDP2 workers crashed")
            raise TimeoutError(f"FSDP2 result not received after {timeout}s")
        return result

    def _run_model_pass(
        self,
        prepared_batch: types.PreparedModelPassBatch,
        compute_gradients: bool,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        cmd_type = "forward_backward" if compute_gradients else "forward"
        self._refresh_micro_batch_size()

        batch_dict = {
            "all_input_ids": prepared_batch.all_input_ids,
            "all_targets": prepared_batch.all_targets,
            "all_token_weights": prepared_batch.all_token_weights,
            "all_sampling_logprobs": prepared_batch.all_sampling_logprobs,
            "all_advantages": prepared_batch.all_advantages,
            "all_loss_fns": prepared_batch.all_loss_fns,
            "all_loss_fn_configs": prepared_batch.all_loss_fn_configs,
        }

        self._send_command({"type": cmd_type, "batch": batch_dict, "micro_batch_size": self._micro_batch_size})

        n_examples = len(prepared_batch.all_input_ids)
        max_len = max((len(ids) for ids in prepared_batch.all_input_ids), default=0)
        timeout = max(300, n_examples * max_len / 100)
        result = self._read_result(timeout=timeout)

        results: dict[str, types.ForwardBackwardOutput | types.ErrorResponse] = {}
        for request_id, model_id, start_idx, end_idx in prepared_batch.request_batch_slices:
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                loss_fn_outputs.append(
                    {
                        "logprobs": {"data": result["logprobs"][i], "dtype": "float32"},
                        "elementwise_loss": {"data": result["losses"][i], "dtype": "float32"},
                    }
                )
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
        return self._run_model_pass(prepared_batch, compute_gradients=True)

    def forward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        return self._run_model_pass(prepared_batch, compute_gradients=False)

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        adam = request_data.adam_params
        self._send_command(
            {
                "type": "optim_step",
                "adam_params": {
                    "learning_rate": adam.learning_rate,
                    "beta1": adam.beta1,
                    "beta2": adam.beta2,
                    "eps": adam.eps,
                    "weight_decay": adam.weight_decay,
                },
            }
        )

        result = self._read_result(timeout=300)
        grad_norm = result.get("grad_norm", 0.0)

        if self.config.vllm_sync_url and "lora_path" in result:
            self._sync_lora_to_vllm(model_id, result["lora_path"])

        return types.OptimStepOutput(
            metrics={
                "skyrl.ai/grad_norm": grad_norm,
                "skyrl.ai/learning_rate": adam.learning_rate,
            }
        )

    def _sync_lora_to_vllm(self, model_id: str, lora_path: str) -> None:
        """Reload LoRA adapter on vLLM."""
        import requests as req_mod

        url = self.config.vllm_sync_url
        try:
            req_mod.post(f"{url}/v1/unload_lora_adapter", json={"lora_name": model_id}, timeout=30)
        except Exception:
            pass
        try:
            r = req_mod.post(
                f"{url}/v1/load_lora_adapter",
                json={"lora_name": model_id, "lora_path": os.path.abspath(lora_path)},
                timeout=60,
            )
            if r.status_code == 200:
                logger.info(f"Synced LoRA to vLLM: {model_id}")
        except Exception as e:
            logger.warning(f"vLLM LoRA sync failed: {e}")

    def sample(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        results = {}
        for request_id, model_id, start, end, _ in prepared_batch.request_batch_slices:
            results[request_id] = types.ErrorResponse(
                error="Sampling not supported in FSDP2 backend. Use external vLLM.",
                status="error",
            )
        return results

    def save_checkpoint(self, output_path, model_id: str) -> None:
        import tarfile
        import tempfile as tmpmod

        with tmpmod.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "adapter")
            self._send_command({"type": "save_checkpoint", "save_dir": save_dir})
            self._read_result(timeout=300)

            tar_path = os.path.join(tmpdir, "checkpoint.tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(save_dir, arcname="adapter")

            if hasattr(output_path, "write_bytes"):
                output_path.write_bytes(open(tar_path, "rb").read())
            else:
                import shutil

                shutil.copy2(tar_path, str(output_path))

    def load_checkpoint(self, checkpoint_path, model_id: str) -> None:
        logger.warning("load_checkpoint not implemented for FSDP2 backend")

    def save_sampler_checkpoint(self, output_path, model_id: str, persist: bool = True) -> None:
        if persist:
            self.save_checkpoint(output_path, model_id)
        else:
            import tarfile
            import tempfile as tmpmod

            with tmpmod.TemporaryDirectory() as tmpdir:
                tar_path = os.path.join(tmpdir, "marker.tar.gz")
                with tarfile.open(tar_path, "w:gz") as tar:
                    pass
                if hasattr(output_path, "write_bytes"):
                    output_path.write_bytes(open(tar_path, "rb").read())
                else:
                    import shutil

                    shutil.copy2(tar_path, str(output_path))

    def delete_model(self, model_id: str) -> None:
        self._models.pop(model_id, None)

    def _shutdown_workers(self) -> None:
        if self._worker_processes:
            try:
                self._send_command({"type": "shutdown"})
                for p in self._worker_processes:
                    p.join(timeout=30)
            except Exception:
                pass
            for p in self._worker_processes:
                if p.is_alive():
                    p.kill()
            self._worker_processes = []

    def shutdown(self) -> None:
        self._shutdown_workers()
        import shutil

        shutil.rmtree(self._ipc_dir, ignore_errors=True)
