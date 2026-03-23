"""Megatron-Core backend for hosted-tinker.

Uses Megatron-Core for distributed training with tensor parallelism.
Runs as a torchrun subprocess with multiple workers.

Supports both H100 and B200 GPUs:
- H100: NCCL P2P works, standard collectives
- B200: NCCL P2P disabled, Gloo group for object collectives

Usage:
    python -m hosted_tinker.api --base-model Qwen/Qwen3-30B-A3B \
        --backend megatron --backend-config '{"n_train_gpus": 4}'
"""
from __future__ import annotations

import logging
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time

from pydantic import BaseModel, Field

from hosted_tinker.backend import AbstractBackend
from hosted_tinker import types

logger = logging.getLogger(__name__)


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


class MegatronBackendConfig(BaseModel, extra="forbid"):
    """Configuration for the Megatron backend."""

    n_train_gpus: int = Field(default=4, description="Number of GPUs for training")
    train_gpu_offset: int = Field(default=0, description="First GPU index for training")
    gradient_checkpointing: bool = Field(default=True, description="Enable activation checkpointing")
    micro_batch_size: int = Field(default=1, description="Micro-batch size for gradient accumulation")
    # Worker mode: "ddp" (HF model, data parallel) or "tp" (Megatron-Core, tensor parallel)
    mode: str = Field(default="ddp", description="Worker mode: ddp (any model) or tp (Megatron Bridge models only)")
    # vLLM LoRA sync
    vllm_sync_url: str | None = Field(default=None, description="vLLM base URL for LoRA sync")
    lora_sync_dir: str = Field(default="/dev/shm/lora_adapters", description="Dir for LoRA weight sync")


class MegatronBackend(AbstractBackend):
    """Megatron-Core backend using torchrun subprocess workers.

    Similar to FSDP2Backend but uses Megatron's tensor parallelism
    (all_reduce-based) instead of FSDP's parameter sharding.
    This works on B200 GPUs where FSDP hangs.
    """

    def __init__(self, base_model: str, config: MegatronBackendConfig):
        self.base_model_name = base_model
        self.config = config
        self.metrics = types.EngineMetrics()

        self._models: dict[str, types.ModelMetadata] = {}
        self._worker_process: subprocess.Popen | None = None
        self._adapter_counter = 0

        # IPC files
        self._ipc_dir = tempfile.mkdtemp(prefix="megatron_ipc_")
        self._cmd_file = os.path.join(self._ipc_dir, "cmd.pkl")
        self._result_file = os.path.join(self._ipc_dir, "result.pkl")

    def has_model(self, model_id: str) -> bool:
        return model_id in self._models

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        if self._worker_process is not None:
            self._shutdown_workers()

        gpu_ids = list(range(self.config.train_gpu_offset,
                             self.config.train_gpu_offset + self.config.n_train_gpus))
        master_port = random.randint(29500, 29999)

        # Select worker based on mode
        if self.config.mode == "tp":
            worker_module = "hosted_tinker.megatron_tp_worker"
        else:
            worker_module = "hosted_tinker.megatron_worker"
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(self.config.n_train_gpus),
            "--master_port", str(master_port),
            "-m", worker_module,
            "--base-model", self.base_model_name,
            "--lora-rank", str(lora_config.rank),
            "--lora-alpha", str(int(lora_config.alpha)),
            "--cmd-file", self._cmd_file,
            "--result-file", self._result_file,
            "--lora-sync-dir", self.config.lora_sync_dir,
        ]
        if self.config.gradient_checkpointing:
            cmd.append("--gradient-checkpointing")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        env["HF_HUB_OFFLINE"] = "1"
        # GCP VMs set NCCL_NET=gIB for multi-node clusters but
        # libibverbs.so is not installed on single-node VMs
        env["NCCL_NET_PLUGIN"] = ""
        env.pop("NCCL_NET", None)
        # B200 GPUs need NCCL P2P disabled (pytorch#165727)
        # H100/A100 GPUs work fine with P2P enabled
        gpu_type = _detect_gpu_type()
        if gpu_type == "B200":
            env["NCCL_P2P_DISABLE"] = "1"
        else:
            env.pop("NCCL_P2P_DISABLE", None)

        logger.info(f"Launching Megatron workers on GPUs {gpu_ids} (port {master_port})")

        log_path = os.path.join(self._ipc_dir, "worker.log")
        self._worker_log = open(log_path, "w")
        self._worker_process = subprocess.Popen(
            cmd, env=env, stdout=self._worker_log, stderr=subprocess.STDOUT,
        )

        time.sleep(2)
        if self._worker_process.poll() is not None:
            output = open(log_path).read()[-2000:] if os.path.exists(log_path) else ""
            raise RuntimeError(f"Megatron workers crashed: {output}")

        logger.info("Waiting for Megatron workers to load model...")
        # Send init probe
        self._send_command({"type": "forward", "batch": {
            "all_input_ids": [[1, 2, 3]], "all_targets": [[2, 3, 0]],
            "all_token_weights": [[1.0, 1.0, 1.0]], "all_sampling_logprobs": [[0.0, 0.0, 0.0]],
            "all_advantages": [[0.0, 0.0, 0.0]], "all_loss_fns": ["cross_entropy"],
            "all_loss_fn_configs": [None],
        }})
        result = self._read_result(timeout=900)
        logger.info("Megatron workers ready")

        self._models[model_id] = types.ModelMetadata(
            adapter_index=self._adapter_counter,
            lora_config=lora_config,
        )
        self._adapter_counter += 1

    def _send_command(self, cmd: dict) -> None:
        with open(self._cmd_file, "wb") as f:
            pickle.dump(cmd, f)

    def _read_result(self, timeout: float = 300) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(self._result_file):
                time.sleep(0.01)
                try:
                    with open(self._result_file, "rb") as f:
                        result = pickle.load(f)
                    os.unlink(self._result_file)
                    return result
                except (EOFError, pickle.UnpicklingError):
                    time.sleep(0.1)
                    continue

            if self._worker_process and self._worker_process.poll() is not None:
                log_path = os.path.join(self._ipc_dir, "worker.log")
                output = open(log_path).read()[-2000:] if os.path.exists(log_path) else ""
                raise RuntimeError(f"Megatron workers crashed: {output}")

            time.sleep(0.05)
        raise TimeoutError(f"Megatron result not received after {timeout}s")

    def _run_model_pass(
        self, prepared_batch: types.PreparedModelPassBatch, compute_gradients: bool,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        cmd_type = "forward_backward" if compute_gradients else "forward"
        batch_dict = {
            "all_input_ids": prepared_batch.all_input_ids,
            "all_targets": prepared_batch.all_targets,
            "all_token_weights": prepared_batch.all_token_weights,
            "all_sampling_logprobs": prepared_batch.all_sampling_logprobs,
            "all_advantages": prepared_batch.all_advantages,
            "all_loss_fns": prepared_batch.all_loss_fns,
            "all_loss_fn_configs": prepared_batch.all_loss_fn_configs,
        }
        self._send_command({"type": cmd_type, "batch": batch_dict})

        n_examples = len(prepared_batch.all_input_ids)
        max_len = max((len(ids) for ids in prepared_batch.all_input_ids), default=0)
        timeout = max(300, n_examples * max_len / 100)
        result = self._read_result(timeout=timeout)

        results: dict[str, types.ForwardBackwardOutput | types.ErrorResponse] = {}
        for request_id, model_id, start_idx, end_idx in prepared_batch.request_batch_slices:
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                loss_fn_outputs.append({
                    "logprobs": {"data": result["logprobs"][i], "dtype": "float32"},
                    "elementwise_loss": {"data": result["losses"][i], "dtype": "float32"},
                })
            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar",
                loss_fn_outputs=loss_fn_outputs,
                metrics={},
            )
        return results

    def forward_backward(self, prepared_batch):
        return self._run_model_pass(prepared_batch, compute_gradients=True)

    def forward(self, prepared_batch):
        return self._run_model_pass(prepared_batch, compute_gradients=False)

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        adam = request_data.adam_params
        self._send_command({
            "type": "optim_step",
            "adam_params": {
                "learning_rate": adam.learning_rate, "beta1": adam.beta1,
                "beta2": adam.beta2, "eps": adam.eps, "weight_decay": adam.weight_decay,
            },
        })
        result = self._read_result(timeout=300)
        grad_norm = result.get("grad_norm", 0.0)

        if self.config.vllm_sync_url and "lora_path" in result:
            self._sync_lora_to_vllm(model_id, result["lora_path"])

        return types.OptimStepOutput(metrics={
            "skyrl.ai/grad_norm": grad_norm,
            "skyrl.ai/learning_rate": adam.learning_rate,
        })

    def _sync_lora_to_vllm(self, model_id, lora_path):
        import requests as req
        url = self.config.vllm_sync_url
        try:
            req.post(f"{url}/v1/unload_lora_adapter", json={"lora_name": model_id}, timeout=30)
        except Exception:
            pass
        try:
            req.post(f"{url}/v1/load_lora_adapter",
                     json={"lora_name": model_id, "lora_path": os.path.abspath(lora_path)}, timeout=60)
        except Exception as e:
            logger.warning(f"vLLM sync failed: {e}")

    def sample(self, prepared_batch):
        results = {}
        for request_id, *_ in prepared_batch.request_batch_slices:
            results[request_id] = types.ErrorResponse(
                error="Sampling not supported. Use external vLLM.", status="error")
        return results

    def save_checkpoint(self, output_path, model_id):
        import tarfile
        with tempfile.TemporaryDirectory() as tmpdir:
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

    def load_checkpoint(self, checkpoint_path, model_id):
        logger.warning("load_checkpoint not implemented for Megatron backend")

    def save_sampler_checkpoint(self, output_path, model_id, persist=True):
        if persist:
            self.save_checkpoint(output_path, model_id)
        else:
            import tarfile
            with tempfile.TemporaryDirectory() as tmpdir:
                tar_path = os.path.join(tmpdir, "marker.tar.gz")
                with tarfile.open(tar_path, "w:gz") as tar:
                    pass
                if hasattr(output_path, "write_bytes"):
                    output_path.write_bytes(open(tar_path, "rb").read())
                else:
                    import shutil
                    shutil.copy2(tar_path, str(output_path))

    def delete_model(self, model_id):
        self._models.pop(model_id, None)

    def _shutdown_workers(self):
        if self._worker_process:
            try:
                self._send_command({"type": "shutdown"})
                self._worker_process.wait(timeout=30)
            except Exception:
                try:
                    self._worker_process.kill()
                except Exception:
                    pass
            self._worker_process = None

    def shutdown(self):
        self._shutdown_workers()
        import shutil
        shutil.rmtree(self._ipc_dir, ignore_errors=True)
