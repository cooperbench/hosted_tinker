"""vLLM inference server management and LoRA weight synchronization."""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time

import requests

logger = logging.getLogger(__name__)


class VLLMManager:
    """Manages a vLLM inference server subprocess with LoRA adapter hot-reloading.

    Handles:
    - Starting/stopping vLLM on specified GPUs
    - Health checks
    - LoRA adapter reload via vLLM runtime API
    """

    def __init__(
        self,
        model_name: str,
        port: int = 8001,
        gpu_ids: list[int] | None = None,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 32768,
        max_num_seqs: int = 16,
        max_lora_rank: int = 32,
        dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.port = port
        self.gpu_ids = gpu_ids or [0]
        self.tensor_parallel_size = tensor_parallel_size or len(self.gpu_ids)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.max_lora_rank = max_lora_rank
        self.dtype = dtype
        self.base_url = f"http://localhost:{port}"
        self._process: subprocess.Popen | None = None
        self._log_file = None

    def start(self) -> None:
        """Launch vLLM server as a subprocess."""
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", self.dtype,
            "--max-model-len", str(self.max_model_len),
            "--max-num-seqs", str(self.max_num_seqs),
            "--enable-lora",
            "--max-lora-rank", str(self.max_lora_rank),
            "--trust-remote-code",
            "--enforce-eager",
        ]

        if self.tensor_parallel_size > 1:
            cmd.extend(["--tensor-parallel-size", str(self.tensor_parallel_size)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpu_ids)
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        # gIB NCCL fix: on GCP VMs without RDMA (a3-highgpu), the gIB NET
        # plugin fails to init. Force Socket transport for intra-node comms.
        env["NCCL_NET"] = "Socket"
        env.setdefault("NCCL_TUNER_CONFIG_PATH", "/usr/local/gib/configs")

        log_path = f"vllm_server_{self.port}.log"
        self._log_file = open(log_path, "w")

        self._process = subprocess.Popen(
            cmd, stdout=self._log_file, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid, env=env,
        )
        logger.info(f"vLLM started on GPUs {self.gpu_ids}, port {self.port}, PID {self._process.pid}")

    def wait_until_ready(self, timeout: float = 600) -> bool:
        """Block until vLLM responds to health checks."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=5)
                if r.status_code == 200:
                    logger.info(f"vLLM ready on port {self.port}")
                    return True
            except requests.ConnectionError:
                pass

            if self._process and self._process.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {self._process.returncode}")

            time.sleep(5)

        raise TimeoutError(f"vLLM not ready after {timeout}s")

    def reload_lora_adapters(self, adapter_dirs: dict[str, str]) -> None:
        """Reload LoRA adapters via vLLM runtime API.

        Args:
            adapter_dirs: Mapping of adapter_name -> directory path.
        """
        for name in adapter_dirs:
            try:
                requests.post(
                    f"{self.base_url}/v1/unload_lora_adapter",
                    json={"lora_name": name}, timeout=30,
                )
            except Exception:
                pass

        for name, path in adapter_dirs.items():
            abs_path = os.path.abspath(path)
            try:
                r = requests.post(
                    f"{self.base_url}/v1/load_lora_adapter",
                    json={"lora_name": name, "lora_path": abs_path}, timeout=60,
                )
                if r.status_code == 200:
                    logger.info(f"Reloaded LoRA adapter: {name}")
                else:
                    logger.warning(f"LoRA reload {name}: {r.status_code} {r.text[:200]}")
            except Exception as e:
                logger.error(f"LoRA reload {name} failed: {e}")

    def stop(self) -> None:
        """Stop vLLM server."""
        if self._process:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                self._process.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._process = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def __del__(self):
        self.stop()


def save_lora_for_vllm(
    model,
    adapter_dir: str,
    adapter_name: str = "default",
) -> str:
    """Extract LoRA weights from a PEFT model and save in vLLM-compatible format.

    Uses PEFT's save_pretrained for proper adapter format that vLLM can load.

    Args:
        model: PEFT model (from get_peft_model)
        adapter_dir: Base directory to save adapter
        adapter_name: Name for the adapter subdirectory

    Returns:
        Path to the saved adapter directory
    """
    save_path = os.path.join(adapter_dir, adapter_name)
    os.makedirs(save_path, exist_ok=True)

    try:
        # Use PEFT's save_pretrained — produces adapter_model.safetensors + adapter_config.json
        # in the correct format for vLLM
        model.save_pretrained(save_path, safe_serialization=True)
        logging.getLogger(__name__).info(f"Saved LoRA adapter to {save_path}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"save_pretrained failed ({e}), falling back to manual save")
        # Fallback: manual save (may not work with vLLM)
        import json
        from safetensors.torch import save_file

        lora_state = {
            k: v.cpu().contiguous()
            for k, v in model.state_dict().items()
            if "lora_" in k or "modules_to_save" in k
        }
        save_file(lora_state, os.path.join(save_path, "adapter_model.safetensors"))

        if hasattr(model, "peft_config"):
            for cfg_name, peft_cfg in model.peft_config.items():
                cfg_dict = peft_cfg.to_dict() if hasattr(peft_cfg, "to_dict") else {}
                def _serialize(obj):
                    if isinstance(obj, set): return list(obj)
                    if isinstance(obj, dict): return {k: _serialize(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)): return [_serialize(v) for v in obj]
                    return obj
                with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
                    json.dump(_serialize(cfg_dict), f)
                break

    return save_path
