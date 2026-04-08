"""Sweep backend configs to find best throughput setting.

Runs each config sequentially on GPUs 0-3, prints a comparison table.

Usage:
    .venv/bin/python benchmarks/sweep_configs.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import requests

os.environ.setdefault("TINKER_API_KEY", "tml-dummy")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(__file__))
from bench_gpu_throughput import make_mixed_data, GpuPoller, run_pass

import tinker

MODEL = "Qwen/Qwen3.5-9B"
PORT = 8771
GPU_IDS = [0, 1, 2, 3]
N_GPUS = 4
N_EXAMPLES = 32
MIN_SEQ = 64
MAX_SEQ = 8192
SEED = 42
WARMUP = 1
REPEAT = 3

CONFIGS = [
    # (backend, mbs, gc, remove_padding, label)
    # FSDP2: gc=off and mbs=4 OOM on Qwen3.5-9B with max_seq=8192
    ("fsdp2",          1, True,  False, "fsdp2 mbs=1 gc=on"),
    ("fsdp2",          2, True,  False, "fsdp2 mbs=2 gc=on"),
    ("fsdp2",          2, True,  True,  "fsdp2 mbs=2 gc=on rp=on"),
    ("megatron_local", 1, True,  False, "megatron mbs=1 gc=on"),
    ("megatron_local", 2, True,  False, "megatron mbs=2 gc=on"),
    ("megatron_local", 4, True,  False, "megatron mbs=4 gc=on"),
    ("megatron_local", 2, False, False, "megatron mbs=2 gc=off"),
    ("megatron_local", 4, False, False, "megatron mbs=4 gc=off"),
]


def make_backend_config(backend, mbs, gc, rp):
    cfg = {
        "n_train_gpus": N_GPUS,
        "train_gpu_offset": 0,
        "micro_batch_size": mbs,
        "gradient_checkpointing": gc,
    }
    if backend == "megatron_local":
        cfg["mode"] = "ddp"
    if backend == "fsdp2" and rp:
        cfg["remove_padding"] = True
    return cfg


def start_server(backend, backend_config):
    db = os.path.join(os.path.dirname(__file__), f"_sweep_{PORT}.db")
    for ext in ("", "-shm", "-wal"):
        try:
            os.remove(db + ext)
        except FileNotFoundError:
            pass

    cmd = [
        sys.executable, "-m", "hosted_tinker.api",
        "--base-model", MODEL,
        "--backend", backend,
        "--backend-config", json.dumps(backend_config),
        "--port", str(PORT),
        "--database-url", f"sqlite:///{db}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for ready
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{PORT}/api/v1/healthz", timeout=3)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError("Server died during startup")
        time.sleep(3)
    raise RuntimeError("Server startup timeout")


def kill_server(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    # Also kill any lingering torchrun workers
    subprocess.run(["pkill", "-9", "-f", "torchrun"], capture_output=True)
    time.sleep(8)  # allow GPU memory to fully free


def run_benchmark(data, seq_lens):
    url = f"http://localhost:{PORT}"
    service = tinker.ServiceClient(base_url=url)
    tc = service.create_lora_training_client(base_model=MODEL, rank=32)

    total_tokens = sum(seq_lens)
    poller = GpuPoller(GPU_IDS)
    results = {}

    for pass_type in ("forward", "forward_backward"):
        # Warmup
        for _ in range(WARMUP):
            try:
                run_pass(tc, data, pass_type, timeout=600)
            except Exception as e:
                return {"error": str(e)}

        elapsed_list = []
        for i in range(REPEAT):
            try:
                poller.start()
                elapsed = run_pass(tc, data, pass_type, timeout=600)
                poller.stop()
                elapsed_list.append(elapsed)
            except Exception as e:
                poller.stop()
                return {"error": str(e)}

        mean_elapsed = np.mean(elapsed_list)
        gpu_stats = poller.summary()
        results[pass_type] = {
            "tok_s": total_tokens / mean_elapsed,
            "elapsed": mean_elapsed,
            "gpu_util": gpu_stats.get("mean", 0),
            "gpu_mem": gpu_stats.get("mem_pct_mean", 0),
        }

    return results


def main():
    data, seq_lens = make_mixed_data(N_EXAMPLES, MIN_SEQ, MAX_SEQ, SEED)
    total_tokens = sum(seq_lens)
    print(f"Data: {N_EXAMPLES} examples, {total_tokens:,} tokens, "
          f"seq_len=[{MIN_SEQ},{MAX_SEQ}], seed={SEED}")
    print(f"GPUs: {GPU_IDS}, warmup={WARMUP}, repeat={REPEAT}")
    print()

    all_results = []

    for backend, mbs, gc, rp, label in CONFIGS:
        print(f"--- {label} ---")
        cfg = make_backend_config(backend, mbs, gc, rp)
        proc = None
        try:
            proc = start_server(backend, cfg)
            print(f"  Server up, running benchmark...")
            result = run_benchmark(data, seq_lens)
            if "error" in result:
                print(f"  ERROR: {result['error']}")
                all_results.append((label, None))
            else:
                fwd = result["forward"]
                fb = result["forward_backward"]
                print(f"  fwd: {fwd['tok_s']:.0f} tok/s  "
                      f"fwd+bwd: {fb['tok_s']:.0f} tok/s  "
                      f"gpu_util: {fb['gpu_util']:.0f}%  "
                      f"gpu_mem: {fb['gpu_mem']:.0f}%")
                all_results.append((label, result))
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append((label, None))
        finally:
            kill_server(proc)

    # Print summary table
    print()
    print("=" * 90)
    print(f"{'config':<30} {'fwd tok/s':>10} {'fwd+bwd tok/s':>14} "
          f"{'gpu_util%':>10} {'gpu_mem%':>10}")
    print("-" * 90)
    for label, result in all_results:
        if result is None:
            print(f"{label:<30} {'FAILED':>10}")
        else:
            fwd = result["forward"]
            fb = result["forward_backward"]
            print(f"{label:<30} {fwd['tok_s']:>10.0f} {fb['tok_s']:>14.0f} "
                  f"{fb['gpu_util']:>10.0f} {fb['gpu_mem']:>10.0f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
