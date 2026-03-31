"""Unified backend throughput benchmark: FSDP2 vs Megatron DDP.

Sweeps backend × micro_batch_size, running two configs in parallel
across two GPU halves (0-3 and 4-7). No server restart between mbs
values for the same backend.

Usage:
    python benchmarks/bench_backends.py \\
        --base-model Qwen/Qwen3.5-35B-A3B \\
        --backends megatron_local,fsdp2 \\
        --micro-batch-sizes 1,2

    # Single backend sweep only:
    python benchmarks/bench_backends.py \\
        --backends megatron_local \\
        --micro-batch-sizes 1,2,4
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import requests

os.environ.setdefault("TINKER_API_KEY", "tml-dummy")

sys.path.insert(0, os.path.dirname(__file__))
from bench_gpu_throughput import make_mixed_data, GpuPoller, run_pass

import tinker

_SLOTS = [
    {"gpu_offset": 0, "gpu_ids": [0, 1, 2, 3], "port": 8771},
    {"gpu_offset": 4, "gpu_ids": [4, 5, 6, 7], "port": 8772},
]


def make_backend_config(backend: str, n_train_gpus: int, gpu_offset: int,
                        micro_batch_size: int, gradient_checkpointing: bool,
                        remove_padding: bool = False) -> dict:
    cfg = {
        "n_train_gpus": n_train_gpus,
        "train_gpu_offset": gpu_offset,
        "micro_batch_size": micro_batch_size,
        "gradient_checkpointing": gradient_checkpointing,
    }
    if backend == "megatron_local":
        cfg["mode"] = "ddp"
    if backend == "fsdp2" and remove_padding:
        cfg["remove_padding"] = True
    return cfg


def wait_server_ready(url: str, timeout: int = 600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/api/v1/healthz", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def start_server(base_model: str, backend: str, backend_config: dict, port: int) -> subprocess.Popen:
    db_path = f"/tmp/tinker_bench_{port}.db"
    # Remove stale DB from previous runs to avoid engine processing old requests
    try:
        os.remove(db_path)
        os.remove(db_path + "-shm")
        os.remove(db_path + "-wal")
    except FileNotFoundError:
        pass
    cmd = [
        sys.executable, "-m", "hosted_tinker.api",
        "--base-model", base_model,
        "--backend", backend,
        "--backend-config", json.dumps(backend_config),
        "--port", str(port),
        "--database-url", f"sqlite:///{db_path}",
    ]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    return subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def benchmark_mbs_list(
    url: str,
    base_model: str,
    lora_rank: int,
    mbs_list: list[int],
    data,
    seq_lens: list[int],
    gpu_ids: list[int],
    n_warmup: int,
    n_repeat: int,
    label_prefix: str,
) -> dict:
    """Benchmark multiple mbs values on an already-running server."""
    service = tinker.ServiceClient(base_url=url)
    tc = service.create_lora_training_client(base_model=base_model, rank=lora_rank)
    total_tokens = sum(seq_lens)
    poller = GpuPoller(gpu_ids)
    results = {}

    for mbs in mbs_list:
        try:
            r = requests.post(f"{url}/admin/set_micro_batch_size", json={"n": mbs}, timeout=10)
            r.raise_for_status()
        except Exception as e:
            print(f"  [{label_prefix} mbs={mbs}] WARNING: set_micro_batch_size failed: {e}")

        label = f"{label_prefix} mbs={mbs}"
        mbs_results = {}
        for pass_type in ("forward", "forward_backward"):
            print(f"  [{label}][{pass_type}] warmup ({n_warmup})...")
            warmup_ok = True
            for _ in range(n_warmup):
                try:
                    run_pass(tc, data, pass_type)
                except Exception as e:
                    print(f"  [{label}][{pass_type}] warmup FAILED: {e}")
                    mbs_results[pass_type] = None
                    warmup_ok = False
                    break

            if not warmup_ok:
                continue

            elapsed_list, gpu_stats_list = [], []
            for i in range(n_repeat):
                print(f"  [{label}][{pass_type}] run {i+1}/{n_repeat}...")
                try:
                    poller.start()
                    elapsed = run_pass(tc, data, pass_type)
                    poller.stop()
                    elapsed_list.append(elapsed)
                    gpu_stats_list.append(poller.summary())
                    print(f"    elapsed={elapsed:.1f}s  tok/s={total_tokens/elapsed:.0f}  "
                          f"gpu_util={gpu_stats_list[-1]['mean']:.0f}%  "
                          f"gpu_mem={gpu_stats_list[-1]['mem_pct_mean']:.0f}%")
                except Exception as e:
                    poller.stop()
                    print(f"  [{label}][{pass_type}] run {i+1} FAILED: {e}")
                    break

            if elapsed_list:
                mean_e = np.mean(elapsed_list)
                mbs_results[pass_type] = {
                    "elapsed_s": float(mean_e),
                    "tok_per_s": float(total_tokens / mean_e),
                    "gpu_util_mean": float(np.mean([s["mean"] for s in gpu_stats_list])),
                    "gpu_mem_pct_mean": float(np.mean([s["mem_pct_mean"] for s in gpu_stats_list])),
                }
            else:
                mbs_results[pass_type] = None

        results[mbs] = mbs_results
    return results


def run_slot(
    slot: dict,
    backend: str,
    mbs_list: list[int],
    base_model: str,
    lora_rank: int,
    n_train_gpus: int,
    gradient_checkpointing: bool,
    data,
    seq_lens: list[int],
    n_warmup: int,
    n_repeat: int,
    server_start_timeout: int,
    out: dict,
    remove_padding: bool = False,
) -> None:
    gpu_offset = slot["gpu_offset"]
    gpu_ids = slot["gpu_ids"]
    port = slot["port"]
    url = f"http://localhost:{port}"
    first_mbs = mbs_list[0]
    backend_config = make_backend_config(backend, n_train_gpus, gpu_offset,
                                         first_mbs, gradient_checkpointing,
                                         remove_padding=remove_padding)
    label = f"{backend} gpus={gpu_ids[0]}-{gpu_ids[-1]}"

    print(f"\n[{label}] Starting server (mbs={first_mbs}, gc={'on' if gradient_checkpointing else 'off'})...")
    proc = start_server(base_model, backend, backend_config, port)
    try:
        if not wait_server_ready(url, timeout=server_start_timeout):
            print(f"  [{label}] ERROR: server not ready")
            for mbs in mbs_list:
                out[(backend, mbs)] = None
            return

        results = benchmark_mbs_list(
            url=url,
            base_model=base_model,
            lora_rank=lora_rank,
            mbs_list=mbs_list,
            data=data,
            seq_lens=seq_lens,
            gpu_ids=gpu_ids,
            n_warmup=n_warmup,
            n_repeat=n_repeat,
            label_prefix=backend,
        )
        for mbs, r in results.items():
            out[(backend, mbs)] = r
    except Exception as e:
        print(f"  [{label}] FAILED: {e}")
        for mbs in mbs_list:
            out[(backend, mbs)] = None
    finally:
        stop_server(proc)


def print_summary(all_results: dict, n_examples: int, total_tokens: int) -> None:
    W = 115
    print(f"\n{'='*W}")
    print(f"  Backend throughput  |  {n_examples} examples, {total_tokens:,} total tokens")
    print(f"{'='*W}")
    print(f"{'backend':>15} {'gpus':>5} {'mbs':>5} | "
          f"{'fwd tok/s':>11} {'gpu%':>6} {'mem%':>6} | "
          f"{'fwd+bwd tok/s':>14} {'gpu%':>6} {'mem%':>6}")
    print(f"{'-'*W}")

    for (backend, mbs), r in sorted(all_results.items(), key=lambda x: (x[0][0], x[0][1])):
        n_gpus = 4
        if r is None:
            print(f"{backend:>15} {n_gpus:>5} {mbs:>5}   {'FAILED':>11}")
            continue
        fwd = r.get("forward")
        fwdbwd = r.get("forward_backward")
        fwd_tps  = f"{fwd['tok_per_s']:>11.0f}"    if fwd    else f"{'OOM':>11}"
        fwd_gpu  = f"{fwd['gpu_util_mean']:>5.0f}%"  if fwd    else f"{'---':>6}"
        fwd_mem  = f"{fwd['gpu_mem_pct_mean']:>5.0f}%" if fwd   else f"{'---':>6}"
        fb_tps   = f"{fwdbwd['tok_per_s']:>14.0f}"  if fwdbwd else f"{'OOM':>14}"
        fb_gpu   = f"{fwdbwd['gpu_util_mean']:>5.0f}%" if fwdbwd else f"{'---':>6}"
        fb_mem   = f"{fwdbwd['gpu_mem_pct_mean']:>5.0f}%" if fwdbwd else f"{'---':>6}"
        print(f"{backend:>15} {n_gpus:>5} {mbs:>5} | {fwd_tps} {fwd_gpu} {fwd_mem} | {fb_tps} {fb_gpu} {fb_mem}")


def main():
    parser = argparse.ArgumentParser(description="Unified backend throughput benchmark")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--backends", default="megatron_local,fsdp2",
                        help="Comma-separated: megatron_local,fsdp2")
    parser.add_argument("--micro-batch-sizes", default="1,2",
                        help="Comma-separated micro_batch_size values")
    parser.add_argument("--n-train-gpus", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing for all backends")
    parser.add_argument("--gc-backends", default="",
                        help="Comma-separated backends to enable gc for (overrides --gradient-checkpointing)")
    parser.add_argument("--n-examples", type=int, default=32)
    parser.add_argument("--min-seq-len", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--server-start-timeout", type=int, default=900)
    parser.add_argument("--subsample", type=int, default=0,
                        help="Pick N evenly-spaced examples from generated dataset (0=use all)")
    parser.add_argument("--remove-padding", action="store_true", default=False,
                        help="Enable remove-padding (sequence packing) for FSDP2 backend")
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",")]
    mbs_list = [int(x) for x in args.micro_batch_sizes.split(",")]
    gc_backends = set(b.strip() for b in args.gc_backends.split(",") if b.strip())

    def backend_gc(backend: str) -> bool:
        if gc_backends:
            return backend in gc_backends
        return args.gradient_checkpointing

    print(f"Generating {args.n_examples} examples (len [{args.min_seq_len}, {args.max_seq_len}])...")
    data, seq_lens = make_mixed_data(args.n_examples, args.min_seq_len, args.max_seq_len, args.seed)
    if args.subsample and args.subsample < len(data):
        idxs = np.round(np.linspace(0, len(data) - 1, args.subsample)).astype(int)
        data = [data[i] for i in idxs]
        seq_lens = [seq_lens[i] for i in idxs]
        print(f"Subsampled to {len(data)} examples")
    total_tokens = sum(seq_lens)
    print(f"Total tokens: {total_tokens:,}")
    gc_str = ", ".join(f"{b}:{'gc=on' if backend_gc(b) else 'gc=off'}" for b in backends)
    print(f"Backends: {backends}  mbs: {mbs_list}  ({gc_str})")

    all_results: dict = {}

    for i in range(0, len(backends), 2):
        batch_backends = backends[i:i+2]
        out: dict = {}
        threads = []
        for slot, backend in zip(_SLOTS, batch_backends):
            t = threading.Thread(
                target=run_slot,
                args=(slot, backend, mbs_list, args.base_model, args.lora_rank,
                      args.n_train_gpus, backend_gc(backend),
                      data, seq_lens, args.warmup, args.repeat,
                      args.server_start_timeout, out, args.remove_padding),
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        all_results.update(out)
        if i + 2 < len(backends):
            print(f"\n  [batch done — waiting 15s for GPU memory to free]")
            time.sleep(15)

    print_summary(all_results, len(data), total_tokens)


if __name__ == "__main__":
    main()
