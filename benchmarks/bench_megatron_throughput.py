"""Megatron backend GPU throughput benchmark.

Sweeps micro_batch_size and mode (ddp/tp) by starting/stopping servers.
Runs two configs in parallel using both GPU halves (0-3 and 4-7).

Usage:
    # Sweep micro_batch_size over DDP mode (2 configs run at a time):
    python benchmarks/bench_megatron_throughput.py \\
        --base-model Qwen/Qwen3.5-35B-A3B \\
        --modes ddp \\
        --micro-batch-sizes 1,2,4,8

    # Compare DDP vs TP:
    python benchmarks/bench_megatron_throughput.py \\
        --base-model Qwen/Qwen3.5-35B-A3B \\
        --modes ddp,tp \\
        --micro-batch-sizes 2
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

# Two GPU sets and ports for parallel runs
_SLOTS = [
    {"gpu_offset": 0, "gpu_ids": [0, 1, 2, 3], "port": 8765},
    {"gpu_offset": 4, "gpu_ids": [4, 5, 6, 7], "port": 8766},
]


def wait_server_ready(url: str, timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/api/v1/healthz", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_server(
    base_model: str,
    n_train_gpus: int,
    train_gpu_offset: int,
    micro_batch_size: int,
    mode: str,
    gradient_checkpointing: bool,
    port: int,
) -> subprocess.Popen:
    backend_config = {
        "n_train_gpus": n_train_gpus,
        "train_gpu_offset": train_gpu_offset,
        "micro_batch_size": micro_batch_size,
        "mode": mode,
        "gradient_checkpointing": gradient_checkpointing,
    }
    cmd = [
        sys.executable, "-m", "hosted_tinker.api",
        "--base-model", base_model,
        "--backend", "megatron_local",
        "--backend-config", json.dumps(backend_config),
        "--port", str(port),
    ]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def benchmark_config(
    url: str,
    base_model: str,
    lora_rank: int,
    data,
    seq_lens: list[int],
    gpu_ids: list[int],
    n_warmup: int,
    n_repeat: int,
    label: str,
) -> dict:
    service = tinker.ServiceClient(base_url=url)
    tc = service.create_lora_training_client(base_model=base_model, rank=lora_rank)

    total_tokens = sum(seq_lens)
    poller = GpuPoller(gpu_ids)
    results = {}

    for pass_type in ("forward", "forward_backward"):
        print(f"  [{label}][{pass_type}] warmup ({n_warmup})...")
        for _ in range(n_warmup):
            run_pass(tc, data, pass_type)

        elapsed_list, gpu_stats_list = [], []
        for i in range(n_repeat):
            print(f"  [{label}][{pass_type}] run {i+1}/{n_repeat}...")
            poller.start()
            elapsed = run_pass(tc, data, pass_type)
            poller.stop()
            elapsed_list.append(elapsed)
            gpu_stats_list.append(poller.summary())
            print(f"    elapsed={elapsed:.1f}s  tok/s={total_tokens/elapsed:.0f}  "
                  f"gpu_util={gpu_stats_list[-1]['mean']:.0f}%")

        mean_elapsed = np.mean(elapsed_list)
        results[pass_type] = {
            "elapsed_s": mean_elapsed,
            "elapsed_std": float(np.std(elapsed_list)),
            "tok_per_s": total_tokens / mean_elapsed,
            "gpu_util_mean": float(np.mean([s["mean"] for s in gpu_stats_list])),
            "gpu_util_max": float(np.max([s["max"] for s in gpu_stats_list])),
            "per_gpu_mean": np.mean(
                [s["per_gpu_mean"] for s in gpu_stats_list if s["per_gpu_mean"]], axis=0
            ).tolist() if any(s["per_gpu_mean"] for s in gpu_stats_list) else [],
        }

    return {"total_tokens": total_tokens, "n_examples": len(data), "passes": results}


def run_one(
    slot: dict,
    cfg: tuple,
    base_model: str,
    lora_rank: int,
    n_train_gpus: int,
    data,
    seq_lens: list[int],
    n_warmup: int,
    n_repeat: int,
    server_start_timeout: int,
    out: dict,
) -> None:
    mode, mbs, gc = cfg
    gpu_offset = slot["gpu_offset"]
    gpu_ids = slot["gpu_ids"]
    port = slot["port"]
    url = f"http://localhost:{port}"
    label = f"mbs={mbs},gc={'Y' if gc else 'N'},gpus={gpu_ids[0]}-{gpu_ids[-1]}"

    print(f"\n[slot gpus={gpu_ids[0]}-{gpu_ids[-1]}] Starting: mode={mode} mbs={mbs} gc={'on' if gc else 'off'}")
    proc = start_server(
        base_model=base_model,
        n_train_gpus=n_train_gpus,
        train_gpu_offset=gpu_offset,
        micro_batch_size=mbs,
        mode=mode,
        gradient_checkpointing=gc,
        port=port,
    )
    try:
        if not wait_server_ready(url, timeout=server_start_timeout):
            print(f"  [{label}] ERROR: server not ready in time")
            out[cfg] = None
            return
        r = benchmark_config(
            url=url,
            base_model=base_model,
            lora_rank=lora_rank,
            data=data,
            seq_lens=seq_lens,
            gpu_ids=gpu_ids,
            n_warmup=n_warmup,
            n_repeat=n_repeat,
            label=label,
        )
        out[cfg] = r
    except Exception as e:
        print(f"  [{label}] FAILED: {e}")
        out[cfg] = None
    finally:
        stop_server(proc)


def print_summary(all_results: dict, total_tokens: int, n_examples: int) -> None:
    print(f"\n{'='*90}")
    print(f"  Megatron throughput sweep  |  {n_examples} examples, {total_tokens:,} total tokens")
    print(f"{'='*90}")
    print(f"{'mode':>5} {'mbs':>5} {'gc':>4} | "
          f"{'fwd tok/s':>10} {'fwd gpu%':>9} | "
          f"{'fwd+bwd tok/s':>14} {'fwd+bwd gpu%':>13}")
    print(f"{'-'*90}")
    for key, r in all_results.items():
        mode, mbs, gc = key
        if r is None:
            print(f"{mode:>5} {mbs:>5} {'Y' if gc else 'N':>4}   {'FAILED':>10}")
            continue
        fwd = r["passes"]["forward"]
        fwdbwd = r["passes"]["forward_backward"]
        print(f"{mode:>5} {mbs:>5} {'Y' if gc else 'N':>4} | "
              f"{fwd['tok_per_s']:>10.0f} {fwd['gpu_util_mean']:>8.0f}% | "
              f"{fwdbwd['tok_per_s']:>14.0f} {fwdbwd['gpu_util_mean']:>12.0f}%")

    valid = {k: v for k, v in all_results.items() if v}
    if valid:
        best_fwdbwd = max(valid, key=lambda k: valid[k]["passes"]["forward_backward"]["tok_per_s"])
        m, mbs, gc = best_fwdbwd
        print(f"\n  Best fwd+bwd: mode={m} micro_batch_size={mbs} gc={'on' if gc else 'off'}")


def main():
    parser = argparse.ArgumentParser(description="Megatron throughput benchmark sweep")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--modes", default="ddp",
                        help="Comma-separated modes to sweep: ddp,tp")
    parser.add_argument("--micro-batch-sizes", default="1,2,4,8",
                        help="Comma-separated micro_batch_size values to sweep")
    parser.add_argument("--gradient-checkpointing", default="true,false",
                        help="Comma-separated: true,false")
    parser.add_argument("--n-train-gpus", type=int, default=4)
    parser.add_argument("--n-examples", type=int, default=128)
    parser.add_argument("--min-seq-len", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--server-start-timeout", type=int, default=600,
                        help="Seconds to wait for server HTTP to become ready")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    mbs_list = [int(x) for x in args.micro_batch_sizes.split(",")]
    gc_list = [s.strip().lower() in ("true", "1", "yes") for s in args.gradient_checkpointing.split(",")]

    print(f"Generating {args.n_examples} examples (len [{args.min_seq_len}, {args.max_seq_len}])...")
    data, seq_lens = make_mixed_data(args.n_examples, args.min_seq_len, args.max_seq_len, args.seed)
    total_tokens = sum(seq_lens)
    print(f"Total tokens: {total_tokens:,}")

    # Build ordered sweep configs
    configs = [
        (mode, mbs, gc)
        for mode in modes
        for mbs in mbs_list
        for gc in gc_list
    ]

    all_results: dict = {}

    # Process configs in pairs — one per GPU slot, running in parallel threads
    for i in range(0, len(configs), 2):
        batch = configs[i:i+2]
        out: dict = {}
        threads = []
        for slot, cfg in zip(_SLOTS, batch):
            t = threading.Thread(
                target=run_one,
                args=(slot, cfg, args.base_model, args.lora_rank, args.n_train_gpus,
                      data, seq_lens, args.warmup, args.repeat,
                      args.server_start_timeout, out),
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        all_results.update(out)
        # For odd-length configs, mark missing slot as done
        if len(batch) == 1:
            pass

        print(f"\n  [batch done — waiting 10s for GPU memory to free]")
        time.sleep(10)

    print_summary(all_results, total_tokens, args.n_examples)


if __name__ == "__main__":
    main()
