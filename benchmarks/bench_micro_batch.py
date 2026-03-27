"""Micro-batch size sweep for FSDP2 backend.

Sweeps micro_batch_size values on a single running server by calling
POST /admin/set_micro_batch_size between steps — no server restart needed.

Usage:
    # Start server with FSDP2 backend:
    #   python -m hosted_tinker.api --base-model Qwen/Qwen3.5-35B-A3B \\
    #       --backend fsdp2 --backend-config '{"n_train_gpus": 4, "micro_batch_size": 1}'
    #
    # Then run the sweep:
    python benchmarks/bench_micro_batch.py --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import requests

os.environ.setdefault("TINKER_API_KEY", "tml-dummy")
import tinker

sys.path.insert(0, os.path.dirname(__file__))
from bench_gpu_throughput import make_mixed_data, GpuPoller, run_pass


def set_micro_batch_size(url: str, n: int) -> None:
    r = requests.post(f"{url}/admin/set_micro_batch_size", json={"n": n}, timeout=10)
    r.raise_for_status()


def benchmark_one(tc, data, seq_lens, gpu_ids, n_warmup, n_repeat):
    total_tokens = sum(seq_lens)
    poller = GpuPoller(gpu_ids)
    results = {}
    for pass_type in ("forward", "forward_backward"):
        for _ in range(n_warmup):
            run_pass(tc, data, pass_type)

        elapsed_list, gpu_means, gpu_maxes = [], [], []
        for _ in range(n_repeat):
            poller.start()
            e = run_pass(tc, data, pass_type)
            poller.stop()
            elapsed_list.append(e)
            s = poller.summary()
            gpu_means.append(s["mean"])
            gpu_maxes.append(s["max"])

        mean_e = np.mean(elapsed_list)
        results[pass_type] = {
            "tok_per_s": total_tokens / mean_e,
            "gpu_util_mean": np.mean(gpu_means),
            "gpu_util_max": np.max(gpu_maxes),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Micro-batch size sweep benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--micro-batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--n-examples", type=int, default=128)
    parser.add_argument("--min-seq-len", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    args = parser.parse_args()

    mbs_list = [int(x) for x in args.micro_batch_sizes.split(",")]
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    data, seq_lens = make_mixed_data(args.n_examples, args.min_seq_len, args.max_seq_len, args.seed)
    total_tokens = sum(seq_lens)
    print(f"Data: {args.n_examples} examples, {total_tokens:,} tokens")

    service = tinker.ServiceClient(base_url=args.url)
    tc = service.create_lora_training_client(base_model=args.model, rank=args.lora_rank)

    all_results = {}
    for mbs in mbs_list:
        print(f"\nmicro_batch_size={mbs}: setting via admin endpoint...")
        try:
            set_micro_batch_size(args.url, mbs)
        except Exception as e:
            print(f"  WARNING: failed to set mbs via admin endpoint: {e}")

        print(f"micro_batch_size={mbs}: benchmarking...")
        try:
            r = benchmark_one(tc, data, seq_lens, gpu_ids, args.warmup, args.repeat)
            all_results[mbs] = r
            fwd = r["forward"]
            fwdbwd = r["forward_backward"]
            print(f"  fwd: {fwd['tok_per_s']:.0f} tok/s  gpu_util={fwd['gpu_util_mean']:.0f}%")
            print(f"  fwd+bwd: {fwdbwd['tok_per_s']:.0f} tok/s  gpu_util={fwdbwd['gpu_util_mean']:.0f}%")
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results[mbs] = None

    # Summary table
    print(f"\n{'='*80}")
    print(f"  Micro-batch size sweep  |  {args.n_examples} examples, {total_tokens:,} tokens")
    print(f"{'='*80}")
    print(f"{'micro_bs':>8} {'fwd tok/s':>12} {'fwd gpu%':>10} {'fwd+bwd tok/s':>14} {'fwd+bwd gpu%':>13}")
    print(f"{'-'*60}")
    for mbs in mbs_list:
        r = all_results.get(mbs)
        if r is None:
            print(f"{mbs:>8} {'FAILED':>12}")
            continue
        fwd = r["forward"]
        fwdbwd = r["forward_backward"]
        print(f"{mbs:>8} {fwd['tok_per_s']:>12.0f} {fwd['gpu_util_mean']:>9.0f}% "
              f"{fwdbwd['tok_per_s']:>14.0f} {fwdbwd['gpu_util_mean']:>12.0f}%")

    valid = {m: v for m, v in all_results.items() if v}
    if valid:
        best_fwd = max(valid, key=lambda m: valid[m]["forward"]["tok_per_s"])
        best_fwdbwd = max(valid, key=lambda m: valid[m]["forward_backward"]["tok_per_s"])
        print(f"\n  Best fwd tok/s: micro_batch_size={best_fwd}")
        print(f"  Best fwd+bwd tok/s: micro_batch_size={best_fwdbwd}")


if __name__ == "__main__":
    main()
