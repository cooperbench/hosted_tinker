"""Benchmark PEFT vs FSDP2 backend latency and throughput.

Measures forward_backward + optim_step latency at various sequence lengths
for both backends.

Usage:
    # Benchmark PEFT backend (must be running on port 8000):
    python benchmarks/bench_backends.py --backend pytorch --url http://localhost:8000

    # Benchmark FSDP2 backend (must be running on port 8000):
    python benchmarks/bench_backends.py --backend fsdp2 --url http://localhost:8000

    # Compare two running servers:
    python benchmarks/bench_backends.py --compare \
        --url-a http://localhost:8000 --label-a PEFT \
        --url-b http://localhost:8001 --label-b FSDP2
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

os.environ.setdefault("TINKER_API_KEY", "tml-dummy")
import tinker


def make_datum(seq_len: int, seed: int = 42) -> tinker.Datum:
    """Create a cross_entropy training datum with random tokens."""
    rng = np.random.RandomState(seed)
    tokens = rng.randint(100, 150000, size=seq_len).tolist()
    target_tokens = tokens[1:] + [0]
    train_start = seq_len // 2
    weights = [0.0] * train_start + [1.0] * (seq_len - train_start)
    zeros = [0.0] * seq_len
    return tinker.Datum(
        model_input=tinker.ModelInput(chunks=[tinker.EncodedTextChunk(tokens=tokens)]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=target_tokens, dtype="int64"),
            "weights": tinker.TensorData(data=weights, dtype="float32"),
            "logprobs": tinker.TensorData(data=zeros, dtype="float32"),
            "advantages": tinker.TensorData(data=zeros, dtype="float32"),
        },
    )


def benchmark_server(
    base_url: str,
    model_name: str,
    lora_rank: int,
    lengths: list[int],
    n_warmup: int = 1,
    n_repeat: int = 3,
) -> dict[int, dict[str, float]]:
    """Benchmark a running tinker server.

    Returns:
        Dict mapping seq_len -> {"fwd_bwd_s": float, "optim_s": float, "total_s": float}
    """
    service = tinker.ServiceClient(base_url=base_url)
    tc = service.create_lora_training_client(base_model=model_name, rank=lora_rank)
    adam = tinker.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)

    results = {}

    for length in lengths:
        datum = make_datum(length)
        times_fwd = []
        times_opt = []

        for i in range(n_warmup + n_repeat):
            t0 = time.time()
            try:
                tc.forward_backward([datum], loss_fn="cross_entropy").result(timeout=1200)
            except Exception as e:
                print(f"  FAILED at length={length}: {str(e)[:100]}")
                results[length] = {"fwd_bwd_s": float("inf"), "optim_s": float("inf"),
                                   "total_s": float("inf"), "status": "FAILED"}
                break
            t_fwd = time.time() - t0

            t1 = time.time()
            tc.optim_step(adam).result(timeout=120)
            t_opt = time.time() - t1

            if i >= n_warmup:
                times_fwd.append(t_fwd)
                times_opt.append(t_opt)
        else:
            results[length] = {
                "fwd_bwd_s": np.mean(times_fwd),
                "optim_s": np.mean(times_opt),
                "total_s": np.mean(times_fwd) + np.mean(times_opt),
                "fwd_bwd_std": np.std(times_fwd),
                "status": "OK",
            }

    return results


def print_results(label: str, results: dict[int, dict[str, float]]) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"{'Seq Len':>10} {'fwd_bwd':>10} {'optim':>10} {'total':>10} {'tok/s':>10} {'status':>8}")
    print(f"{'-'*60}")

    for length, r in sorted(results.items()):
        if r["status"] == "OK":
            tok_per_s = length / r["fwd_bwd_s"] if r["fwd_bwd_s"] > 0 else 0
            print(f"{length:>10} {r['fwd_bwd_s']:>8.1f}s {r['optim_s']:>8.1f}s "
                  f"{r['total_s']:>8.1f}s {tok_per_s:>8.0f} {'OK':>8}")
        else:
            print(f"{length:>10} {'---':>10} {'---':>10} {'---':>10} {'---':>10} {'FAIL':>8}")


def print_comparison(results_a: dict, results_b: dict, label_a: str, label_b: str) -> None:
    """Print side-by-side comparison."""
    print(f"\n{'='*80}")
    print(f"  Comparison: {label_a} vs {label_b}")
    print(f"{'='*80}")
    print(f"{'Seq Len':>10} {'':>3} {label_a+' fwd':>10} {label_b+' fwd':>10} {'Speedup':>10}")
    print(f"{'-'*80}")

    all_lengths = sorted(set(results_a.keys()) | set(results_b.keys()))
    for length in all_lengths:
        ra = results_a.get(length, {"fwd_bwd_s": float("inf"), "status": "N/A"})
        rb = results_b.get(length, {"fwd_bwd_s": float("inf"), "status": "N/A"})

        if ra["status"] == "OK" and rb["status"] == "OK":
            speedup = ra["fwd_bwd_s"] / rb["fwd_bwd_s"] if rb["fwd_bwd_s"] > 0 else 0
            print(f"{length:>10} {'':>3} {ra['fwd_bwd_s']:>8.1f}s {rb['fwd_bwd_s']:>8.1f}s "
                  f"{speedup:>8.2f}x")
        else:
            sa = f"{ra['fwd_bwd_s']:.1f}s" if ra["status"] == "OK" else ra["status"]
            sb = f"{rb['fwd_bwd_s']:.1f}s" if rb["status"] == "OK" else rb["status"]
            print(f"{length:>10} {'':>3} {sa:>10} {sb:>10} {'---':>10}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark hosted-tinker backends")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--label", default="Backend")
    parser.add_argument("--lengths", default="1024,4096,8192,16384,32768",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    # Comparison mode
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--url-a", default=None)
    parser.add_argument("--url-b", default=None)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]

    if args.compare:
        assert args.url_a and args.url_b, "Provide --url-a and --url-b for comparison"
        print(f"Benchmarking {args.label_a} ({args.url_a})...")
        results_a = benchmark_server(args.url_a, args.model, args.lora_rank, lengths, args.warmup, args.repeat)
        print_results(args.label_a, results_a)

        print(f"\nBenchmarking {args.label_b} ({args.url_b})...")
        results_b = benchmark_server(args.url_b, args.model, args.lora_rank, lengths, args.warmup, args.repeat)
        print_results(args.label_b, results_b)

        print_comparison(results_a, results_b, args.label_a, args.label_b)
    else:
        print(f"Benchmarking {args.label} ({args.url})...")
        results = benchmark_server(args.url, args.model, args.lora_rank, lengths, args.warmup, args.repeat)
        print_results(args.label, results)


if __name__ == "__main__":
    main()
