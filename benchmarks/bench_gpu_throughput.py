"""GPU token throughput benchmark for hosted-tinker.

Sends 128 mixed-length sequences (short + mid + long, up to 32768 tokens) as
a single batch and measures token throughput for forward-only and
forward+backward passes. Polls nvidia-smi to report GPU utilization.

Usage:
    # Start server first, e.g. FSDP2 with micro_batch_size=4:
    #   python -m hosted_tinker.api --base-model Qwen/Qwen3.5-35B-A3B \\
    #       --backend fsdp2 --backend-config '{"n_train_gpus": 4, "micro_batch_size": 4}'
    #
    # Then run this benchmark:
    python benchmarks/bench_gpu_throughput.py --url http://localhost:8000

    # Specify venv python if needed:
    /path/to/venv/bin/python benchmarks/bench_gpu_throughput.py --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time

import numpy as np

os.environ.setdefault("TINKER_API_KEY", "tml-dummy")
import tinker


def make_mixed_data(n_examples: int, min_len: int, max_len: int, seed: int) -> list[tinker.Datum]:
    """Generate mixed-length data: short, mid, and long sequences."""
    rng = np.random.RandomState(seed)

    # Distribute: ~15% short, ~70% mid, ~15% long
    n_short = max(1, int(n_examples * 0.15))
    n_long = max(1, int(n_examples * 0.15))
    n_mid = n_examples - n_short - n_long

    short_max = min(256, max_len)
    long_min = max(min_len, max_len * 3 // 4)

    lengths = np.concatenate([
        rng.randint(min_len, short_max + 1, size=n_short),
        rng.randint(short_max + 1, long_min, size=n_mid),
        rng.randint(long_min, max_len + 1, size=n_long),
    ])
    rng.shuffle(lengths)

    data = []
    for seq_len in lengths:
        seq_len = int(seq_len)
        tokens = rng.randint(100, 150000, size=seq_len).tolist()
        target_tokens = tokens[1:] + [0]
        train_start = seq_len // 2
        weights = [0.0] * train_start + [1.0] * (seq_len - train_start)
        zeros = [0.0] * seq_len
        data.append(tinker.Datum(
            model_input=tinker.ModelInput(chunks=[tinker.EncodedTextChunk(tokens=tokens)]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(data=target_tokens, dtype="int64"),
                "weights": tinker.TensorData(data=weights, dtype="float32"),
                "logprobs": tinker.TensorData(data=zeros, dtype="float32"),
                "advantages": tinker.TensorData(data=zeros, dtype="float32"),
            },
        ))
    return data, lengths.tolist()


class GpuPoller:
    """Poll nvidia-smi GPU utilization in background thread."""

    def __init__(self, gpu_ids: list[int], interval: float = 0.5):
        self.gpu_ids = gpu_ids
        self.interval = interval
        self._samples: list[list[int]] = []  # [sample_i][gpu_j]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _poll(self):
        query = "utilization.gpu,memory.used,memory.total"
        gpu_arg = ",".join(str(g) for g in self.gpu_ids)
        cmd = [
            "nvidia-smi",
            f"--id={gpu_arg}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
                lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
                utils, mem_used, mem_total = [], [], []
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    utils.append(int(parts[0]))
                    mem_used.append(int(parts[1]))
                    mem_total.append(int(parts[2]))
                if utils:
                    self._samples.append({"util": utils, "mem_used": mem_used, "mem_total": mem_total})
            except Exception:
                pass
            time.sleep(self.interval)

    def summary(self) -> dict:
        """Return mean util%, mem% per GPU, and overall means."""
        if not self._samples:
            return {"mean": 0, "max": 0, "mem_pct_mean": 0, "per_gpu_mean": [], "per_gpu_max": []}
        util_arr = np.array([s["util"] for s in self._samples])  # [T, G]
        mem_pct_arr = np.array([
            [u / t * 100 for u, t in zip(s["mem_used"], s["mem_total"])]
            for s in self._samples
        ])  # [T, G]
        return {
            "mean": float(util_arr.mean()),
            "max": float(util_arr.max()),
            "mem_pct_mean": float(mem_pct_arr.mean()),
            "per_gpu_mean": util_arr.mean(axis=0).tolist(),
            "per_gpu_max": util_arr.max(axis=0).tolist(),
        }


def run_pass(
    tc,
    data: list[tinker.Datum],
    pass_type: str,
    timeout: int = 7200,
) -> float:
    """Run forward or forward_backward, return elapsed seconds."""
    t0 = time.time()
    if pass_type == "forward":
        tc.forward(data, loss_fn="cross_entropy").result(timeout=timeout)
    else:
        tc.forward_backward(data, loss_fn="cross_entropy").result(timeout=timeout)
    return time.time() - t0


def benchmark(
    url: str,
    model_name: str,
    lora_rank: int,
    data: list[tinker.Datum],
    seq_lens: list[int],
    gpu_ids: list[int],
    n_warmup: int,
    n_repeat: int,
) -> dict:
    service = tinker.ServiceClient(base_url=url)
    tc = service.create_lora_training_client(base_model=model_name, rank=lora_rank)

    total_tokens = sum(seq_lens)
    poller = GpuPoller(gpu_ids)

    results = {}
    for pass_type in ("forward", "forward_backward"):
        print(f"  [{pass_type}] warming up ({n_warmup} pass)...")
        for _ in range(n_warmup):
            run_pass(tc, data, pass_type)

        elapsed_list = []
        gpu_stats_list = []
        for i in range(n_repeat):
            print(f"  [{pass_type}] run {i+1}/{n_repeat}...")
            poller.start()
            elapsed = run_pass(tc, data, pass_type)
            poller.stop()
            elapsed_list.append(elapsed)
            gpu_stats_list.append(poller.summary())
            print(f"    elapsed={elapsed:.1f}s  tok/s={total_tokens/elapsed:.0f}  "
                  f"gpu_util_mean={gpu_stats_list[-1]['mean']:.0f}%  "
                  f"gpu_util_max={gpu_stats_list[-1]['max']:.0f}%")

        mean_elapsed = np.mean(elapsed_list)
        results[pass_type] = {
            "elapsed_s": mean_elapsed,
            "elapsed_std": np.std(elapsed_list),
            "tok_per_s": total_tokens / mean_elapsed,
            "gpu_util_mean": np.mean([s["mean"] for s in gpu_stats_list]),
            "gpu_util_max": np.max([s["max"] for s in gpu_stats_list]),
            "per_gpu_mean": np.mean([s["per_gpu_mean"] for s in gpu_stats_list if s["per_gpu_mean"]], axis=0).tolist()
            if any(s["per_gpu_mean"] for s in gpu_stats_list) else [],
        }

    return {"total_tokens": total_tokens, "n_examples": len(data), "passes": results}


def print_results(result: dict, seq_lens: list[int]) -> None:
    total = result["total_tokens"]
    n = result["n_examples"]

    # Seq-len distribution
    lens = sorted(seq_lens)
    short = sum(1 for l in lens if l <= 256)
    mid = sum(1 for l in lens if 256 < l <= 16384)
    long_ = sum(1 for l in lens if l > 16384)
    print(f"\n{'='*70}")
    print(f"  Sequence distribution: {n} examples, {total:,} total tokens")
    print(f"  short(≤256): {short}  mid(257-16384): {mid}  long(>16384): {long_}")
    print(f"  min={min(lens)}  median={int(np.median(lens))}  max={max(lens)}")
    print(f"{'='*70}")
    print(f"{'pass_type':<18} {'examples':>8} {'total_tok':>10} {'elapsed_s':>10} "
          f"{'tok/s':>8} {'gpu_mean%':>10} {'gpu_max%':>9}")
    print(f"{'-'*70}")
    for pass_type, r in result["passes"].items():
        per_gpu = "  [" + " ".join(f"{v:.0f}" for v in r["per_gpu_mean"]) + "]" if r["per_gpu_mean"] else ""
        print(f"{pass_type:<18} {n:>8} {total:>10,} {r['elapsed_s']:>9.1f}s "
              f"{r['tok_per_s']:>8.0f} {r['gpu_util_mean']:>9.0f}% {r['gpu_util_max']:>8.0f}%{per_gpu}")
    print()


def main():
    parser = argparse.ArgumentParser(description="GPU token throughput benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--n-examples", type=int, default=128)
    parser.add_argument("--min-seq-len", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--gpu-ids", default="0,1,2,3",
                        help="GPU indices to poll with nvidia-smi (comma-separated)")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]

    print(f"Generating {args.n_examples} examples (len [{args.min_seq_len}, {args.max_seq_len}])...")
    data, seq_lens = make_mixed_data(args.n_examples, args.min_seq_len, args.max_seq_len, args.seed)
    print(f"Total tokens: {sum(seq_lens):,}  |  server: {args.url}")

    result = benchmark(
        url=args.url,
        model_name=args.model,
        lora_rank=args.lora_rank,
        data=data,
        seq_lens=seq_lens,
        gpu_ids=gpu_ids,
        n_warmup=args.warmup,
        n_repeat=args.repeat,
    )
    print_results(result, seq_lens)


if __name__ == "__main__":
    main()
