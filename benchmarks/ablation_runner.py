#!/usr/bin/env python3
"""Ablation study for issue #5 perf tricks on FSDP2 with remove-padding.

Runs bench_backends.py with different env var toggles to isolate
each trick's contribution to throughput.

Leave-one-out design: baseline has all tricks ON, each ablation
disables one trick.
"""

import json
import os
import re
import subprocess
import sys

BENCH_CMD = [
    sys.executable, "benchmarks/bench_backends.py",
    "--base-model", "Qwen/Qwen3.5-9B",
    "--backends", "fsdp2",
    "--micro-batch-sizes", "1,2",
    "--n-train-gpus", "4",
    "--gradient-checkpointing",
    "--n-examples", "32",
    "--max-seq-len", "8192",
    "--lora-rank", "32",
    "--remove-padding",
    "--warmup", "1",
    "--repeat", "3",
]

CONFIGS = [
    ("baseline (all tricks ON)", {}),
    ("-queue_ipc (torchrun+file)", {"ABLATE_TORCHRUN": "1"}),
    ("-deterministic_sync (allgather)", {"ABLATE_ALLGATHER": "1"}),
    ("-event_dispatch (100ms poll)", {"ABLATE_POLLING": "1"}),
    ("-batched_transfer (eager)", {"ABLATE_EAGER_TRANSFER": "1"}),
    ("-conditional_gloo (always gloo)", {"ABLATE_ALWAYS_GLOO": "1"}),
]


def parse_summary(output: str) -> dict:
    """Parse the summary table from bench_backends output."""
    results = {}
    for line in output.split("\n"):
        m = re.match(r"\s*fsdp2\s+4\s+(\d+)\s+\|\s+([\d.]+|OOM)\s+", line)
        if m:
            mbs = int(m.group(1))
            # Parse fwd tok/s and fwd+bwd tok/s
            parts = line.split("|")
            if len(parts) >= 3:
                fwd_part = parts[1].strip().split()
                fb_part = parts[2].strip().split()
                fwd_tps = float(fwd_part[0]) if fwd_part and fwd_part[0] != "OOM" else None
                fb_tps = float(fb_part[0]) if fb_part and fb_part[0] != "OOM" else None
                results[mbs] = {"fwd": fwd_tps, "fwd_bwd": fb_tps}
    return results


def main():
    all_results = {}

    for name, env_vars in CONFIGS:
        print(f"\n{'='*80}")
        print(f"  ABLATION: {name}")
        print(f"  env: {env_vars or '(none)'}")
        print(f"{'='*80}\n")

        env = os.environ.copy()
        env.update(env_vars)

        proc = subprocess.run(
            BENCH_CMD, env=env, capture_output=True, text=True, timeout=900,
        )
        output = proc.stdout + proc.stderr
        print(output[-3000:])  # Print last 3K chars

        results = parse_summary(output)
        all_results[name] = results
        print(f"\n  → Parsed: {results}")

    # Print comparison table
    print(f"\n\n{'='*100}")
    print(f"  ABLATION RESULTS SUMMARY")
    print(f"{'='*100}")
    print(f"{'config':>40} | {'fwd mbs=1':>10} {'fwd mbs=2':>10} | {'fb mbs=1':>10} {'fb mbs=2':>10}")
    print(f"{'-'*100}")

    for name, results in all_results.items():
        fwd1 = f"{results.get(1, {}).get('fwd', 0):>10.0f}" if results.get(1, {}).get('fwd') else f"{'---':>10}"
        fwd2 = f"{results.get(2, {}).get('fwd', 0):>10.0f}" if results.get(2, {}).get('fwd') else f"{'---':>10}"
        fb1 = f"{results.get(1, {}).get('fwd_bwd', 0):>10.0f}" if results.get(1, {}).get('fwd_bwd') else f"{'---':>10}"
        fb2 = f"{results.get(2, {}).get('fwd_bwd', 0):>10.0f}" if results.get(2, {}).get('fwd_bwd') else f"{'---':>10}"
        print(f"{name:>40} | {fwd1} {fwd2} | {fb1} {fb2}")

    # Save raw results
    with open("benchmarks/ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to benchmarks/ablation_results.json")


if __name__ == "__main__":
    main()
