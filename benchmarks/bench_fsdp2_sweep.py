"""FSDP2 throughput sweep on GPUs 4-7 with max_seq_len=16384.

Sweeps micro_batch_size × gradient_checkpointing × remove_padding.
Starts a fresh server for each (gc, remove_padding) combo, then
changes mbs at runtime via /admin/set_micro_batch_size.

Usage:
    python benchmarks/bench_fsdp2_sweep.py --base-model Qwen/Qwen3.5-9B
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import requests

os.environ.setdefault("TINKER_API_KEY", "tml-dummy")

sys.path.insert(0, os.path.dirname(__file__))
from bench_gpu_throughput import make_mixed_data, GpuPoller, run_pass

import tinker

GPU_OFFSET = 4
GPU_IDS = [4, 5, 6, 7]
N_TRAIN_GPUS = 4
PORT = 8773


def start_server(base_model: str, mbs: int, gc: bool, remove_padding: bool) -> subprocess.Popen:
    db_path = f"/tmp/tinker_bench_{PORT}.db"
    for suffix in ("", "-shm", "-wal"):
        try:
            os.remove(db_path + suffix)
        except FileNotFoundError:
            pass

    cfg = {
        "n_train_gpus": N_TRAIN_GPUS,
        "train_gpu_offset": GPU_OFFSET,
        "micro_batch_size": mbs,
        "gradient_checkpointing": gc,
        "remove_padding": remove_padding,
    }
    cmd = [
        sys.executable, "-m", "hosted_tinker.api",
        "--base-model", base_model,
        "--backend", "fsdp2",
        "--backend-config", json.dumps(cfg),
        "--port", str(PORT),
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


def wait_ready(timeout: int = 600) -> bool:
    url = f"http://localhost:{PORT}/api/v1/healthz"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def set_mbs(mbs: int) -> bool:
    try:
        r = requests.post(f"http://localhost:{PORT}/admin/set_micro_batch_size",
                          json={"n": mbs}, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"  WARNING: set_mbs({mbs}) failed: {e}")
        return False


def bench_pass(tc, data, pass_type, total_tokens, n_warmup, n_repeat):
    poller = GpuPoller(GPU_IDS)
    # warmup
    for _ in range(n_warmup):
        try:
            run_pass(tc, data, pass_type)
        except Exception as e:
            print(f"    warmup {pass_type} FAILED: {e}")
            return None

    elapsed_list, gpu_stats = [], []
    for i in range(n_repeat):
        try:
            poller.start()
            elapsed = run_pass(tc, data, pass_type)
            poller.stop()
            elapsed_list.append(elapsed)
            stats = poller.summary()
            gpu_stats.append(stats)
            print(f"    {pass_type} run {i+1}: {elapsed:.1f}s  "
                  f"{total_tokens/elapsed:.0f} tok/s  "
                  f"gpu={stats['mean']:.0f}%  mem={stats['mem_pct_mean']:.0f}%")
        except Exception as e:
            poller.stop()
            print(f"    {pass_type} run {i+1} FAILED: {e}")
            return None

    mean_e = np.mean(elapsed_list)
    return {
        "tok_per_s": float(total_tokens / mean_e),
        "elapsed_s": float(mean_e),
        "gpu_util": float(np.mean([s["mean"] for s in gpu_stats])),
        "mem_pct": float(np.mean([s["mem_pct_mean"] for s in gpu_stats])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--n-examples", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--min-seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--mbs-list", default="1,2,4",
                        help="Comma-separated micro_batch_size values")
    args = parser.parse_args()

    mbs_list = [int(x) for x in args.mbs_list.split(",")]

    print(f"Generating {args.n_examples} examples (len [{args.min_seq_len}, {args.max_seq_len}])...")
    data, seq_lens = make_mixed_data(args.n_examples, args.min_seq_len, args.max_seq_len, args.seed)
    total_tokens = sum(seq_lens)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Seq len: min={min(seq_lens)} median={int(np.median(seq_lens))} max={max(seq_lens)}")

    # Configs: (gc, remove_padding)
    server_configs = [
        (False, False),   # gc=off, no packing
        (True, False),    # gc=on, no packing
        (True, True),     # gc=on, packing
        (False, True),    # gc=off, packing
    ]

    all_results = []  # list of dicts

    for gc, rp in server_configs:
        label = f"gc={'on' if gc else 'off'} pack={'on' if rp else 'off'}"
        first_mbs = mbs_list[0]
        print(f"\n{'='*70}")
        print(f"Starting server: {label} (initial mbs={first_mbs})")
        print(f"{'='*70}")

        proc = start_server(args.base_model, first_mbs, gc, rp)
        try:
            if not wait_ready():
                print(f"  ERROR: server not ready for {label}")
                for mbs in mbs_list:
                    all_results.append({
                        "gc": gc, "remove_padding": rp, "mbs": mbs,
                        "forward": None, "forward_backward": None,
                    })
                continue

            url = f"http://localhost:{PORT}"
            service = tinker.ServiceClient(base_url=url)
            tc = service.create_lora_training_client(
                base_model=args.base_model, rank=args.lora_rank)

            for mbs in mbs_list:
                print(f"\n  --- mbs={mbs} {label} ---")
                set_mbs(mbs)
                time.sleep(1)

                fwd = bench_pass(tc, data, "forward", total_tokens,
                                 args.warmup, args.repeat)
                fwdbwd = bench_pass(tc, data, "forward_backward", total_tokens,
                                    args.warmup, args.repeat)
                all_results.append({
                    "gc": gc, "remove_padding": rp, "mbs": mbs,
                    "forward": fwd, "forward_backward": fwdbwd,
                })
        finally:
            stop_server(proc)
            time.sleep(5)  # let GPUs free

    # Print summary
    W = 110
    print(f"\n{'='*W}")
    print(f"  FSDP2 Throughput Sweep | {args.base_model} | 4×H100 (GPUs 4-7)")
    print(f"  {len(data)} examples, {total_tokens:,} tokens, max_seq_len={args.max_seq_len}")
    print(f"{'='*W}")
    print(f"{'gc':>4} {'pack':>5} {'mbs':>4} | "
          f"{'fwd tok/s':>10} {'gpu%':>5} {'mem%':>5} | "
          f"{'fwd+bwd tok/s':>14} {'gpu%':>5} {'mem%':>5}")
    print(f"{'-'*W}")

    for r in all_results:
        gc_s = "on" if r["gc"] else "off"
        rp_s = "on" if r["remove_padding"] else "off"
        fwd = r["forward"]
        fb = r["forward_backward"]
        fwd_tps = f"{fwd['tok_per_s']:>10.0f}" if fwd else f"{'OOM':>10}"
        fwd_gpu = f"{fwd['gpu_util']:>4.0f}%" if fwd else f"{'---':>5}"
        fwd_mem = f"{fwd['mem_pct']:>4.0f}%" if fwd else f"{'---':>5}"
        fb_tps = f"{fb['tok_per_s']:>14.0f}" if fb else f"{'OOM':>14}"
        fb_gpu = f"{fb['gpu_util']:>4.0f}%" if fb else f"{'---':>5}"
        fb_mem = f"{fb['mem_pct']:>4.0f}%" if fb else f"{'---':>5}"
        print(f"{gc_s:>4} {rp_s:>5} {r['mbs']:>4} | {fwd_tps} {fwd_gpu} {fwd_mem} | {fb_tps} {fb_gpu} {fb_mem}")

    print(f"{'='*W}")


if __name__ == "__main__":
    main()
