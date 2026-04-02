"""vLLM inference throughput benchmark.

Launches vLLM on specified GPUs with different hyperparameter configurations
and measures output token throughput using concurrent requests.

Usage:
    python benchmarks/bench_vllm_inference.py \
        --model Qwen/Qwen3.5-9B \
        --gpus 0,1,2,3 \
        --max-output-len 2048 \
        --max-model-len 8192
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

import aiohttp
import numpy as np


@dataclass
class VLLMConfig:
    """A single vLLM configuration to benchmark."""
    tp: int
    max_num_seqs: int
    gpu_mem: float
    enforce_eager: bool
    label: str

    def __str__(self):
        eager = "eager" if self.enforce_eager else "graph"
        return f"tp={self.tp} max_seqs={self.max_num_seqs} gpu_mem={self.gpu_mem} {eager}"


def launch_vllm(
    model: str,
    gpu_ids: list[int],
    port: int,
    tp: int,
    max_num_seqs: int,
    gpu_mem: float,
    max_model_len: int,
    enforce_eager: bool,
) -> subprocess.Popen:
    """Launch a vLLM server and wait until ready."""
    vllm_python = os.environ.get("VLLM_PYTHON", sys.executable)
    cmd = [
        vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem),
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--trust-remote-code",
    ]
    if tp > 1:
        cmd.extend(["--tensor-parallel-size", str(tp)])
    if enforce_eager:
        cmd.append("--enforce-eager")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    # GCP non-RDMA VMs need Socket transport (gIB plugin fails without IB)
    env["NCCL_NET"] = "Socket"
    env.setdefault("NCCL_TUNER_CONFIG_PATH", "/usr/local/gib/configs")

    log_path = f"vllm_bench_{port}_{os.getpid()}.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, env=env,
    )
    print(f"  vLLM PID={proc.pid}, log={log_path}")

    # Wait for health
    import requests
    start = time.time()
    while time.time() - start < 600:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=5)
            if r.status_code == 200:
                print(f"  vLLM ready in {time.time() - start:.0f}s")
                return proc
        except requests.ConnectionError:
            pass
        if proc.poll() is not None:
            # Print last few lines of log for debugging
            log_file.flush()
            try:
                with open(log_path) as f:
                    lines = f.readlines()
                    print("  vLLM crashed! Last 20 log lines:")
                    for line in lines[-20:]:
                        print(f"    {line.rstrip()}")
            except Exception:
                pass
            raise RuntimeError(f"vLLM exited with code {proc.returncode}")
        time.sleep(3)
    raise TimeoutError("vLLM not ready after 600s")


def kill_vllm(proc: subprocess.Popen):
    """Kill vLLM server."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def make_prompts(n: int, prompt_lens: list[int], rng: np.random.RandomState) -> list[str]:
    """Generate prompts of varying lengths using repeated words."""
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "hello", "world", "data", "model", "train", "test", "code", "run"]
    prompts = []
    for plen in prompt_lens:
        # Each word is ~1 token, generate roughly plen words
        prompt_words = [words[rng.randint(len(words))] for _ in range(plen)]
        prompts.append("Continue this story: " + " ".join(prompt_words))
    return prompts


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Send a single completion request and measure timing."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    t0 = time.time()
    ttft = None
    output_tokens = 0

    # Use streaming to measure TTFT
    payload["stream"] = True
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            text = await resp.text()
            return {"error": text, "elapsed": 0, "ttft": 0, "output_tokens": 0}
        async for line in resp.content:
            decoded = line.decode().strip()
            if not decoded or not decoded.startswith("data: "):
                continue
            if decoded == "data: [DONE]":
                break
            if ttft is None:
                ttft = time.time() - t0
            try:
                chunk = json.loads(decoded[6:])
                choices = chunk.get("choices", [])
                if choices and choices[0].get("text"):
                    output_tokens += 1
            except json.JSONDecodeError:
                pass

    elapsed = time.time() - t0
    return {
        "elapsed": elapsed,
        "ttft": ttft or elapsed,
        "output_tokens": output_tokens,
    }


async def run_benchmark_async(
    port: int,
    model: str,
    prompts: list[str],
    max_output_len: int,
    concurrency: int,
) -> dict:
    """Run benchmark with given concurrency level."""
    url = f"http://localhost:{port}/v1/completions"
    sem = asyncio.Semaphore(concurrency)

    async def bounded_request(session, prompt):
        async with sem:
            return await send_request(session, url, model, prompt, max_output_len)

    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=600)) as session:
        t0 = time.time()
        tasks = [bounded_request(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)
        wall_time = time.time() - t0

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]

    if not ok:
        return {"error": f"All {len(errors)} requests failed", "wall_time": wall_time}

    total_output_tokens = sum(r["output_tokens"] for r in ok)
    ttfts = [r["ttft"] for r in ok]
    per_req_elapsed = [r["elapsed"] for r in ok]

    return {
        "n_requests": len(prompts),
        "n_ok": len(ok),
        "n_errors": len(errors),
        "concurrency": concurrency,
        "wall_time_s": wall_time,
        "total_output_tokens": total_output_tokens,
        "output_tok_per_s": total_output_tokens / wall_time,
        "ttft_mean_ms": np.mean(ttfts) * 1000,
        "ttft_p50_ms": np.median(ttfts) * 1000,
        "ttft_p99_ms": np.percentile(ttfts, 99) * 1000,
        "req_latency_mean_s": np.mean(per_req_elapsed),
        "req_latency_p50_s": np.median(per_req_elapsed),
        "req_latency_p99_s": np.percentile(per_req_elapsed, 99),
    }


def run_benchmark(port, model, prompts, max_output_len, concurrency):
    return asyncio.run(run_benchmark_async(port, model, prompts, max_output_len, concurrency))


def main():
    parser = argparse.ArgumentParser(description="vLLM inference throughput benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--gpus", default="0,1,2,3", help="GPU IDs (comma-separated)")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--max-output-len", type=int, default=2048,
                        help="Max tokens to generate per request")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max sequence length for vLLM")
    parser.add_argument("--n-requests", type=int, default=16,
                        help="Number of concurrent requests per run")
    parser.add_argument("--prompt-len", type=int, default=256,
                        help="Approximate prompt length in tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup requests before timed runs")
    parser.add_argument("--repeat", type=int, default=2,
                        help="Timed runs per configuration")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    n_gpus = len(gpu_ids)
    rng = np.random.RandomState(args.seed)

    # Generate prompts with varying lengths
    prompt_lens = rng.randint(
        max(64, args.prompt_len // 2),
        args.prompt_len * 2,
        size=args.n_requests,
    ).tolist()
    prompts = make_prompts(args.n_requests, prompt_lens, rng)

    # Small warmup prompts
    warmup_prompts = make_prompts(args.warmup, [128] * args.warmup, rng)

    # Configurations to test — all use tp=n_gpus for max throughput
    configs = []
    best_tp = n_gpus

    # 1. Eager baseline
    configs.append(VLLMConfig(
        tp=best_tp, max_num_seqs=16, gpu_mem=0.90,
        enforce_eager=True, label="eager_s16",
    ))

    # 2. CUDA graphs (key optimization)
    configs.append(VLLMConfig(
        tp=best_tp, max_num_seqs=16, gpu_mem=0.90,
        enforce_eager=False, label="graph_s16",
    ))

    # 3. max_num_seqs sweep with CUDA graphs
    for seqs in [32, 64, 128, 256]:
        configs.append(VLLMConfig(
            tp=best_tp, max_num_seqs=seqs, gpu_mem=0.90,
            enforce_eager=False, label=f"graph_s{seqs}",
        ))

    # 4. gpu_memory_utilization sweep with CUDA graphs + best seqs
    for mem in [0.80, 0.95]:
        configs.append(VLLMConfig(
            tp=best_tp, max_num_seqs=64, gpu_mem=mem,
            enforce_eager=False, label=f"graph_s64_m{int(mem*100)}",
        ))

    # 5. High concurrency + high mem
    configs.append(VLLMConfig(
        tp=best_tp, max_num_seqs=128, gpu_mem=0.95,
        enforce_eager=False, label="graph_s128_m95",
    ))
    configs.append(VLLMConfig(
        tp=best_tp, max_num_seqs=256, gpu_mem=0.95,
        enforce_eager=False, label="graph_s256_m95",
    ))

    all_results = []

    print(f"Model: {args.model}")
    print(f"GPUs: {gpu_ids} ({n_gpus} GPUs)")
    print(f"Max output len: {args.max_output_len}, Max model len: {args.max_model_len}")
    print(f"Requests per run: {args.n_requests}, Prompt len ~{args.prompt_len}")
    print(f"Configs to test: {len(configs)}")
    print()

    for i, cfg in enumerate(configs):
        print(f"{'='*70}")
        print(f"[{i+1}/{len(configs)}] {cfg.label}: {cfg}")
        print(f"{'='*70}")

        # Select GPU subset for this TP size
        use_gpus = gpu_ids[:cfg.tp]

        try:
            proc = launch_vllm(
                model=args.model,
                gpu_ids=use_gpus,
                port=args.port,
                tp=cfg.tp,
                max_num_seqs=cfg.max_num_seqs,
                gpu_mem=cfg.gpu_mem,
                max_model_len=args.max_model_len,
                enforce_eager=cfg.enforce_eager,
            )
        except (RuntimeError, TimeoutError) as e:
            print(f"  SKIP: {e}")
            all_results.append({"config": str(cfg), "label": cfg.label, "error": str(e)})
            continue

        try:
            # Warmup
            print(f"  Warming up ({args.warmup} requests, max_tokens=64)...")
            run_benchmark(args.port, args.model, warmup_prompts, 64, concurrency=args.warmup)

            # Timed runs
            run_results = []
            for r in range(args.repeat):
                print(f"  Run {r+1}/{args.repeat}...")
                res = run_benchmark(
                    args.port, args.model, prompts,
                    args.max_output_len, concurrency=args.n_requests,
                )
                if "error" in res:
                    print(f"    ERROR: {res['error']}")
                    continue
                run_results.append(res)
                print(f"    output_tok/s={res['output_tok_per_s']:.0f}  "
                      f"wall={res['wall_time_s']:.1f}s  "
                      f"ttft_p50={res['ttft_p50_ms']:.0f}ms  "
                      f"total_out_tok={res['total_output_tokens']}")

            if run_results:
                avg = {
                    "config": str(cfg),
                    "label": cfg.label,
                    "tp": cfg.tp,
                    "max_num_seqs": cfg.max_num_seqs,
                    "gpu_mem": cfg.gpu_mem,
                    "enforce_eager": cfg.enforce_eager,
                    "output_tok_per_s": np.mean([r["output_tok_per_s"] for r in run_results]),
                    "output_tok_per_s_std": np.std([r["output_tok_per_s"] for r in run_results]),
                    "wall_time_s": np.mean([r["wall_time_s"] for r in run_results]),
                    "ttft_mean_ms": np.mean([r["ttft_mean_ms"] for r in run_results]),
                    "ttft_p50_ms": np.mean([r["ttft_p50_ms"] for r in run_results]),
                    "ttft_p99_ms": np.mean([r["ttft_p99_ms"] for r in run_results]),
                    "total_output_tokens": np.mean([r["total_output_tokens"] for r in run_results]),
                    "n_errors": sum(r.get("n_errors", 0) for r in run_results),
                }
                all_results.append(avg)
            else:
                all_results.append({"config": str(cfg), "label": cfg.label, "error": "all runs failed"})

        finally:
            print("  Stopping vLLM...")
            kill_vllm(proc)
            time.sleep(5)  # let GPU memory release

    # Final summary
    print(f"\n{'='*90}")
    print(f"  BENCHMARK SUMMARY: {args.model} on GPUs {gpu_ids}")
    print(f"  max_output_len={args.max_output_len}  max_model_len={args.max_model_len}  "
          f"n_requests={args.n_requests}  prompt_len~{args.prompt_len}")
    print(f"{'='*90}")
    print(f"{'label':<16} {'config':<48} {'tok/s':>8} {'±':>6} {'ttft_p50':>10} {'wall_s':>8}")
    print(f"{'-'*90}")
    for r in all_results:
        if "error" in r:
            print(f"{r['label']:<16} {r['config']:<48} {'ERROR':>8}   {r.get('error','')[:30]}")
        else:
            print(f"{r['label']:<16} {r['config']:<48} "
                  f"{r['output_tok_per_s']:>7.0f} {r['output_tok_per_s_std']:>6.0f} "
                  f"{r['ttft_p50_ms']:>8.0f}ms {r['wall_time_s']:>7.1f}s")
    print()

    # Save JSON
    out_path = "benchmarks/vllm_bench_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "gpus": gpu_ids,
            "max_output_len": args.max_output_len,
            "max_model_len": args.max_model_len,
            "n_requests": args.n_requests,
            "prompt_len": args.prompt_len,
            "results": [{k: (v if not isinstance(v, np.floating) else float(v))
                         for k, v in r.items()} for r in all_results],
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
