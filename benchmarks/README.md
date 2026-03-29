# Benchmarks

Token throughput benchmarks for hosted-tinker backends on B200 GPUs.

## Scripts

| Script | Purpose |
|--------|---------|
| `bench_backends.py` | Unified sweep: runs multiple backends × mbs in parallel across two 4-GPU slots |
| `bench_gpu_throughput.py` | Single-server throughput: forward and fwd+bwd tok/s against a running server |
| `bench_micro_batch.py` | mbs sweep against a single running FSDP2 server (no restart between mbs values) |
| `bench_megatron_throughput.py` | Megatron-specific sweep across modes (ddp/tp) and mbs values |

---

## bench_backends.py — unified parallel benchmark

Starts two servers simultaneously (GPU 0–3 and GPU 4–7), sweeps backends × mbs, prints a summary table.

```bash
cd /path/to/hosted_tinker
source .venv/bin/activate

# Compare FSDP2 vs Megatron DDP, mbs=1 and mbs=2, gc=on
python benchmarks/bench_backends.py \
    --base-model Qwen/Qwen3.5-35B-A3B \
    --backends megatron_local,fsdp2 \
    --micro-batch-sizes 1,2 \
    --gradient-checkpointing \
    --n-examples 128 \
    --warmup 1 --repeat 3

# Quick iteration (warmup=0, repeat=1, no wait)
python benchmarks/bench_backends.py \
    --base-model Qwen/Qwen3.5-35B-A3B \
    --backends megatron_local,fsdp2 \
    --micro-batch-sizes 1,2,4 \
    --gradient-checkpointing \
    --warmup 0 --repeat 1

# Single backend only
python benchmarks/bench_backends.py \
    --backends fsdp2 \
    --micro-batch-sizes 1,2
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | `Qwen/Qwen3.5-35B-A3B` | HuggingFace model ID |
| `--backends` | `megatron_local,fsdp2` | Comma-separated backends |
| `--micro-batch-sizes` | `1,2` | mbs values to sweep |
| `--gradient-checkpointing` | off | Enable gc (reduces activation memory) |
| `--n-examples` | 128 | Total examples in dataset |
| `--warmup` | 1 | Warmup passes per (mbs, pass_type) |
| `--repeat` | 3 | Timed passes per (mbs, pass_type) |
| `--gc-backends` | — | Enable gc for specific backends only (overrides `--gradient-checkpointing`) |

**Output:**

```
        backend  gpus   mbs |   fwd tok/s   gpu%   mem% |  fwd+bwd tok/s   gpu%   mem%
          fsdp2     4     1 |       23403    64%    42% |           2550    76%    54%
          fsdp2     4     2 |       27032    73%    61% |           2631    87%    82%
 megatron_local     4     1 |       23276    56%    48% |           2788    64%    59%
 megatron_local     4     2 |       23429    57%    66% |           2798    64%    66%
```

---

## bench_gpu_throughput.py — single-server benchmark

Runs against an already-running server. Start the server first, then run the benchmark.

```bash
# 1. Start the server (example: FSDP2 backend, 4 GPUs, mbs=2, gc=on)
python -m hosted_tinker.api \
    --base-model Qwen/Qwen3.5-35B-A3B \
    --backend fsdp2 \
    --backend-config '{"n_train_gpus": 4, "micro_batch_size": 2, "gradient_checkpointing": true}'

# 2. In another terminal, run the benchmark
python benchmarks/bench_gpu_throughput.py \
    --url http://localhost:8000 \
    --model Qwen/Qwen3.5-35B-A3B \
    --gpu-ids 0,1,2,3
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `http://localhost:8000` | Server base URL |
| `--model` | `Qwen/Qwen3.5-35B-A3B` | Model name (must match server) |
| `--gpu-ids` | `0,1,2,3` | GPU indices to poll with nvidia-smi |
| `--n-examples` | 128 | Benchmark dataset size |
| `--warmup` / `--repeat` | 1 / 3 | Passes |

---

## bench_micro_batch.py — mbs sweep on a running FSDP2 server

Sweeps mbs values on an already-running server using `/admin/set_micro_batch_size` — no restart between values.

```bash
# 1. Start server
python -m hosted_tinker.api \
    --base-model Qwen/Qwen3.5-35B-A3B \
    --backend fsdp2 \
    --backend-config '{"n_train_gpus": 4, "micro_batch_size": 1}'

# 2. Sweep mbs=1,2,4
python benchmarks/bench_micro_batch.py \
    --url http://localhost:8000 \
    --micro-batch-sizes 1,2,4
```

---

## bench_megatron_throughput.py — Megatron DDP/TP sweep

Sweeps Megatron modes (ddp / tp) and mbs values across two parallel GPU slots.

```bash
# DDP sweep
python benchmarks/bench_megatron_throughput.py \
    --base-model Qwen/Qwen3.5-35B-A3B \
    --modes ddp \
    --micro-batch-sizes 1,2,4

# Compare DDP vs TP
python benchmarks/bench_megatron_throughput.py \
    --base-model Qwen/Qwen3.5-35B-A3B \
    --modes ddp,tp \
    --micro-batch-sizes 2
```

---

## Notes

- All scripts require `HF_HUB_OFFLINE=1` or a local model cache (set automatically by `bench_backends.py`).
- `bench_backends.py` creates isolated SQLite DBs per server port under `/tmp/` and cleans them up on restart.
- GPU memory polling uses `nvidia-smi`; ensure it is available in `PATH`.
- With `gc=on`, mbs=1 and mbs=2 fwd+bwd complete on B200. mbs=4 fwd+bwd OOMs on both backends (logits tensor bottleneck).
