# Launch Tinker Server on 4x H100

Guide for launching hosted-tinker on a 4x H100 80GB node. Covers every config option.

## Command Structure

```bash
python -m hosted_tinker.api \
    --base-model <HF_MODEL_ID> \
    --backend <backend_name> \
    --backend-config '<json>' \
    [--vllm-gpus ...] [--port ...]
```

## CLI Flags

### Core

| Flag | Default | Meaning |
|------|---------|---------|
| `--base-model` | *required* | HuggingFace model ID (e.g. `Qwen/Qwen3.5-35B-A3B`). Must be downloaded first if offline. |
| `--backend` | `jax` | Training backend: `fsdp2`, `megatron_local`/`ddp`, `megatron_tp`, `pytorch`. |
| `--backend-config` | `{}` | JSON string with backend-specific settings. See tables below. |
| `--host` | `0.0.0.0` | Bind address. |
| `--port` | `8000` | API server port. |

### Database & Checkpoints

| Flag | Default | Meaning |
|------|---------|---------|
| `--database-url` | `sqlite:///tinker.db` | Request queue DB. Supports PostgreSQL. Env: `SKYRL_DATABASE_URL`. |
| `--checkpoints-base` | `/tmp/skyrl_checkpoints` | Where model checkpoints are stored. |

### Session Management

| Flag | Default | Meaning |
|------|---------|---------|
| `--session-cleanup-interval-sec` | `60` | Seconds between stale-session checks. `-1` disables. |
| `--session-timeout-sec` | `300` | Seconds without heartbeat before session is reaped. `-1` disables. |

### vLLM Split-GPU Mode

For dedicating separate GPUs to inference (e.g. GPUs 0-1 infer, GPUs 2-3 train):

| Flag | Default | Meaning |
|------|---------|---------|
| `--vllm-gpus` | `""` | GPU IDs for vLLM (e.g. `0,1`). Empty = no vLLM. |
| `--vllm-port` | `8001` | Port for vLLM subprocess. |
| `--vllm-tp` | `1` | vLLM tensor-parallel degree. Match number of `--vllm-gpus`. |
| `--vllm-max-model-len` | `32768` | Max sequence length vLLM accepts. |
| `--vllm-max-num-seqs` | `16` | Max concurrent sequences. |
| `--vllm-gpu-mem` | `0.90` | Fraction of GPU memory vLLM may use. |
| `--vllm-max-lora-rank` | `32` | Must match or exceed training LoRA rank. |

---

## Backend Configs (`--backend-config`)

### Megatron DDP (`--backend megatron_local` or `ddp`) — recommended

Full model copy on each GPU, gradients synced via all-reduce. Best throughput on H100.

| Field | Default | Meaning |
|-------|---------|---------|
| `n_train_gpus` | `4` | Number of training GPUs. |
| `train_gpu_offset` | `0` | First training GPU index. Non-zero to reserve lower GPUs for inference. |
| `micro_batch_size` | `1` | Sequences per forward pass. `mbs=2` optimal on H100 with gc=on. |
| `gradient_checkpointing` | `true` | Recompute activations in backward. Trades ~30% compute for ~40% memory. Required for mbs>=2. |
| `mode` | `ddp` | `"ddp"` = HF model + data parallel. `"tp"` = Megatron-Core tensor parallel (needs megatron-bridge). |
| `vllm_sync_url` | `null` | vLLM URL to push LoRA weights after `optim_step`. |
| `lora_sync_dir` | `/dev/shm/lora_adapters` | Tmpfs directory for LoRA weight handoff to vLLM. |

### FSDP2 (`--backend fsdp2`)

Shards parameters across GPUs. Lower per-GPU memory but all ranks must join every forward/backward.

| Field | Default | Meaning |
|-------|---------|---------|
| `n_train_gpus` | `6` | Number of training GPUs. Set to `4` on 4xH100. |
| `train_gpu_offset` | `2` | First training GPU. Set to `0` when all GPUs train. |
| `micro_batch_size` | `1` | Same as DDP. `mbs=2` fits with gc=on. `mbs=4` OOMs on fwd+bwd. |
| `gradient_checkpointing` | `true` | Same as DDP. |
| `loss_chunk_size` | `1024` | Chunk size for logprob computation. Smaller = less peak memory. `0` = no chunking. |
| `vllm_sync_url` | `null` | Same as DDP. |
| `lora_sync_dir` | `/dev/shm/lora_adapters` | Same as DDP. |

### PyTorch / PEFT (`--backend pytorch`)

Single-process HuggingFace + PEFT with `device_map="auto"`. Simplest, no distributed setup.

| Field | Default | Meaning |
|-------|---------|---------|
| `gradient_checkpointing` | `true` | Same as above. |
| `torch_dtype` | `bfloat16` | Model precision: `bfloat16`, `float16`, `float32`. H100 has native bf16 — use `bfloat16`. |
| `micro_batch_size` | `1` | Same as above. |
| `loss_chunk_size` | `1024` | Same as FSDP2. |
| `train_gpus` | `""` | Comma-separated GPU IDs (e.g. `"0,1,2,3"`). Empty = all. |
| `vllm_sync_url` | `null` | Same as above. |
| `lora_sync_dir` | `/dev/shm/lora_adapters` | Same as above. |

---

## Example Commands

### All 4 GPUs training (no inference)

```bash
# Megatron DDP — best throughput on H100
python -m hosted_tinker.api \
    --base-model Qwen/Qwen3.5-35B-A3B \
    --backend megatron_local \
    --backend-config '{"n_train_gpus": 4, "train_gpu_offset": 0, "micro_batch_size": 2, "gradient_checkpointing": true, "mode": "ddp"}'

# FSDP2 — lower memory per GPU
python -m hosted_tinker.api \
    --base-model Qwen/Qwen3.5-35B-A3B \
    --backend fsdp2 \
    --backend-config '{"n_train_gpus": 4, "train_gpu_offset": 0, "micro_batch_size": 2, "gradient_checkpointing": true}'
```

### Split-GPU (2 train + 2 infer)

```bash
python -m hosted_tinker.api \
    --base-model Qwen/Qwen3.5-35B-A3B \
    --backend fsdp2 \
    --backend-config '{"n_train_gpus": 2, "train_gpu_offset": 2, "micro_batch_size": 1, "gradient_checkpointing": true, "vllm_sync_url": "http://localhost:8001"}' \
    --vllm-gpus 0,1 --vllm-tp 2 --vllm-port 8001
```

---

## NCCL on GCP Single-Node VMs

GCP installs `nccl-gib` which adds `/usr/local/gib/lib64` to `LD_LIBRARY_PATH`. This shim crashes when `libibverbs.so` is missing (no InfiniBand on single-node VMs). The backends auto-strip it. If running manually:

```bash
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -iv gib | tr '\n' ':' | sed 's/:$//')
export NCCL_NET_PLUGIN=""
unset NCCL_NET
```

H100 does **not** need `NCCL_P2P_DISABLE=1` (B200-only workaround).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Silent crash, `Failed to open libibverbs.so` | Strip gIB from `LD_LIBRARY_PATH` (see above) |
| Forward 10x slower | `pip install causal-conv1d --no-build-isolation && pip install flash-linear-attention` |
| Port in use | `fuser -k 8000/tcp` or `--port 8001` |
| Can't connect to huggingface.co | Download model first, then it runs offline |
| `mbs=4` OOMs on fwd+bwd | Use `mbs=2`. Can use `mbs=4` for forward-only via `/admin/set_micro_batch_size` |
