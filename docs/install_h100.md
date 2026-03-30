# Installation Guide: 4x H100 Setup

This guide covers setting up hosted-tinker on a machine with 4x NVIDIA H100 80GB GPUs.

## Prerequisites

- Ubuntu 22.04 (or compatible)
- NVIDIA driver 550+ with CUDA 12.8+
- 4x H100 80GB GPUs
- Python 3.12+

## Step 1: Install Python 3.12

If your system only has Python 3.10, install 3.12 from the deadsnakes PPA:

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
```

## Step 2: Install system dependencies

```bash
sudo apt-get install -y pkg-config libsentencepiece-dev
```

## Step 3: Create virtual environment

```bash
cd hosted-tinker
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
```

## Step 4: Install PyTorch with CUDA

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Step 5: Install hosted-tinker and dependencies

```bash
.venv/bin/pip install -e ".[all]"
.venv/bin/pip install transformers peft safetensors tinker numpy
```

## Step 6: Install model-specific libraries

For Qwen3.5 models (required for efficient inference):

```bash
# causal-conv1d must be built from source (use --no-build-isolation to match CUDA versions)
.venv/bin/pip install causal-conv1d --no-build-isolation
.venv/bin/pip install flash-linear-attention
```

Without these libraries, Qwen3.5 falls back to a slow PyTorch implementation, making forward passes ~10x slower.

## Step 7: Download model weights

```bash
.venv/bin/python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-9B', dtype='auto')
"
```

## Step 8: Verify installation

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# Expected: 4x NVIDIA H100 80GB HBM3

.venv/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
import transformers, peft, tinker
print('All packages OK')
"
```

## Running the server

### FSDP2 backend (recommended)

```bash
.venv/bin/python -m hosted_tinker.api \
    --base-model Qwen/Qwen3.5-9B \
    --backend fsdp2 \
    --backend-config '{"n_train_gpus": 4, "train_gpu_offset": 0, "micro_batch_size": 2, "gradient_checkpointing": true}'
```

### Megatron DDP backend

```bash
.venv/bin/python -m hosted_tinker.api \
    --base-model Qwen/Qwen3.5-9B \
    --backend megatron_local \
    --backend-config '{"n_train_gpus": 4, "train_gpu_offset": 0, "micro_batch_size": 2, "gradient_checkpointing": true, "mode": "ddp"}'
```

## Running benchmarks

```bash
# FSDP2 benchmark (~5 min)
.venv/bin/python benchmarks/bench_backends.py \
    --base-model Qwen/Qwen3.5-9B \
    --backends fsdp2 \
    --micro-batch-sizes 1,2 \
    --gradient-checkpointing \
    --n-examples 32 \
    --warmup 1 --repeat 2

# Megatron DDP benchmark (~5 min)
.venv/bin/python benchmarks/bench_backends.py \
    --base-model Qwen/Qwen3.5-9B \
    --backends megatron_local \
    --micro-batch-sizes 1,2 \
    --gradient-checkpointing \
    --n-examples 32 \
    --warmup 1 --repeat 2
```

## H100-specific notes

- H100 GPUs work with all backends (FSDP2, DDP, Megatron TP) out of the box
- No NCCL P2P workarounds needed (unlike B200)
- 80GB HBM3 is sufficient for Qwen3.5-9B with gc=on at mbs=2
- Megatron DDP recommended over FSDP2 for better GPU utilization (89% vs 72%)

## Troubleshooting

### NCCL gIB crash on GCP VMs

GCP VMs set `NCCL_NET=gIB` and include `/usr/local/gib` in `LD_LIBRARY_PATH` for multi-node InfiniBand. On single-node VMs without InfiniBand hardware, this causes NCCL to crash or hang.

The backends auto-detect and strip these settings. If running manually:
```bash
unset NCCL_NET
export NCCL_NET_PLUGIN=""
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -iv gib | tr '\n' ':')
```

### Slow forward pass

If forward passes are ~10x slower than expected, install optimized kernels:
```bash
.venv/bin/pip install causal-conv1d --no-build-isolation
.venv/bin/pip install flash-linear-attention
```

## Expected performance (4x H100, Qwen3.5-9B, gc=on)

| Backend | mbs | Forward tok/s | Fwd+Bwd tok/s | GPU Util | Memory |
|---------|-----|---------------|---------------|----------|--------|
| Megatron DDP | 1 | 25,134 | 9,974 | 76% | 32% |
| **Megatron DDP** | **2** | **35,350** | **15,554** | **89%** | **40%** |
| FSDP2 | 1 | 25,591 | 8,621 | 72% | 40% |
| FSDP2 | 2 | 17,060 | 7,995 | 79% | 63% |
