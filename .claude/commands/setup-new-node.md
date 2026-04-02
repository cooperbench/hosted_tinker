# Setup New Node for Hosted Tinker

Complete guide for setting up a new GCP GPU node (H100 or B200) to run hosted-tinker with FSDP2 remove_padding at full throughput.

## Prerequisites

- 4+ NVIDIA GPUs (H100 80GB or B200 192GB)
- Ubuntu 22.04+ with NVIDIA driver 570+
- Python 3.12+ (use `uv` to install if system Python is older)

## Step 1: Fix NCCL (GCP VMs only)

GCP GPU VMs ship with `nccl-gib` which forces the gIB (Google InfiniBand) transport. On single-node VMs without RDMA hardware, this crashes NCCL. Fix:

### 1a. Install libibverbs

```bash
sudo apt-get install -y libibverbs1 ibverbs-providers
```

### 1b. Fix /etc/nccl.conf

```bash
sudo cp /etc/nccl.conf /etc/nccl.conf.bak
sudo tee /etc/nccl.conf > /dev/null << 'EOF'
# Fixed for non-RDMA VMs (a3-highgpu, etc.)
# Removed NCCL_NET=gIB which forces gIB-only transport
NCCL_TUNER_CONFIG_PATH=/usr/local/gib/configs
EOF
```

### 1c. Strip gIB env vars from current shell

```bash
unset NCCL_NET NCCL_CROSS_NIC NCCL_NET_GDR_LEVEL NCCL_P2P_NET_CHUNKSIZE
unset NCCL_NVLS_CHUNKSIZE NCCL_IB_ADAPTIVE_ROUTING NCCL_IB_QPS_PER_CONNECTION
unset NCCL_IB_TC NCCL_IB_FIFO_TC NCCL_TUNER_CONFIG_PATH
export NCCL_NET_PLUGIN=""
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -iv gib | tr '\n' ':' | sed 's/:$//')
```

### 1d. Verify NCCL works

```bash
python3 -c "
import torch, torch.distributed as dist, torch.multiprocessing as mp, os
def worker(rank, ws):
    os.environ['MASTER_ADDR']='127.0.0.1'; os.environ['MASTER_PORT']='29501'
    dist.init_process_group('nccl', world_size=ws, rank=rank)
    t = torch.ones(10, device=f'cuda:{rank}')
    dist.all_reduce(t)
    print(f'Rank {rank}: OK, result={t[0].item()}')
    dist.destroy_process_group()
if __name__=='__main__':
    mp.set_start_method('fork', force=True)
    ps = [mp.Process(target=worker, args=(r, 2)) for r in range(2)]
    [p.start() for p in ps]; [p.join() for p in ps]
    assert all(p.exitcode==0 for p in ps), 'NCCL FAILED'
    print('NCCL multi-GPU OK')
"
```

**Note:** The hosted-tinker FSDP2 and Megatron backends auto-strip gIB from `LD_LIBRARY_PATH` and clear `NCCL_NET` when launching worker subprocesses. But the env must be clean for manual torchrun launches and NCCL tests.

## Step 2: Create Python 3.12 venv

System Python on Ubuntu 22.04 is 3.10, but `tinker` SDK requires 3.11+.

```bash
pip3 install uv
uv venv --python 3.12 .venv
```

## Step 3: Install PyTorch with CUDA

```bash
uv pip install --python .venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
.venv/bin/python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'NCCL:', torch.cuda.nccl.version(), 'GPUs:', torch.cuda.device_count())"
```

## Step 4: Install hosted-tinker dependencies

```bash
uv pip install --python .venv/bin/python \
    peft transformers safetensors tinker numpy requests \
    fastapi sqlmodel sqlalchemy aiosqlite httpx psutil \
    rich pydantic cloudpathlib uvicorn accelerate wheel
```

## Step 5: Install flash-linear-attention + causal-conv1d (CRITICAL)

**This is the most important step for Qwen3.5 models.** Without these libraries, the GDN (Gated Delta Networks) layers in Qwen3.5 fall back to a slow torch implementation, causing **~7x lower throughput**.

```bash
uv pip install --python .venv/bin/python causal-conv1d --no-build-isolation
uv pip install --python .venv/bin/python flash-linear-attention --no-build-isolation
```

Verify:

```bash
.venv/bin/python -c "
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.modules import FusedRMSNormGated
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
print('All GDN fast-path imports OK')
"
```

**How to tell if it's missing:** Worker logs show:
```
The fast path is not available because one of the required library is not installed.
Falling back to torch implementation.
```

## Step 6: Install flash-attn (required for remove_padding)

```bash
uv pip install --python .venv/bin/python flash-attn --no-build-isolation
```

Verify:

```bash
.venv/bin/python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)"
```

## Step 7: Download model

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-9B')  # or Qwen/Qwen3.5-35B-A3B for larger model
print('Done')
"
```

## Step 8: Launch server with remove_padding

```bash
PYTHONPATH=/path/to/hosted_tinker .venv/bin/python -m hosted_tinker.api \
    --base-model Qwen/Qwen3.5-9B \
    --backend fsdp2 \
    --backend-config '{"n_train_gpus": 4, "train_gpu_offset": 0, "micro_batch_size": 2, "gradient_checkpointing": true, "remove_padding": true}' \
    --port 8000
```

## Step 9: Verify throughput

```bash
PYTHONPATH=/path/to/hosted_tinker TINKER_API_KEY=tml-dummy \
    .venv/bin/python benchmarks/bench_gpu_throughput.py \
    --url http://localhost:8000 \
    --model Qwen/Qwen3.5-9B \
    --lora-rank 32 \
    --n-examples 32 \
    --min-seq-len 64 \
    --max-seq-len 8192 \
    --warmup 3 \
    --repeat 5 \
    --gpu-ids 0,1,2,3
```

### Expected throughput (FSDP2 remove_padding, 4x H100, Qwen3.5-9B)

| mbs | fwd tok/s | fwd+bwd tok/s | GPU mem (fwd+bwd) |
|-----|-----------|---------------|-------------------|
| 1   | ~44,000   | ~15,000       | 43%               |
| 2   | **~49,000** | **~18,000** | 55%               |
| 4   | ~49,000   | ~11,000       | 72% (may OOM)     |

**If throughput is far below these numbers:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| fwd+bwd ~2,000 tok/s (7x slower) | Missing fla/causal-conv1d | Step 5 |
| fwd ~18,000 tok/s (2.5x slower) | Missing fla/causal-conv1d | Step 5 |
| Workers crash silently | NCCL gIB crash | Step 1 |
| `Error: network gIB not found` | gIB env vars not stripped | Step 1c |
| mbs=4 fwd+bwd OOMs | Normal — use mbs=2 | Set `micro_batch_size: 2` |
| Flash attention error | Missing flash-attn | Step 6 |

## Changing micro_batch_size at runtime

No server restart needed:

```bash
curl -s -X POST http://localhost:8000/admin/set_micro_batch_size \
    -H "Content-Type: application/json" -d '{"n": 2}'
```

## GPU-specific notes

| GPU | NCCL P2P | Extra env needed |
|-----|----------|-----------------|
| H100 | Works | None (default) |
| B200 | Broken ([pytorch#165727](https://github.com/pytorch/pytorch/issues/165727)) | `NCCL_P2P_DISABLE=1` (auto-set by backend) |
| A100 | Works | None |
