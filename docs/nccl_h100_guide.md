# NCCL Installation & Troubleshooting Guide for 4x H100 Nodes

This guide covers correctly installing and configuring NCCL on a 4x NVIDIA H100 80GB node, with specific fixes for GCP single-node VMs.

## Environment

| Component | Tested Version |
|-----------|---------------|
| GPU | 4x NVIDIA H100 80GB HBM3 |
| Driver | 570.211.01+ |
| CUDA | 12.8 |
| PyTorch | 2.7.1+cu128 |
| NCCL | 2.26.2 (bundled with PyTorch) |
| OS | Ubuntu 22.04 |

## Quick Check

```bash
# Verify GPU setup
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

# Verify NCCL version
python3 -c "import torch; print('NCCL:', torch.cuda.nccl.version()); print('CUDA:', torch.version.cuda)"

# Check for gIB package (GCP)
dpkg -l | grep nccl-gib
```

## The GCP gIB Bug (Single-Node VMs)

### Symptoms

- NCCL operations (`all_reduce`, `broadcast`, etc.) crash silently with exit code 1
- No Python traceback — the process is killed by a native crash
- `NCCL_DEBUG=INFO` shows:
  ```
  NET/gIB : Initializing gIB v1.1.0
  Failed to open libibverbs.so[.1]
  ```
- Happens even with `NCCL_NET_PLUGIN=""` set

### Root Cause

GCP installs the `nccl-gib` package which places its libraries at `/usr/local/gib/lib64/`:
- `libnccl.so.2.27.5` (a modified NCCL with shim hooks)
- `libnccl-net.so` (shim plugin)
- `libnccl-net_internal.so` (gIB transport)

This path is added to `LD_LIBRARY_PATH` automatically. The shim plugin loads regardless of `NCCL_NET_PLUGIN=""` because it's baked into the modified `libnccl.so` that gets loaded first via `LD_LIBRARY_PATH`. When it tries to initialize gIB, it needs `libibverbs.so` (InfiniBand user-space library), which is not installed on single-node VMs without RDMA networking.

### Fix

Strip `/usr/local/gib/lib64` from `LD_LIBRARY_PATH` **before** launching any NCCL process:

```bash
# Remove gIB paths from LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/usr/local/gib' | tr '\n' ':' | sed 's/:$//')

# Also clear these for good measure
export NCCL_NET_PLUGIN=""
unset NCCL_NET
```

In Python (for subprocess launches):

```python
import os

env = os.environ.copy()
env["NCCL_NET_PLUGIN"] = ""
env.pop("NCCL_NET", None)

# Strip gIB from LD_LIBRARY_PATH
raw_ldpath = env.get("LD_LIBRARY_PATH", "")
fixed_ldpath = ":".join(
    p for p in raw_ldpath.split(":") if "gib" not in p.lower() and p
)
env["LD_LIBRARY_PATH"] = fixed_ldpath
```

This is already applied in `hosted_tinker`'s FSDP2, Megatron, and vLLM backends.

### Alternative: Install libibverbs

If you actually have RDMA/InfiniBand hardware (multi-node clusters):

```bash
sudo apt-get install -y libibverbs1 libibverbs-dev
```

This lets gIB initialize properly. Only do this if you need multi-node NCCL communication over RDMA.

## H100 vs B200: Key Differences

| Feature | H100 | B200 |
|---------|------|------|
| NCCL P2P | Works | Broken (pytorch#165727) |
| `NCCL_P2P_DISABLE=1` | Not needed | Required |
| Object collectives | NCCL works | Need Gloo fallback |
| `send()`/`recv()` | Works | Broken without P2P disable |

On H100, **do not** set `NCCL_P2P_DISABLE=1` — it works correctly and disabling P2P would reduce NVLink bandwidth.

## NCCL Version Compatibility

| NCCL Version | H100 Support | Notes |
|-------------|-------------|-------|
| < 2.15 | No | No Hopper support |
| 2.15 - 2.17 | Basic | Early support |
| 2.18.1+ | Good | LL128 corruption fix for HGX H100 |
| 2.22.3+ | Recommended | Fast NVLink detection, PCI P2P on split switches |
| 2.24+ | Production | cuMem host allocations, RAS monitoring |
| 2.26.2 | Current | Bundled with PyTorch 2.7.1+cu128 |

**Avoid**: NCCL 2.18.5 with CUDA 12.1 (dropped support), NCCL 2.19.1-2.19.4 at 16+ node scale (init hang).

## Installing NCCL

### Option 1: Use PyTorch-bundled NCCL (recommended)

PyTorch ships with a compatible NCCL. No separate installation needed:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
python3 -c "import torch; print(torch.cuda.nccl.version())"
# (2, 26, 2)
```

### Option 2: System package (APT)

```bash
# Add NVIDIA CUDA repo if not present
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install NCCL matching your CUDA version
sudo apt-get install -y libnccl2=2.26.2-1+cuda12.8 libnccl-dev=2.26.2-1+cuda12.8
```

### Option 3: Build from source (for H100-optimized binary)

```bash
git clone https://github.com/NVIDIA/nccl.git
cd nccl
git checkout v2.26.2-1

# Build only for Hopper (sm_90) — faster compilation
make -j$(nproc) src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

# Install
sudo make install PREFIX=/usr/local/nccl
echo "/usr/local/nccl/lib" | sudo tee /etc/ld.so.conf.d/nccl.conf
sudo ldconfig
```

## Environment Variables Reference

### Required on GCP Single-Node

```bash
export NCCL_NET_PLUGIN=""          # Disable GCP network plugin
unset NCCL_NET                     # Clear GCP network override
# + strip /usr/local/gib from LD_LIBRARY_PATH (see fix above)
```

### Debugging

```bash
export NCCL_DEBUG=WARN             # Warnings only (default for production)
export NCCL_DEBUG=INFO             # Verbose — use for troubleshooting
export NCCL_DEBUG_SUBSYS=ALL       # All subsystems
```

### Performance Tuning (H100)

```bash
# NVLS (NVLink SHARP) — enabled by default on H100
export NCCL_NVLS_ENABLE=1

# Memory allocator — enabled by default with CUDA >= 12.6 driver
export NCCL_CUMEM_HOST_ENABLE=1

# Expandable segments for PyTorch memory allocator
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

### Variables to NOT Set on H100

```bash
# DO NOT set these on H100 — they degrade performance:
# NCCL_P2P_DISABLE=1        # Disables NVLink P2P (only needed on B200)
# NCCL_IB_DISABLE=1         # Disables InfiniBand (only if no IB hardware)
# NCCL_PROTO=Simple,LL      # Forces suboptimal protocol
# NCCL_ALGO=Ring             # Forces suboptimal algorithm
```

NVIDIA recommends **not** cargo-culting environment variables. NCCL has built-in topology detection and tuning that automatically selects optimal parameters.

## Verification Test

Save this as `test_nccl.py` and run with:

```bash
# First, fix LD_LIBRARY_PATH (GCP)
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/usr/local/gib' | tr '\n' ':' | sed 's/:$//')
export NCCL_NET_PLUGIN=""
unset NCCL_NET

# Run on all 4 GPUs
python3 -m torch.distributed.run --nproc_per_node=4 test_nccl.py
```

```python
import os, torch, torch.distributed as dist
from datetime import timedelta

def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    os.environ["NCCL_NET_PLUGIN"] = ""
    os.environ.pop("NCCL_NET", None)
    os.environ.pop("TORCH_NCCL_ASYNC_ERROR_HANDLING", None)

    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"NCCL {torch.cuda.nccl.version()}, "
              f"CUDA {torch.version.cuda}, "
              f"PyTorch {torch.__version__}, "
              f"{world_size} GPUs")

    # all_reduce
    t = torch.ones(1024, device=device) * (rank + 1)
    dist.all_reduce(t)
    expected = sum(range(1, world_size + 1))
    assert torch.allclose(t, torch.full_like(t, expected))
    if rank == 0: print("[PASS] all_reduce")

    # broadcast
    t = torch.full((1024,), 42.0, device=device) if rank == 0 else torch.zeros(1024, device=device)
    dist.broadcast(t, src=0)
    assert torch.allclose(t, torch.full_like(t, 42.0))
    if rank == 0: print("[PASS] broadcast")

    # all_gather
    t = torch.full((1024,), float(rank), device=device)
    gathered = [torch.zeros(1024, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    if rank == 0: print("[PASS] all_gather")

    # reduce_scatter
    input_list = [torch.full((1024,), float(rank + i), device=device) for i in range(world_size)]
    output = torch.zeros(1024, device=device)
    dist.reduce_scatter(output, input_list)
    if rank == 0: print("[PASS] reduce_scatter")

    # P2P send/recv
    if world_size >= 2:
        if rank == 0:
            dist.send(torch.full((1024,), 99.0, device=device), dst=1)
        elif rank == 1:
            t = torch.zeros(1024, device=device)
            dist.recv(t, src=0)
            assert torch.allclose(t, torch.full_like(t, 99.0))
        dist.barrier()
        if rank == 0: print("[PASS] P2P send/recv")

    # object collectives (broadcast_object_list)
    objects = [{"test": "data"}] if rank == 0 else [None]
    dist.broadcast_object_list(objects, src=0)
    assert objects[0]["test"] == "data"
    if rank == 0: print("[PASS] broadcast_object_list")

    if rank == 0: print("\n=== All NCCL tests passed! ===")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Silent crash (exit code 1, no traceback)

Almost always the gIB plugin. Check with `NCCL_DEBUG=INFO` — if you see `Failed to open libibverbs.so`, apply the LD_LIBRARY_PATH fix above.

### NCCL timeout (600s default)

```python
# Increase timeout for large model initialization
dist.init_process_group("nccl", timeout=timedelta(seconds=1800))
```

### "No device id is provided" warning

Harmless warning from PyTorch 2.7+. Suppress by passing device_id to init:

```python
dist.init_process_group("nccl", timeout=timedelta(seconds=1800),
                        device_id=torch.device(f"cuda:{local_rank}"))
```

### Docker / container considerations

```bash
docker run \
    --gpus all \
    --shm-size=10g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NCCL_NET_PLUGIN="" \
    your-image
```

### Checking NVLink topology

```bash
nvidia-smi topo -m
# H100 should show NV18 (NVLink 4.0) between all GPU pairs
```
