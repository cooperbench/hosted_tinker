# Setup NCCL for Multi-GPU

Guide the user through fixing NCCL for multi-GPU (TP/DDP) on GCP VMs.

## Diagnosis

Run this to check if NCCL multi-GPU works:

```bash
python3 -c "
import torch, torch.distributed as dist, torch.multiprocessing as mp, os
def worker(rank, ws):
    os.environ['MASTER_ADDR']='127.0.0.1'; os.environ['MASTER_PORT']='29500'
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

If it fails with `Failed to initialize any NET plugin`, follow the fix below.

## Root cause

GCP GPU VMs ship with `nccl-gib` which writes `/etc/nccl.conf` containing `NCCL_NET=gIB`. This forces NCCL to use only the gIB (Google InfiniBand) transport. On single-node VMs without RDMA hardware (e.g. `a3-highgpu-8g`), gIB finds no devices and NCCL fails entirely — even though P2P/NVLink is available for intra-node communication.

## Fix

### 1. Install libibverbs (prevents silent crash)

```bash
sudo apt-get install -y libibverbs1 ibverbs-providers
```

### 2. Fix /etc/nccl.conf

Back up and replace the gIB config:

```bash
sudo cp /etc/nccl.conf /etc/nccl.conf.bak
sudo tee /etc/nccl.conf > /dev/null << 'EOF'
# Fixed for non-RDMA VMs (a3-highgpu, etc.)
# Removed NCCL_NET=gIB which forces gIB-only transport
# NCCL auto-detects P2P/SHM/NVLink for intra-node communication
NCCL_TUNER_CONFIG_PATH=/usr/local/gib/configs
EOF
```

### 3. Clear stale env vars from current shell

The gIB package also sets env vars via shell profiles. Unset them:

```bash
unset NCCL_NET NCCL_CROSS_NIC NCCL_NET_GDR_LEVEL NCCL_P2P_NET_CHUNKSIZE
unset NCCL_NVLS_CHUNKSIZE NCCL_IB_ADAPTIVE_ROUTING NCCL_IB_QPS_PER_CONNECTION
unset NCCL_IB_TC NCCL_IB_FIFO_TC
export NCCL_TUNER_CONFIG_PATH=/usr/local/gib/configs
```

### 4. Verify

Re-run the diagnosis script above. Should print `NCCL multi-GPU OK`.

## When does this apply?

- **Affected**: single-node GCP VMs without RDMA (`a3-highgpu-8g`, etc.) that have `nccl-gib` installed
- **Not affected**: multi-node VMs with RDMA hardware (`a3-ultragpu`, `a3-megagpu`), or non-GCP machines
- **Check**: run `ibv_devices` — if it shows no devices, you need this fix
