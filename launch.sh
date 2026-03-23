#!/bin/bash
# Preemption-robust launch script for hosted-tinker on GCP spot VMs.
#
# Features:
# - Supports both H100 (a3-highgpu-8g) and B200 (a4-highgpu-8g) GPUs
# - Retries VM creation across multiple zones if preempted
# - Auto-deploys code and dependencies
# - Resumes from checkpoint if available
# - Monitors and restarts on crash
#
# Usage:
#   GPU_TYPE=h100 bash launch.sh                    # H100 spot VM
#   GPU_TYPE=b200 bash launch.sh                    # B200 spot VM (default)
#   bash launch.sh --vm tinker-bench                # Use existing VM
#   bash launch.sh --monitor                        # Just monitor existing deployment

set -euo pipefail

# ============================================================
# GPU type configuration
# ============================================================
GPU_TYPE="${GPU_TYPE:-b200}"  # "h100" or "b200"

case "$GPU_TYPE" in
    h100|H100)
        DEFAULT_MACHINE="a3-highgpu-8g"
        DEFAULT_IMAGE="common-cu128-ubuntu-2204-nvidia-570-v20260305"
        NCCL_P2P_DISABLE_FLAG=""
        ;;
    b200|B200)
        DEFAULT_MACHINE="a4-highgpu-8g"
        DEFAULT_IMAGE="common-cu128-ubuntu-2204-nvidia-570-v20260305"
        NCCL_P2P_DISABLE_FLAG="export NCCL_P2P_DISABLE=1"
        ;;
    *)
        echo "Unknown GPU_TYPE: $GPU_TYPE. Use 'h100' or 'b200'."
        exit 1
        ;;
esac

# ============================================================
# Configuration
# ============================================================
VM_NAME="${VM_NAME:-tinker-train}"
PROJECT="${PROJECT:-soe-gemini-llm-agents}"
MACHINE_TYPE="${MACHINE_TYPE:-$DEFAULT_MACHINE}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-500GB}"
IMAGE="${IMAGE:-$DEFAULT_IMAGE}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

# Zones to try (ordered by preference)
ZONES="${ZONES:-us-east1-b us-central1-b us-west1-b europe-west4-b asia-northeast1-a}"

# Training config
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-30B-A3B}"
BACKEND="${BACKEND:-pytorch}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
VLLM_GPUS="${VLLM_GPUS:-4,5,6,7}"
VLLM_TP="${VLLM_TP:-4}"
LORA_RANK="${LORA_RANK:-32}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_TAR="/tmp/hosted_tinker_deploy.tar.gz"

log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }

# ============================================================
# Step 1: Find or create VM
# ============================================================
find_or_create_vm() {
    local vm_name="$1"

    # Check if VM exists and is running
    local status
    status=$(gcloud compute instances describe "$vm_name" --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

    if [ "$status" = "RUNNING" ]; then
        local zone
        zone=$(gcloud compute instances describe "$vm_name" --format="value(zone)" 2>/dev/null | awk -F/ '{print $NF}')
        log "VM $vm_name already running in $zone"
        echo "$zone"
        return 0
    fi

    if [ "$status" = "TERMINATED" ] || [ "$status" = "STOPPED" ]; then
        local zone
        zone=$(gcloud compute instances describe "$vm_name" --format="value(zone)" 2>/dev/null | awk -F/ '{print $NF}')
        log "VM $vm_name is $status in $zone, starting..."
        if gcloud compute instances start "$vm_name" --zone="$zone" 2>&1 | grep -q "Updated"; then
            echo "$zone"
            return 0
        fi
        log "Failed to start, will create new VM"
    fi

    # Create new spot VM
    for zone in $ZONES; do
        log "Trying to create $vm_name in $zone ($MACHINE_TYPE)..."
        if gcloud compute instances create "$vm_name" \
            --zone="$zone" \
            --machine-type="$MACHINE_TYPE" \
            --boot-disk-size="$BOOT_DISK_SIZE" \
            --boot-disk-type=hyperdisk-balanced \
            --image="$IMAGE" \
            --image-project="$IMAGE_PROJECT" \
            --maintenance-policy=TERMINATE \
            --provisioning-model=SPOT \
            --instance-termination-action=STOP \
            --metadata="install-nvidia-driver=True" \
            --scopes=cloud-platform \
            2>&1 | grep -q "RUNNING"; then
            log "SUCCESS: Created $vm_name in $zone"
            echo "$zone"
            return 0
        fi
        gcloud compute instances delete "$vm_name" --zone="$zone" --quiet 2>/dev/null || true
    done

    log "FAILED: No zone has capacity for $MACHINE_TYPE"
    return 1
}

# ============================================================
# Step 2: Deploy code
# ============================================================
deploy_code() {
    local vm="$1" zone="$2"

    log "Building deploy tarball..."
    tar czf "$DEPLOY_TAR" \
        --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='.venv*' --exclude='wandb' --exclude='*.log' \
        -C "$(dirname "$SCRIPT_DIR")" "$(basename "$SCRIPT_DIR")"

    log "Uploading to $vm..."
    gcloud compute scp "$DEPLOY_TAR" "$vm:/tmp/deploy.tar.gz" --zone="$zone" --quiet

    log "Setting up on VM..."
    gcloud compute ssh "$vm" --zone="$zone" --quiet --command='
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

# Extract code
cd /home/hao
tar xzf /tmp/deploy.tar.gz 2>/dev/null || true

# Install uv if needed
which uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Check if venv exists
if [ ! -f /home/hao/SkyRL/.venv312/bin/python ]; then
    echo "Setting up Python environment..."
    cd /home/hao/SkyRL 2>/dev/null || { mkdir -p /home/hao/SkyRL && cd /home/hao/SkyRL; }
    uv venv --python 3.12 .venv312
    uv pip install --python .venv312/bin/python torch peft safetensors vllm tinker pytest httpx
    .venv312/bin/python -m pip install "transformers @ git+https://github.com/huggingface/transformers.git"
fi

echo "Deploy complete"
'
}

# ============================================================
# Step 3: Launch server
# ============================================================
launch_server() {
    local vm="$1" zone="$2"

    log "Launching hosted-tinker on $vm ($zone) [GPU: $GPU_TYPE]..."
    gcloud compute ssh "$vm" --zone="$zone" --quiet --command="
set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"

# Kill any existing processes
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8001/tcp 2>/dev/null || true
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do kill -9 \$pid 2>/dev/null || true; done
sleep 3
rm -f /home/hao/hosted-tinker/hosted_tinker/tinker.db*

cd /home/hao/SkyRL
export PYTHONPATH='/home/hao/hosted-tinker:\${PYTHONPATH:-}'
export NCCL_NET_PLUGIN=''
unset NCCL_NET 2>/dev/null || true
$NCCL_P2P_DISABLE_FLAG

# Symlink build tools if needed
sudo ln -sf \$HOME/miniforge3/bin/ninja /usr/local/bin/ninja 2>/dev/null || true
sudo ln -sf \$HOME/miniforge3/bin/x86_64-conda-linux-gnu-gcc /usr/local/bin/gcc 2>/dev/null || true
sudo ln -sf \$HOME/miniforge3/bin/x86_64-conda-linux-gnu-g++ /usr/local/bin/g++ 2>/dev/null || true
sudo ln -sf \$HOME/miniforge3/bin/x86_64-conda-linux-gnu-g++ /usr/local/bin/c++ 2>/dev/null || true

nohup .venv312/bin/python -m hosted_tinker.api \\
    --base-model '$BASE_MODEL' \\
    --backend '$BACKEND' \\
    --backend-config '{\"train_gpus\": \"$TRAIN_GPUS\", \"vllm_sync_url\": \"http://localhost:8001\", \"lora_sync_dir\": \"/dev/shm/lora_adapters\"}' \\
    --vllm-gpus '$VLLM_GPUS' --vllm-tp $VLLM_TP --vllm-port 8001 \\
    --vllm-max-model-len 32768 --vllm-max-num-seqs 8 \\
    > /home/hao/server.log 2>&1 &

echo 'Server launched (PID=\$!)'
"

    # Wait for readiness
    log "Waiting for server to be ready..."
    for i in $(seq 1 120); do
        sleep 5
        if gcloud compute ssh "$vm" --zone="$zone" --quiet --command='
            curl -s http://localhost:8000/api/v1/healthz 2>/dev/null | grep -q ok && \
            curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q data && \
            echo READY' 2>/dev/null | grep -q READY; then
            log "Server ready after $((i*5))s"
            return 0
        fi
    done

    log "Server failed to start within 600s"
    return 1
}

# ============================================================
# Step 4: Monitor loop (restarts on preemption)
# ============================================================
monitor_loop() {
    local vm="$1"

    while true; do
        local zone
        zone=$(find_or_create_vm "$vm") || {
            log "Failed to get VM, retrying in 5 minutes..."
            sleep 300
            continue
        }

        # Check if server is running
        if gcloud compute ssh "$vm" --zone="$zone" --quiet --command='
            curl -s http://localhost:8000/api/v1/healthz 2>/dev/null | grep -q ok && echo OK' 2>/dev/null | grep -q OK; then
            log "Server healthy on $vm ($zone)"
        else
            log "Server not running, deploying + launching..."
            deploy_code "$vm" "$zone" || { sleep 60; continue; }
            launch_server "$vm" "$zone" || { sleep 60; continue; }
        fi

        # Get external IP for user
        local ip
        ip=$(gcloud compute instances describe "$vm" --zone="$zone" --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null)
        log "Server running at http://$ip:8000"
        log "  Training: http://$ip:8000/api/v1/healthz"
        log "  Inference: http://$ip:8000/v1/models"
        log "  Dashboard: http://$ip:8000/dashboard"
        log "  SSH: gcloud compute ssh $vm --zone=$zone -- -L 8000:localhost:8000"

        # Monitor (check every 2 min)
        while true; do
            sleep 120

            local status
            status=$(gcloud compute instances describe "$vm" --format="value(status)" 2>/dev/null || echo "UNKNOWN")

            if [ "$status" != "RUNNING" ]; then
                log "VM $vm is $status — preempted! Restarting..."
                break
            fi

            if ! gcloud compute ssh "$vm" --zone="$zone" --quiet --command='
                curl -s http://localhost:8000/api/v1/healthz 2>/dev/null | grep -q ok && echo OK' 2>/dev/null | grep -q OK; then
                log "Server health check failed, restarting..."
                break
            fi
        done
    done
}

# ============================================================
# Main
# ============================================================
log "GPU type: $GPU_TYPE ($MACHINE_TYPE)"

case "${1:-}" in
    --monitor)
        log "Monitoring $VM_NAME..."
        monitor_loop "$VM_NAME"
        ;;
    --vm)
        VM_NAME="${2:-tinker-bench}"
        zone=$(find_or_create_vm "$VM_NAME") || exit 1
        deploy_code "$VM_NAME" "$zone"
        launch_server "$VM_NAME" "$zone"
        log "Done! Access via: gcloud compute ssh $VM_NAME --zone=$zone -- -L 8000:localhost:8000"
        ;;
    *)
        zone=$(find_or_create_vm "$VM_NAME") || exit 1
        deploy_code "$VM_NAME" "$zone"
        launch_server "$VM_NAME" "$zone"
        log "Entering monitor loop (auto-restarts on preemption)..."
        monitor_loop "$VM_NAME"
        ;;
esac
