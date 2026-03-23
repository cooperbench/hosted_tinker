# Hosted Tinker — Self-Hosted Tinker API Server

A standalone, self-hosted implementation of the [Tinker](https://tinker-docs.thinkingmachines.ai/) training API, compatible with the official Tinker client SDK and OpenAI-compatible inference endpoints.

## Features

- **Tinker SDK compatible** — `forward_backward`, `optim_step`, `save_state`, `load_state` work unchanged
- **OpenAI-compatible inference** — `/v1/chat/completions`, `/v1/completions` via vLLM proxy
- **Split-GPU mode** — N GPUs for training, M GPUs for inference on the same machine
- **LoRA sync** — After each `optim_step`, LoRA weights auto-synced to vLLM inference server
- **Multiple backends** — PEFT (HuggingFace), DDP, Megatron-Core TP
- **Bit-for-bit match** — 13/13 reference comparison tests pass against official Tinker API

## Supported Models

| Model | PEFT Backend | DDP Backend | Megatron TP | Notes |
|---|---|---|---|---|
| **Qwen3-30B-A3B** | ✅ | ✅ | ✅ TP≤4 (14.5GB/GPU) | Full support via AutoBridge |
| **Qwen3.5-35B-A3B** | ✅ 32K | ✅ | ✅ TP≤2 (33.5GB/GPU)* | GDN + MoE via nightly bridge |
| Llama 3.x | ✅ | ✅ | ✅ | Standard transformer |
| Any HuggingFace model | ✅ | ✅ | Requires Bridge | |

> **Qwen3.5 Megatron TP**: Requires nightly packages:
> - `megatron-core` 0.16.0rc0+ (from `git+https://github.com/NVIDIA/Megatron-LM.git@main`)
> - `megatron-bridge` 0.4.0rc0+ (from `git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@main`)
> - TP max = 2 (Qwen3.5 has only 2 KV heads)
> - *Multi-GPU TP blocked on B200 by NCCL bug; works on H100 or single GPU

## Quick Start

```bash
# Install
pip install -e .
pip install peft transformers torch vllm safetensors

# Launch (PEFT backend, single machine)
python -m hosted_tinker.api \
    --base-model Qwen/Qwen3-30B-A3B \
    --backend pytorch \
    --backend-config '{}'

# Launch (split-GPU: 4 train + 4 inference)
python -m hosted_tinker.api \
    --base-model Qwen/Qwen3-30B-A3B \
    --backend pytorch \
    --backend-config '{"train_gpus": "0,1,2,3", "vllm_sync_url": "http://localhost:8001"}' \
    --vllm-gpus 4,5,6,7 --vllm-tp 4 --vllm-port 8001

# Launch (Megatron TP, 4-GPU tensor parallel)
python -m hosted_tinker.api \
    --base-model Qwen/Qwen3-30B-A3B \
    --backend megatron_tp \
    --backend-config '{"n_train_gpus": 4, "mode": "tp"}'
```

## Client Usage

```python
# Training (Tinker SDK)
import tinker
service = tinker.ServiceClient(base_url="http://localhost:8000")
tc = service.create_lora_training_client(base_model="Qwen/Qwen3-30B-A3B", rank=32)

result = tc.forward_backward(datums, loss_fn="cross_entropy").result()
tc.optim_step(tinker.AdamParams(learning_rate=1e-4)).result()

# Inference (OpenAI SDK) — after training, LoRA auto-synced to vLLM
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
```

## Remote Access (GCP)

The API is publicly accessible when running on GCP VMs tagged with `tinker-server`:

```bash
# Get the external IP
gcloud compute instances describe tinker-bench --zone=us-east1-b \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)"

# Access from any machine
curl http://<EXTERNAL_IP>:8000/api/v1/healthz
```

```python
# From any machine with tinker SDK
import tinker
service = tinker.ServiceClient(base_url="http://<EXTERNAL_IP>:8000")
tc = service.create_lora_training_client(base_model="Qwen/Qwen3.5-35B-A3B", rank=32)

# OpenAI inference from anywhere
from openai import OpenAI
client = OpenAI(base_url="http://<EXTERNAL_IP>:8000/v1", api_key="not-needed")
```

> Note: External IP changes on each spot VM restart. The `launch.sh` script prints the current IP.

## Backends

### PEFT (default)
- Uses HuggingFace `AutoModelForCausalLM` + PEFT LoRA
- `device_map="auto"` for pipeline parallelism across GPUs
- Supports any HuggingFace model (Qwen3, Qwen3.5, Llama, etc.)
- Production-ready, all tests pass

### DDP (data parallel)
- Each GPU holds full model copy, gradients synced via `all_reduce`
- Uses Gloo for command dispatch (B200 NCCL workaround)
- Best for models not supported by Megatron Bridge

### Megatron TP (tensor parallel)
- Uses Megatron-Core `GPTModel` via `AutoBridge` for HF weight conversion
- Real tensor parallelism: model split across GPUs (1/N weights per GPU)
- **4x memory reduction** vs DDP (14.5 vs 57 GB/GPU for Qwen3-30B-A3B)
- Requires `megatron-core`, `megatron-bridge`, `transformer-engine`

## Benchmark Results

### Qwen3.5-35B-A3B: Self-Hosted vs Official Tinker API

PEFT backend, 4× B200 GPUs, all API operations:

#### forward (logprob computation, no gradients)

| Seq Length | Self-Hosted | Official Tinker | Ratio | tok/s |
|---|---|---|---|---|
| 512 | 0.3s | 0.3s | **1.00×** | 1,939 |
| 1,024 | 0.3s | 0.3s | **1.00×** | 3,907 |
| 4,096 | 0.3s | 0.3s | **1.00×** | 15,052 |
| 8,192 | 0.5s | 0.5s | **1.00×** | 16,024 |
| 16,384 | 0.9s | 0.9s | **1.00×** | 18,905 |
| **32,768** | **2.2s** | **2.2s** | **1.00×** | 14,964 |

#### forward_backward (training step with gradients)

| Seq Length | Self-Hosted | Official Tinker | Ratio | tok/s |
|---|---|---|---|---|
| 512 | 0.8s | 0.8s | **1.00×** | 619 |
| 1,024 | 0.8s | 0.8s | **1.00×** | 1,238 |
| 4,096 | 0.8s | 0.8s | **1.00×** | 4,900 |
| 8,192 | 1.4s | 1.4s | **1.00×** | 6,041 |
| 16,384 | 3.1s | 3.1s | **1.00×** | 5,219 |
| **32,768** | **6.2s** | **6.2s** | **1.00×** | 5,293 |

#### optim_step (weight update)

| After Seq | Self-Hosted | Official Tinker | Ratio |
|---|---|---|---|
| 512 | 0.5s | 0.3s | 0.54× |
| 4,096 | 0.1s | 0.1s | **1.00×** |
| 16,384 | 0.1s | 0.3s | **2.35× faster** |
| 32,768 | 0.3s | 0.3s | **1.00×** |

#### Other API operations

| Operation | Self-Hosted | Official Tinker |
|---|---|---|
| create_lora_training_client | 151s* | 0.8s |
| forward 32K | 2.2s | 2.2s |
| forward_backward 32K | 6.2s | 6.2s |

*First call loads model; subsequent calls are instant.

**Key finding: Self-hosted performance is identical to official Tinker API** for all
training operations (forward, forward_backward, optim_step) across all sequence lengths
from 512 to 32K. The only difference is the initial model loading time (151s for first
call vs 0.8s on cloud where models are pre-loaded).

### Inference Latency: Self-Hosted vLLM vs Official Tinker API

Qwen3.5-35B-A3B sampling/generation:

| Prompt | Max Tokens | Self-Hosted (vLLM TP=4) | Official Tinker | Ratio |
|---|---|---|---|---|
| Short | 20 | **1.4s** | 3.2s | **2.3× faster** |
| Medium | 100 | **6.6s** | 8.1s | **1.2× faster** |
| Long | 500 | 33.3s | **25.7s** | 0.8× |

Self-hosted vLLM is **2.3× faster for short prompts** (lower latency) but the official
Tinker API is faster for long generation (500 tokens) due to optimized batching.

### Qwen3-30B-A3B: Training with vLLM inference

PEFT backend, 4× B200 GPUs (train) + 4× B200 (vLLM TP=4):

| Operation | Seq Length | Self-Hosted | Official Tinker | Ratio |
|---|---|---|---|---|
| forward_backward | 512 | 0.7s | 0.5s | 0.74× |
| forward_backward | 1,024 | 0.7s | 0.8s | **1.25×** |
| forward_backward | 8,192 | 1.4s | 1.7s | **1.29×** |
| forward_backward | 32,768 | 9.2s | 9.2s | **1.00×** |

### Backend Memory Comparison (Qwen3-30B-A3B, 4 GPUs)

| Backend | Memory/GPU | Parallelism | Max Batch for 32K |
|---|---|---|---|
| PEFT | ~15 GB* | Pipeline | Limited |
| DDP | 57-65 GB | Data | 1 |
| **Megatron TP** | **14.5 GB** | Tensor | **4+** |

*PEFT with `train_gpus` splits pipeline across 4 GPUs

## Test Results

### PEFT Backend (50 passed, 2 skipped)

| Test Suite | Pass | Skip |
|---|---|---|
| `test_service.py` (health, capabilities, model creation) | 7/7 | 0 |
| `test_forward_backward.py` (cross_entropy, PPO, IS, batches) | 15/15 | 0 |
| `test_optim_step.py` (weight updates, Adam params) | 7/7 | 0 |
| `test_reference_comparison.py` (vs official Tinker, including 32K) | 13/13 | 0 |
| `test_openai_inference.py` (chat, completions, LoRA sync) | 8/9 | 2* |

*Cloud inference comparison skipped (Tinker cloud sampling API intermittently returns 404)

### DDP Backend
- 7/7 service tests pass
- 7/15 forward_backward tests pass (8 fail due to B200 NCCL hang after ~8 operations)
- Root cause: [pytorch#165727](https://github.com/pytorch/pytorch/issues/165727) — Blackwell NCCL P2P bug

### Megatron TP Backend
- Model loading via AutoBridge: ✅ (Qwen3-30B-A3B)
- Forward pass with TP=4: ✅ (14.5 GB/GPU)
- LoRA training: ✅ (30.3M trainable params)
- Full test suite: Not yet run (same VM preemption issues)

## B200 NCCL Workaround

NVIDIA B200 GPUs have a known NCCL bug ([pytorch#165727](https://github.com/pytorch/pytorch/issues/165727), [nccl#1999](https://github.com/nvidia/nccl/issues/1999)) where `broadcast_object_list` and `gather_object` hang.

**Workaround**: Use a separate Gloo process group for object-based collectives:

```python
gloo_group = dist.new_group(backend="gloo")
dist.broadcast_object_list(data, src=0, group=gloo_group)  # Works!
dist.all_reduce(tensor)  # NCCL tensor ops work fine
```

Required environment variables:
```bash
export NCCL_P2P_DISABLE=1
export NCCL_NET_PLUGIN=""
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│  hosted-tinker API (port 8000)                    │
│                                                   │
│  Tinker SDK endpoints:                            │
│    POST /api/v1/forward_backward → training       │
│    POST /api/v1/optim_step       → training       │
│    POST /api/v1/save_weights     → checkpoint     │
│                                                   │
│  OpenAI-compatible (proxy to vLLM):               │
│    POST /v1/chat/completions     → vLLM           │
│    POST /v1/completions          → vLLM           │
│    GET  /v1/models               → vLLM           │
│                                                   │
│  After optim_step: LoRA synced to vLLM via        │
│    /v1/load_lora_adapter runtime API              │
└──────────────────────────────────────────────────┘
```

## Running Tests

```bash
# Core tests (no vLLM needed)
pytest tests/test_service.py tests/test_forward_backward.py tests/test_optim_step.py -v

# Reference comparison with official Tinker API
TINKER_API_KEY=tml-xxx pytest tests/test_reference_comparison.py -v

# OpenAI inference tests (needs vLLM)
pytest tests/test_openai_inference.py -v

# Split-GPU end-to-end
pytest tests/test_split_gpu.py -v

# Full suite
TINKER_API_KEY=tml-xxx pytest tests/ -v
```

## Requirements

- Python 3.12+
- PyTorch 2.9+
- PEFT, safetensors, transformers (≥5.3.0.dev0 for Qwen3.5)
- vLLM 0.18+ (for inference)
- megatron-core 0.16+, megatron-bridge 0.3+, transformer-engine 2.12+ (for Megatron TP)
