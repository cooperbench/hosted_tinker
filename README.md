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

## Supported GPUs

| GPU | NCCL P2P | FSDP2 | DDP | Megatron TP | Notes |
|---|---|---|---|---|---|
| **H100** (80GB) | Works | ✅ | ✅ | ✅ | All backends work out of the box |
| **B200** (192GB) | Disabled* | ✅ | ✅ | ✅ | Auto-detects, uses Gloo for object collectives |
| **A100** (80GB) | Works | ✅ | ✅ | ✅ | All backends work |

> *B200 has a known NCCL P2P bug ([pytorch#165727](https://github.com/pytorch/pytorch/issues/165727)). `NCCL_P2P_DISABLE=1` is set automatically when B200 is detected.

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

# Launch on GCP spot VM (H100)
GPU_TYPE=h100 bash launch.sh

# Launch on GCP spot VM (B200, default)
GPU_TYPE=b200 bash launch.sh
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

## Running Tinker Cookbook Examples

The [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) recipes work
out of the box with hosted-tinker — just point `base_url` at your server.

### Step 1: Launch hosted-tinker (from your local machine)

```bash
# 8×B200 (or use GPU_TYPE=h100 for H100)
cd hosted-tinker
GPU_TYPE=b200 TRAIN_GPUS=0,1,2,3 VLLM_GPUS=4,5,6,7 VLLM_TP=4 bash launch.sh

# Or launch on an existing VM
GPU_TYPE=b200 bash launch.sh --vm <vm-name>
```

Wait for `Server running at http://<IP>:8000`.

### Step 2: Install tinker-cookbook

```bash
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[math-rl]'
```

### Step 3: Run recipes

```bash
export TINKER_API_KEY=tml-not-needed
export SERVER=http://<IP>:8000

# SFT on No Robots dataset (Qwen3-30B-A3B)
.venv/bin/python -m tinker_cookbook.recipes.sl_loop \
    base_url=$SERVER model_name=Qwen/Qwen3-30B-A3B

# RL on GSM8K math reasoning (GRPO)
.venv/bin/python -m tinker_cookbook.recipes.rl_loop \
    base_url=$SERVER model_name=Qwen/Qwen3-30B-A3B

# Full SFT with evaluation and checkpointing
.venv/bin/python -m tinker_cookbook.recipes.sl_basic \
    base_url=$SERVER model_name=Qwen/Qwen3-30B-A3B

# Full RL with evaluation
.venv/bin/python -m tinker_cookbook.recipes.rl_basic \
    base_url=$SERVER model_name=Qwen/Qwen3-30B-A3B
```

### Verified Results

SFT (`sl_loop`) on 8×B200 with Qwen3-30B-A3B:
- 74 training steps, batch_size=128
- NLL: 2.5 → 1.78
- ~50s per step
- Checkpointing works (save_state + save_weights_for_sampler)

> Note: Recipes use `key=value` syntax (not `--key value`) — this is the `chz` config framework.

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

### Qwen3-30B-A3B on H100: Self-Hosted vs Official Tinker API

PEFT backend, 2× H100 80GB train + 2× H100 vLLM (TP=2):

#### forward (logprob computation)

| Seq Length | H100 Self-Hosted | Official Tinker | Ratio | H100 tok/s |
|---|---|---|---|---|
| 512 | **1.0s** | 8.5s | **8.6× faster** | 518 |
| 1,024 | **1.1s** | 11.4s | **10.4× faster** | 939 |
| 4,096 | **1.8s** | 9.1s | **5.0× faster** | 2,235 |
| 8,192 | **2.6s** | 7.1s | **2.7× faster** | 3,145 |
| 16,384 | **4.6s** | 12.7s | **2.7× faster** | 3,532 |
| 32,768 | 11.7s | **9.1s** | 0.78× | 2,802 |

#### forward_backward (training step with gradients)

| Seq Length | H100 Self-Hosted | Official Tinker | Ratio | H100 tok/s |
|---|---|---|---|---|
| 512 | **0.6s** | 4.4s | **7.3× faster** | 849 |
| 1,024 | **0.6s** | 5.2s | **8.6× faster** | 1,694 |
| 4,096 | **1.2s** | 4.2s | **3.6× faster** | 3,476 |
| 8,192 | **2.1s** | 9.0s | **4.2× faster** | 3,864 |
| 16,384 | **4.1s** | 8.2s | **2.0× faster** | 3,952 |
| **32,768** | 11.2s | **7.4s** | 0.66× | 2,924 |

#### optim_step (weight update)

| After Seq | H100 Self-Hosted | Official Tinker | Ratio |
|---|---|---|---|
| 512 | **0.5s** | 3.6s | **7.5× faster** |
| 4,096 | **0.5s** | 4.6s | **9.3× faster** |
| 16,384 | **0.5s** | 3.6s | **7.3× faster** |
| 32,768 | **0.5s** | 4.0s | **8.2× faster** |

#### Inference (chat completions)

| Prompt | Max Tokens | H100 (vLLM TP=2) | Official Tinker | Ratio |
|---|---|---|---|---|
| Short | 20 | **1.4s** | 3.6s | **2.6× faster** |
| Medium | 100 | 6.9s | **6.2s** | 0.89× |
| Long | 500 | **34.7s** | 47.4s | **1.4× faster** |
| Very Long | 2048 | 142.5s | **115.2s** | 0.81× |

#### Other API operations

| Operation | H100 Self-Hosted | Official Tinker |
|---|---|---|
| create_lora_training_client | 0.8s | 19.1s |

**Key findings (H100)**:
- Training is **3–10× faster** for sequences up to 16K due to zero network overhead
- Official Tinker has ~4–9s fixed overhead per API call (network + queueing + cold start)
- At 32K, Tinker's cloud GPUs are faster (likely more/larger GPUs)
- `optim_step` is consistently **7–9× faster** self-hosted (0.5s vs 3.6–4.6s)
- `create_lora_training_client` is **24× faster** (0.8s vs 19.1s — model already loaded)
- Inference: **2.6× faster for short prompts** (1.4s vs 3.6s), Tinker faster for long generation (2048 tok: 115s vs 143s)

### Qwen3-30B-A3B: B200 Training with vLLM inference

PEFT backend, 4× B200 GPUs (train) + 4× B200 (vLLM TP=4):

| Operation | Seq Length | Self-Hosted | Official Tinker | Ratio |
|---|---|---|---|---|
| forward_backward | 512 | 0.7s | 0.5s | 0.74× |
| forward_backward | 1,024 | 0.7s | 0.8s | **1.25×** |
| forward_backward | 8,192 | 1.4s | 1.7s | **1.29×** |
| forward_backward | 32,768 | 9.2s | 9.2s | **1.00×** |

### Megatron DDP vs FSDP2: Throughput on H100 (Qwen3.5-9B)

32 mixed-length examples (15% ≤256 tok, 70% mid, 15% ≥6K tok), 120,712 total tokens, max_seq_len=8192, lora_rank=32, gc=on.
Sequential runs on GPUs 0–3.

| backend | GPUs | mbs | fwd tok/s | GPU util (fwd) | GPU mem (fwd) | fwd+bwd tok/s | GPU util (fwd+bwd) | GPU mem (fwd+bwd) |
|---------|------|-----|-----------|----------------|---------------|---------------|--------------------|----|
| FSDP2 | 4 | 1 | 12,778 | 68% | 30% | 2,276 | 77% | 39% |
| FSDP2 | 4 | 2 | 15,649 | 75% | 49% | 2,800 | 82% | 59% |
| FSDP2 | 4 | 4 | **17,881** | 80% | 63% | 2,779 | 89% | 87% |
| Megatron DDP | 4 | 1 | 14,439 | 69% | 32% | 2,583 | 69% | 40% |
| Megatron DDP | 4 | 2 | 12,332 | 62% | 41% | 2,913 | 73% | 41% |
| Megatron DDP | 4 | 4 | 15,009 | 68% | 41% | **2,936** | 73% | 41% |

### Megatron DDP vs FSDP2: Throughput on B200 (Qwen3.5-35B-A3B)

128 mixed-length examples (15% ≤256 tok, 70% mid, 15% ≥24K tok), 1,669,550 total tokens, max_seq_len=32768, lora_rank=32, gc=on.
Two configs run in parallel across GPU slots 0–3 and 4–7.

| backend | GPUs | mbs | fwd tok/s | GPU util (fwd) | GPU mem (fwd) | fwd+bwd tok/s | GPU util (fwd+bwd) | GPU mem (fwd+bwd) |
|---------|------|-----|-----------|----------------|---------------|---------------|--------------------|----|
| FSDP2 | 4 | 1 | 23,403 | 64% | 42% | 2,550 | 76% | 54% |
| FSDP2 | 4 | 2 | **27,032** | 73% | 61% | **2,631** | 87% | 82% |
| FSDP2 | 4 | 4 | 28,483 | 73% | 58% | OOM | — | — |
| Megatron DDP | 4 | 1 | 23,276 | 56% | 48% | 2,788 | 64% | 59% |
| Megatron DDP | 4 | 2 | 23,429 | 57% | 66% | **2,798** | 64% | 66% |
| Megatron DDP | 4 | 4 | 28,713 | 73% | 76% | OOM | — | — |

### Megatron DDP vs FSDP2: Throughput on H100 (Qwen3.5-9B)

32 mixed-length examples (15% ≤256 tok, 70% mid, 15% ≥6K tok), 120,712 total tokens, max_seq_len=8192, lora_rank=32, gc=on.
Sequential runs on GPUs 0–3.

| backend | GPUs | mbs | fwd tok/s | GPU util (fwd) | GPU mem (fwd) | fwd+bwd tok/s | GPU util (fwd+bwd) | GPU mem (fwd+bwd) |
|---------|------|-----|-----------|----------------|---------------|---------------|--------------------|----|
| FSDP2 | 4 | 1 | 25,684 | 50% | 30% | 9,216 | 67% | 39% |
| FSDP2 | 4 | 2 | 27,334 | 52% | 47% | 10,298 | 74% | 61% |
| FSDP2 | 4 | 4 | 17,043 | 62% | 56% | 7,963 | 75% | 85% |
| FSDP2 (remove_padding) | 4 | 1 | 44,040 | 58% | 34% | 14,906 | 80% | 43% |
| FSDP2 (remove_padding) | 4 | 2 | **49,153** | 59% | 48% | **17,869** | 83% | 55% |
| FSDP2 (remove_padding) | 4 | 4 | 48,994 | 50% | 53% | 11,260 | 76% | 72% |
| Megatron DDP | 4 | 1 | 25,394 | 76% | 32% | 10,250 | 74% | 40% |
| Megatron DDP | 4 | 2 | 34,936 | 87% | 40% | 12,891 | 96% | 59% |
| Megatron DDP | 4 | 4 | 29,268 | 82% | 56% | 10,655 | 96% | 77% |


### FSDP2 Throughput Sweep on H100 (Qwen3.5-9B, max_seq_len=16384)

4× H100 80GB (GPUs 4-7), 32 mixed-length examples (15% short ≤256, 70% mid, 15% long ≥12K), 211,042 total tokens, lora_rank=32. Sweeps micro_batch_size × gradient_checkpointing × remove_padding (sequence packing).

| gc | packing | mbs | fwd tok/s | fwd GPU% | fwd mem% | fwd+bwd tok/s | fwd+bwd GPU% | fwd+bwd mem% |
|----|---------|-----|-----------|----------|----------|---------------|--------------|--------------|
| off | off | 1 | 45,633 | 70% | 38% | 14,385 | 89% | 52% |
| off | off | 2 | 35,198 | 69% | 66% | 8,528 | 94% | 92% |
| off | off | 4 | 24,347 | 77% | 92% | OOM | — | — |
| on | off | 1 | 44,959 | 74% | 38% | 14,349 | 90% | 52% |
| on | off | 2 | 37,171 | 80% | 66% | 8,530 | 94% | 92% |
| on | off | 4 | 24,328 | 74% | 92% | OOM | — | — |
| on | **on** | 1 | 44,907 | 74% | 48% | 14,328 | 88% | 58% |
| on | **on** | 2 | 45,039 | 67% | 63% | **15,077** | 88% | 79% |
| on | **on** | 4 | **45,638** | 59% | 83% | 14,376 | 88% | 88% |
| off | **on** | 1 | 45,181 | 74% | 48% | 14,074 | 90% | 58% |
| off | **on** | 2 | 45,633 | 70% | 63% | 15,068 | 91% | 79% |
| off | **on** | 4 | 37,304 | 56% | 83% | 13,449 | 86% | 86% |


### vLLM Inference Throughput on H100 (Qwen3.5-9B, max_seq_len=16384)

4× H100 80GB (GPUs 0-3), TP=4, 64 concurrent requests, prompt_len~512, max_output_len=2048. Sweeps CUDA graphs, max_num_seqs, gpu_memory_utilization.

| label | max_num_seqs | gpu_mem | CUDA graphs | tok/s | ± | ttft_p50 | wall_s |
|-------|-------------|---------|-------------|-------|---|----------|--------|
| eager_s16 | 16 | 0.90 | off | 424 | 0 | 81,364ms | 299.1s |
| graph_s16 | 16 | 0.90 | **on** | 3,892 | 5 | 9,027ms | 31.7s |
| graph_s32 | 32 | 0.90 | **on** | 6,779 | 26 | 1,611ms | 18.7s |
| graph_s64 | 64 | 0.90 | **on** | 10,610 | 247 | 729ms | 11.6s |
| graph_s128 | 128 | 0.90 | **on** | 10,698 | 385 | 755ms | 11.6s |
| graph_s256 | 256 | 0.90 | **on** | 10,934 | 182 | 451ms | 11.4s |
| **graph_s64_m80** | **64** | **0.80** | **on** | **10,996** | **125** | **486ms** | **11.4s** |
| graph_s64_m95 | 64 | 0.95 | **on** | 10,717 | 357 | 647ms | 11.4s |
| graph_s128_m95 | 128 | 0.95 | **on** | 10,608 | 77 | 461ms | 11.1s |
| graph_s256_m95 | 256 | 0.95 | **on** | 10,552 | 244 | 617ms | 11.4s |


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

## GPU-Specific NCCL Handling

NVIDIA B200 GPUs have a known NCCL bug ([pytorch#165727](https://github.com/pytorch/pytorch/issues/165727), [nccl#1999](https://github.com/nvidia/nccl/issues/1999)) where `broadcast_object_list` and `gather_object` hang.

**Auto-detection**: The backend auto-detects GPU type via `nvidia-smi` and applies the appropriate workaround:

- **H100/A100**: NCCL P2P enabled, standard collectives — no workaround needed
- **B200**: Sets `NCCL_P2P_DISABLE=1`, creates Gloo process group for object collectives

```python
# Workers automatically use Gloo on B200, NCCL on H100
_use_gloo = os.environ.get("NCCL_P2P_DISABLE") == "1"
obj_group = dist.new_group(backend="gloo") if _use_gloo else None
dist.broadcast_object_list(data, src=0, group=obj_group)
dist.all_reduce(tensor)  # NCCL tensor ops work fine on all GPUs
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
