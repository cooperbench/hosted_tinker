# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Run the server (PEFT backend)
```bash
python -m hosted_tinker.api \
    --base-model <HF_MODEL_ID> \
    --backend pytorch \
    --backend-config '{}'
```

### Run the server (split-GPU: N train + M inference)
```bash
python -m hosted_tinker.api \
    --base-model <HF_MODEL_ID> \
    --backend pytorch \
    --backend-config '{"train_gpus": "0,1,2,3", "vllm_sync_url": "http://localhost:8001"}' \
    --vllm-gpus 4,5,6,7 --vllm-tp 4 --vllm-port 8001
```

### Run tests (no vLLM required)
```bash
pytest tests/test_service.py tests/test_forward_backward.py tests/test_optim_step.py -v
```

### Run all tests (with official Tinker API comparison)
```bash
TINKER_API_KEY=tml-xxx pytest tests/ -v
```

### Lint
```bash
ruff check hosted_tinker/
ruff format hosted_tinker/
```

## Architecture

**Hosted Tinker** is a self-hosted implementation of the Tinker training API, compatible with the Tinker SDK client and OpenAI-compatible inference endpoints.

### Request lifecycle
1. FastAPI server (`api.py`) receives Tinker SDK calls (`/api/v1/forward_backward`, `/api/v1/optim_step`, etc.)
2. Requests are queued as `FutureDB` entries in SQLite/PostgreSQL
3. Background `Engine` (`engine.py`) dequeues and dispatches to the active backend
4. Backend returns results; engine writes them back to DB
5. Client polls for results by request_id

### Backends (`backend.py` defines the interface)
- **PyTorchBackend** (`pytorch_backend.py`): HuggingFace + PEFT, `device_map="auto"` — default, most compatible
- **FSDP2Backend** (`fsdp2_backend.py` + `fsdp2_worker.py`): DDP via forked subprocess
- **MegatronBackend** (`megatron_backend.py` + workers): Tensor-parallel via Megatron-Core; requires megatron-bridge for weight conversion

### Inference / split-GPU mode
- `vllm_manager.py` launches vLLM as a subprocess on separate GPUs
- After each `optim_step`, LoRA weights are written to `/dev/shm/lora_adapters/` and synced to vLLM via its runtime API
- OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`) proxy to vLLM

### Loss functions (`loss_fns.py`)
`cross_entropy_loss`, `importance_sampling_loss`, `ppo_loss` — all support token-level masking.

## GPU quirks

- B200 has a known NCCL P2P bug (pytorch#165727): `NCCL_P2P_DISABLE=1` is set automatically when B200 is detected
- B200 also uses a Gloo fallback process group for object collectives; tensor ops still use NCCL
- GPU type is detected via `nvidia-smi` at startup

## Database

SQLite (default, WAL mode) or PostgreSQL. Schema managed via Alembic (`alembic.ini`). Key tables: `ModelDB`, `FutureDB`, `CheckpointDB`, `SessionDB`.
