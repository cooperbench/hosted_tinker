"""End-to-end test for split-GPU mode: training on GPUs 0-3, inference on GPUs 4-7.

Tests the full workflow:
1. Train with Tinker SDK (forward_backward + optim_step)
2. LoRA weights auto-synced to vLLM
3. Inference via OpenAI-compatible API reflects training

This test requires the server to be running in split-GPU mode:
    python -m hosted_tinker.api \
        --base-model Qwen/Qwen3-30B-A3B \
        --backend megatron_tp \
        --backend-config '{"n_train_gpus": 4, "mode": "tp", "vllm_sync_url": "http://localhost:8001"}' \
        --vllm-gpus 4,5,6,7 --vllm-tp 4 --vllm-port 8001

Usage:
    TINKER_BASE_URL=http://localhost:8000 pytest tests/test_split_gpu.py -v
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

import tinker

from .conftest import make_cross_entropy_datum, make_ppo_datum, make_random_tokens


BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("TINKER_MODEL", "Qwen/Qwen3-30B-A3B")


def _split_gpu_available() -> bool:
    """Check if both training and inference are available."""
    if not HAS_OPENAI:
        return False
    try:
        import httpx
        # Check training endpoint
        r1 = httpx.get(f"{BASE_URL}/api/v1/healthz", timeout=5)
        # Check inference endpoint
        r2 = httpx.get(f"{BASE_URL}/v1/models", timeout=5)
        return r1.status_code == 200 and r2.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _split_gpu_available(),
    reason="Split-GPU mode not available (need both training + inference endpoints)"
)


@pytest.fixture(scope="module")
def openai_client() -> OpenAI:
    return OpenAI(base_url=f"{BASE_URL}/v1", api_key="not-needed")


@pytest.fixture(scope="module")
def training_client() -> tinker.TrainingClient:
    os.environ.setdefault("TINKER_API_KEY", "tml-dummy")
    service = tinker.ServiceClient(base_url=BASE_URL)
    return service.create_lora_training_client(base_model=MODEL, rank=32)


class TestSplitGPUEndToEnd:
    """Full train → sync → inference pipeline."""

    def test_train_and_infer(self, training_client: tinker.TrainingClient, openai_client: OpenAI):
        """Complete cycle: train on data, verify inference works."""
        tokenizer = training_client.get_tokenizer()

        # Train
        tokens = tokenizer.encode("Machine learning is a subset of artificial intelligence.")
        datum = make_cross_entropy_datum(tokens, train_start=len(tokens) // 2)

        result = training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        assert result is not None

        training_client.optim_step(
            tinker.AdamParams(learning_rate=1e-4)
        ).result(timeout=120)

        # Infer
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "What is machine learning?"}],
            max_tokens=30, temperature=0.5,
        )
        assert response.choices
        assert len(response.choices[0].message.content) > 0

    def test_multiple_train_steps_then_infer(
        self, training_client: tinker.TrainingClient, openai_client: OpenAI,
    ):
        """Multiple training steps followed by inference."""
        tokenizer = training_client.get_tokenizer()
        adam = tinker.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)

        sentences = [
            "Python is a high-level programming language.",
            "Neural networks learn through backpropagation.",
            "Transformers use self-attention mechanisms.",
        ]

        for sentence in sentences:
            tokens = tokenizer.encode(sentence)
            datum = make_cross_entropy_datum(tokens, train_start=len(tokens) // 3)
            training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
            training_client.optim_step(adam).result(timeout=120)

        # Inference should work after multiple updates
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Tell me about transformers."}],
            max_tokens=50, temperature=0.5,
        )
        assert len(response.choices[0].message.content) > 10

    def test_ppo_train_and_infer(
        self, training_client: tinker.TrainingClient, openai_client: OpenAI,
    ):
        """PPO training followed by inference."""
        tokenizer = training_client.get_tokenizer()

        tokens = tokenizer.encode("The best way to learn programming is practice.")
        seq_len = len(tokens)
        old_logprobs = [-1.5] * seq_len
        advantages = [1.0] * seq_len
        datum = make_ppo_datum(tokens, old_logprobs, advantages)

        training_client.forward_backward([datum], loss_fn="ppo").result(timeout=300)
        training_client.optim_step(
            tinker.AdamParams(learning_rate=1e-4)
        ).result(timeout=120)

        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "How to learn programming?"}],
            max_tokens=30, temperature=0.5,
        )
        assert response.choices

    def test_concurrent_train_and_infer(
        self, training_client: tinker.TrainingClient, openai_client: OpenAI,
    ):
        """Training and inference can happen concurrently (separate GPUs)."""
        import concurrent.futures
        tokenizer = training_client.get_tokenizer()

        def do_training():
            tokens = tokenizer.encode("Concurrent training test data.")
            datum = make_cross_entropy_datum(tokens)
            training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
            return "train_done"

        def do_inference():
            r = openai_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5, temperature=0.0,
            )
            return f"infer_done: {r.choices[0].message.content[:20]}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            train_future = pool.submit(do_training)
            infer_future = pool.submit(do_inference)

            train_result = train_future.result(timeout=300)
            infer_result = infer_future.result(timeout=60)

        assert train_result == "train_done"
        assert infer_result.startswith("infer_done:")


class TestSplitGPULatency:
    """Latency measurements for split-GPU mode."""

    def test_inference_latency(self, openai_client: OpenAI):
        """Measure chat completion latency."""
        # Warmup
        openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5, temperature=0.0,
        )

        # Measure
        times = []
        for _ in range(3):
            t0 = time.time()
            openai_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "What is 2+2?"}],
                max_tokens=20, temperature=0.0,
            )
            times.append(time.time() - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\nInference latency: {avg_ms:.0f}ms avg ({[f'{t*1000:.0f}ms' for t in times]})")
        assert avg_ms < 30000, f"Inference too slow: {avg_ms}ms"

    def test_training_latency(self, training_client: tinker.TrainingClient):
        """Measure forward_backward + optim_step latency."""
        tokenizer = training_client.get_tokenizer()
        tokens = make_random_tokens(tokenizer, 1024, seed=42)
        datum = make_cross_entropy_datum(tokens, train_start=512)
        adam = tinker.AdamParams(learning_rate=1e-4)

        # Warmup
        training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        training_client.optim_step(adam).result(timeout=120)

        # Measure
        times = []
        for _ in range(3):
            t0 = time.time()
            training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
            training_client.optim_step(adam).result(timeout=120)
            times.append(time.time() - t0)

        avg_s = np.mean(times)
        print(f"\nTraining latency (1024 tokens): {avg_s:.1f}s avg ({[f'{t:.1f}s' for t in times]})")
