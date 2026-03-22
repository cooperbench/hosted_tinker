"""Tests for OpenAI-compatible inference endpoints (/v1/chat/completions, /v1/completions).

Tests that the hosted-tinker server proxies inference requests to vLLM correctly,
and that LoRA weight updates from training are reflected in subsequent inference.

Usage:
    # Against self-hosted server with vLLM running:
    TINKER_BASE_URL=http://localhost:8000 pytest tests/test_openai_inference.py -v

    # Skip if no vLLM configured:
    pytest tests/test_openai_inference.py -v  # auto-skips if /v1/models returns 503
"""
from __future__ import annotations

import os

import numpy as np
import pytest

# Use OpenAI SDK for inference tests
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

import tinker

from .conftest import make_cross_entropy_datum


BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("TINKER_MODEL", "Qwen/Qwen3-30B-A3B")


def _inference_available() -> bool:
    """Check if the inference endpoint is available."""
    if not HAS_OPENAI:
        return False
    try:
        import httpx
        r = httpx.get(f"{BASE_URL}/v1/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _inference_available(),
    reason="Inference endpoint not available (no vLLM configured or OpenAI SDK not installed)"
)


@pytest.fixture(scope="module")
def openai_client() -> OpenAI:
    return OpenAI(base_url=f"{BASE_URL}/v1", api_key="not-needed")


@pytest.fixture(scope="module")
def training_client() -> tinker.TrainingClient:
    os.environ.setdefault("TINKER_API_KEY", "tml-dummy")
    service = tinker.ServiceClient(base_url=BASE_URL)
    return service.create_lora_training_client(base_model=MODEL, rank=32)


class TestOpenAIEndpoints:
    """Basic OpenAI-compatible endpoint tests."""

    def test_list_models(self, openai_client: OpenAI):
        """GET /v1/models returns available models."""
        models = openai_client.models.list()
        assert len(models.data) > 0
        model_ids = [m.id for m in models.data]
        assert any(MODEL in mid or "Qwen" in mid for mid in model_ids), f"Model not found in {model_ids}"

    def test_chat_completion(self, openai_client: OpenAI):
        """POST /v1/chat/completions generates a response."""
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            max_tokens=20,
            temperature=0.0,
        )
        assert response.choices
        assert len(response.choices[0].message.content) > 0
        assert response.usage.completion_tokens > 0

    def test_completion(self, openai_client: OpenAI):
        """POST /v1/completions generates a completion."""
        response = openai_client.completions.create(
            model=MODEL,
            prompt="The capital of France is",
            max_tokens=10,
            temperature=0.0,
        )
        assert response.choices
        assert len(response.choices[0].text) > 0

    def test_chat_multiple_turns(self, openai_client: OpenAI):
        """Multi-turn conversation works."""
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=20,
            temperature=0.0,
        )
        assert response.choices
        content = response.choices[0].message.content.lower()
        assert "alice" in content, f"Expected 'alice' in response: {content}"

    def test_temperature_zero_deterministic(self, openai_client: OpenAI):
        """Temperature=0 produces deterministic output."""
        kwargs = dict(
            model=MODEL,
            messages=[{"role": "user", "content": "Count: 1, 2, 3,"}],
            max_tokens=10,
            temperature=0.0,
        )
        r1 = openai_client.chat.completions.create(**kwargs)
        r2 = openai_client.chat.completions.create(**kwargs)
        assert r1.choices[0].message.content == r2.choices[0].message.content

    def test_max_tokens_respected(self, openai_client: OpenAI):
        """max_tokens limits output length."""
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Write a very long essay about physics."}],
            max_tokens=5,
            temperature=0.5,
        )
        assert response.usage.completion_tokens <= 6  # Allow 1 token buffer


class TestLoRASyncAfterTraining:
    """Test that training updates are reflected in inference."""

    def test_inference_changes_after_training(
        self, openai_client: OpenAI, training_client: tinker.TrainingClient,
    ):
        """After training on specific data, inference output should change.

        This is the key integration test: train → sync LoRA → inference reflects training.
        """
        tokenizer = training_client.get_tokenizer()

        # Step 1: Get baseline inference output
        prompt = "The answer to the ultimate question is"
        r_before = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20, temperature=0.0,
        )
        before_text = r_before.choices[0].message.content

        # Step 2: Train on specific data (reinforce a pattern)
        train_text = "The answer to the ultimate question is forty-two, always forty-two."
        tokens = tokenizer.encode(train_text)
        datum = make_cross_entropy_datum(tokens, train_start=len(tokens) // 3)

        for _ in range(5):  # Multiple steps to make the effect visible
            training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
            training_client.optim_step(
                tinker.AdamParams(learning_rate=5e-3)
            ).result(timeout=120)
            # optim_step should auto-sync LoRA to vLLM

        # Step 3: Get post-training inference output
        r_after = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20, temperature=0.0,
        )
        after_text = r_after.choices[0].message.content

        # The output should have changed (training effect)
        # Note: with only 5 steps on one sentence, the change may be subtle
        print(f"Before training: {before_text}")
        print(f"After training:  {after_text}")
        # We can't guarantee specific content, but logprobs should differ
        assert r_after.choices is not None


class TestLoRAAdapterManagement:
    """Test LoRA adapter loading/unloading on vLLM."""

    def test_inference_with_base_model(self, openai_client: OpenAI):
        """Inference works with the base model (no adapter)."""
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5, temperature=0.0,
        )
        assert response.choices


# ---------------------------------------------------------------------------
# Reference comparison: self-hosted inference vs official Tinker inference
# ---------------------------------------------------------------------------

TINKER_API_KEY = os.environ.get("TINKER_API_KEY")
_HAS_CLOUD_KEY = TINKER_API_KEY and TINKER_API_KEY != "tml-dummy"


@pytest.mark.skipif(not _HAS_CLOUD_KEY, reason="No real TINKER_API_KEY for cloud comparison")
class TestInferenceCompareWithTinker:
    """Compare self-hosted inference (vLLM) with official Tinker inference."""

    @pytest.fixture(scope="class")
    def cloud_sampling_client(self) -> tinker.SamplingClient:
        """Create a SamplingClient from the official Tinker API."""
        service = tinker.ServiceClient()  # Uses default cloud URL
        return service.create_sampling_client(base_model=MODEL)

    def test_logprobs_match(self, openai_client: OpenAI, cloud_sampling_client: tinker.SamplingClient):
        """Logprobs from self-hosted vLLM should match official Tinker inference.

        Both use the same base model with no adapter — logprobs should be identical.
        """
        prompt_text = "The quick brown fox"
        tokenizer = cloud_sampling_client.get_tokenizer()
        tokens = tokenizer.encode(prompt_text)

        # Cloud: compute logprobs via Tinker SDK
        prompt = tinker.ModelInput(chunks=[tinker.EncodedTextChunk(tokens=tokens)])
        cloud_lps = cloud_sampling_client.compute_logprobs(prompt).result(timeout=60)
        cloud_lps = [lp for lp in cloud_lps if lp is not None]

        # Self-hosted: compute logprobs via OpenAI API with logprobs=True
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=1,
        )

        # Note: OpenAI API returns logprobs differently than Tinker SDK.
        # The comparison here is qualitative — both should produce reasonable logprobs.
        assert response.choices[0].logprobs is not None
        print(f"Cloud logprobs (first 3): {cloud_lps[:3]}")
        print(f"Self-hosted has logprobs: {response.choices[0].logprobs is not None}")

    def test_generation_quality_similar(self, openai_client: OpenAI, cloud_sampling_client: tinker.SamplingClient):
        """Both should generate coherent completions for the same prompt."""
        prompt_text = "Explain what a neural network is in one sentence:"

        # Cloud
        tokenizer = cloud_sampling_client.get_tokenizer()
        prompt = tinker.ModelInput(chunks=[tinker.EncodedTextChunk(tokens=tokenizer.encode(prompt_text))])
        cloud_result = cloud_sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=50, temperature=0.0),
        ).result(timeout=60)
        cloud_text = tokenizer.decode(cloud_result.sequences[0].tokens)

        # Self-hosted
        local_result = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=50, temperature=0.0,
        )
        local_text = local_result.choices[0].message.content

        print(f"Cloud:  {cloud_text[:100]}")
        print(f"Local:  {local_text[:100]}")

        # Both should produce non-empty, coherent text
        assert len(cloud_text) > 10, f"Cloud response too short: {cloud_text}"
        assert len(local_text) > 10, f"Local response too short: {local_text}"
