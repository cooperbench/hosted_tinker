"""Tests for SamplingClient: text generation and logprob computation."""
from __future__ import annotations

import pytest
import tinker


class TestSampling:
    """Token generation via SamplingClient."""

    @pytest.fixture(scope="class")
    def sampling_client(self, service_client: tinker.ServiceClient, model_name: str) -> tinker.SamplingClient:
        """Create a SamplingClient from the base model (no LoRA)."""
        return service_client.create_sampling_client(base_model=model_name)

    def test_basic_sample(self, sampling_client: tinker.SamplingClient, tokenizer):
        """Generate a completion from a prompt."""
        prompt = tinker.ModelInput(
            chunks=[tinker.EncodedTextChunk(tokens=tokenizer.encode("Once upon a time"))]
        )
        result = sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=20, temperature=0.7),
        ).result(timeout=60)

        assert result is not None
        assert hasattr(result, "sequences")
        assert len(result.sequences) == 1
        assert len(result.sequences[0].tokens) > 0

    def test_multiple_samples(self, sampling_client: tinker.SamplingClient, tokenizer):
        """Generate multiple samples from the same prompt."""
        prompt = tinker.ModelInput(
            chunks=[tinker.EncodedTextChunk(tokens=tokenizer.encode("The answer is"))]
        )
        result = sampling_client.sample(
            prompt=prompt,
            num_samples=3,
            sampling_params=tinker.SamplingParams(max_tokens=10, temperature=1.0),
        ).result(timeout=60)

        assert len(result.sequences) == 3

    def test_sample_with_logprobs(self, sampling_client: tinker.SamplingClient, tokenizer):
        """Samples include per-token log probabilities."""
        prompt = tinker.ModelInput(
            chunks=[tinker.EncodedTextChunk(tokens=tokenizer.encode("Hello"))]
        )
        result = sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=10, temperature=0.5),
        ).result(timeout=60)

        seq = result.sequences[0]
        if seq.logprobs is not None:
            assert len(seq.logprobs) == len(seq.tokens)
            assert all(lp <= 0.0 + 1e-6 for lp in seq.logprobs)

    def test_sample_deterministic_with_seed(self, sampling_client: tinker.SamplingClient, tokenizer):
        """Same seed produces same output."""
        prompt = tinker.ModelInput(
            chunks=[tinker.EncodedTextChunk(tokens=tokenizer.encode("Deterministic test:"))]
        )
        params = tinker.SamplingParams(max_tokens=20, temperature=0.5, seed=42)

        r1 = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=params).result(timeout=60)
        r2 = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=params).result(timeout=60)

        assert r1.sequences[0].tokens == r2.sequences[0].tokens

    def test_stop_reason_length(self, sampling_client: tinker.SamplingClient, tokenizer):
        """When max_tokens is hit, stop_reason is 'length'."""
        prompt = tinker.ModelInput(
            chunks=[tinker.EncodedTextChunk(tokens=tokenizer.encode("Count: 1, 2, 3,"))]
        )
        result = sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=5, temperature=0.5),
        ).result(timeout=60)

        assert result.sequences[0].stop_reason == "length"

    def test_temperature_zero(self, sampling_client: tinker.SamplingClient, tokenizer):
        """Temperature=0 (greedy) produces deterministic output."""
        prompt = tinker.ModelInput(
            chunks=[tinker.EncodedTextChunk(tokens=tokenizer.encode("2 + 2 ="))]
        )
        params = tinker.SamplingParams(max_tokens=5, temperature=0.0)

        r1 = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=params).result(timeout=60)
        r2 = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=params).result(timeout=60)

        assert r1.sequences[0].tokens == r2.sequences[0].tokens


class TestComputeLogprobs:
    """Compute log probabilities without generation."""

    @pytest.fixture(scope="class")
    def sampling_client(self, service_client: tinker.ServiceClient, model_name: str) -> tinker.SamplingClient:
        return service_client.create_sampling_client(base_model=model_name)

    def test_compute_logprobs(self, sampling_client: tinker.SamplingClient, tokenizer):
        """compute_logprobs returns per-token log probabilities."""
        tokens = tokenizer.encode("The quick brown fox")
        prompt = tinker.ModelInput(
            chunks=[tinker.EncodedTextChunk(tokens=tokens)]
        )
        logprobs = sampling_client.compute_logprobs(prompt).result(timeout=60)

        assert logprobs is not None
        assert len(logprobs) == len(tokens)
        # First token has no logprob (no conditioning), rest should be negative
        for lp in logprobs[1:]:
            if lp is not None:
                assert lp <= 0.0 + 1e-6


class TestTrainedModelSampling:
    """Sample from a model after training."""

    def test_sample_after_training(
        self,
        training_client: tinker.TrainingClient,
        tokenizer,
    ):
        """Train on data, save weights, sample from trained model."""
        from .conftest import make_cross_entropy_datum

        # Train
        tokens = tokenizer.encode("The capital of France is Paris.")
        datum = make_cross_entropy_datum(tokens)
        training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        training_client.optim_step(tinker.AdamParams(learning_rate=1e-4)).result(timeout=120)

        # Save and get sampling client
        sc = training_client.save_weights_and_get_sampling_client()

        # Sample
        prompt = tinker.ModelInput(
            chunks=[tinker.EncodedTextChunk(tokens=tokenizer.encode("The capital of"))]
        )
        result = sc.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=10, temperature=0.5),
        ).result(timeout=60)

        assert len(result.sequences) == 1
        assert len(result.sequences[0].tokens) > 0
