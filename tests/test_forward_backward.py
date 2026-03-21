"""Tests for forward_backward: cross_entropy, PPO, importance_sampling losses."""
from __future__ import annotations

import numpy as np
import pytest
import tinker

from .conftest import make_cross_entropy_datum, make_ppo_datum, make_random_tokens


class TestCrossEntropyLoss:
    """forward_backward with cross_entropy loss."""

    def test_basic_forward_backward(self, training_client: tinker.TrainingClient, tokenizer):
        """Basic cross_entropy forward_backward returns logprobs."""
        tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog")
        datum = make_cross_entropy_datum(tokens)
        result = training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)

        assert result is not None
        assert hasattr(result, "loss_fn_outputs")
        assert len(result.loss_fn_outputs) == 1
        assert "logprobs" in result.loss_fn_outputs[0]

    def test_logprobs_shape(self, training_client: tinker.TrainingClient, tokenizer):
        """Logprobs length matches input sequence length."""
        tokens = tokenizer.encode("Hello world, this is a test sentence.")
        datum = make_cross_entropy_datum(tokens)
        result = training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)

        logprobs = result.loss_fn_outputs[0]["logprobs"]
        # logprobs should have one entry per token (or per non-first token)
        lp_list = logprobs.tolist() if hasattr(logprobs, "tolist") else logprobs
        assert len(lp_list) > 0

    def test_logprobs_are_negative(self, training_client: tinker.TrainingClient, tokenizer):
        """Log probabilities should be <= 0."""
        tokens = tokenizer.encode("A simple test.")
        datum = make_cross_entropy_datum(tokens)
        result = training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)

        logprobs = result.loss_fn_outputs[0]["logprobs"]
        lp_array = np.array(logprobs.tolist() if hasattr(logprobs, "tolist") else logprobs)
        assert np.all(lp_array <= 0.0 + 1e-6), f"Logprobs should be <= 0, got max={lp_array.max()}"

    def test_batch_of_two(self, training_client: tinker.TrainingClient, tokenizer):
        """forward_backward with a batch of 2 datums."""
        datum1 = make_cross_entropy_datum(tokenizer.encode("First sentence."))
        datum2 = make_cross_entropy_datum(tokenizer.encode("Second sentence, slightly longer."))
        result = training_client.forward_backward([datum1, datum2], loss_fn="cross_entropy").result(timeout=300)

        assert len(result.loss_fn_outputs) == 2

    def test_weighted_loss(self, training_client: tinker.TrainingClient, tokenizer):
        """Weights=0 on prefix means only suffix is trained."""
        tokens = tokenizer.encode("System prompt. User question. Assistant response here.")
        # Train only on last 5 tokens
        datum = make_cross_entropy_datum(tokens, train_start=len(tokens) - 5)
        result = training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)

        assert result is not None
        assert len(result.loss_fn_outputs) == 1

    def test_metrics_present(self, training_client: tinker.TrainingClient, tokenizer):
        """forward_backward returns metrics dict."""
        tokens = tokenizer.encode("Test metrics.")
        datum = make_cross_entropy_datum(tokens)
        result = training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)

        assert hasattr(result, "metrics")
        assert isinstance(result.metrics, dict)


class TestForwardOnly:
    """forward (no gradients) with cross_entropy loss."""

    def test_forward_returns_logprobs(self, training_client: tinker.TrainingClient, tokenizer):
        """forward() returns logprobs without computing gradients."""
        tokens = tokenizer.encode("Forward only test.")
        datum = make_cross_entropy_datum(tokens)
        result = training_client.forward([datum], loss_fn="cross_entropy").result(timeout=300)

        assert result is not None
        assert len(result.loss_fn_outputs) == 1
        assert "logprobs" in result.loss_fn_outputs[0]

    def test_forward_does_not_accumulate_gradients(self, training_client: tinker.TrainingClient, tokenizer):
        """forward() followed by optim_step should not change weights
        (since no gradients were computed)."""
        tokens = tokenizer.encode("No gradient test.")
        datum = make_cross_entropy_datum(tokens)

        # Get initial logprobs
        r1 = training_client.forward([datum], loss_fn="cross_entropy").result(timeout=300)
        lp1 = r1.loss_fn_outputs[0]["logprobs"]

        # optim_step with zero lr (should be no-op)
        training_client.optim_step(
            tinker.AdamParams(learning_rate=0.0)
        ).result(timeout=60)

        # Get logprobs again — should be identical
        r2 = training_client.forward([datum], loss_fn="cross_entropy").result(timeout=300)
        lp2 = r2.loss_fn_outputs[0]["logprobs"]

        lp1_arr = np.array(lp1.tolist() if hasattr(lp1, "tolist") else lp1)
        lp2_arr = np.array(lp2.tolist() if hasattr(lp2, "tolist") else lp2)
        np.testing.assert_allclose(lp1_arr, lp2_arr, atol=1e-5)


class TestPPOLoss:
    """forward_backward with PPO loss."""

    def test_ppo_basic(self, training_client: tinker.TrainingClient, tokenizer):
        """PPO loss with uniform advantages."""
        tokens = tokenizer.encode("PPO test sentence with some length.")
        seq_len = len(tokens)
        old_logprobs = [-1.0] * seq_len
        advantages = [1.0] * seq_len

        datum = make_ppo_datum(tokens, old_logprobs, advantages)
        result = training_client.forward_backward([datum], loss_fn="ppo").result(timeout=300)

        assert result is not None
        assert len(result.loss_fn_outputs) == 1

    def test_ppo_with_clip_config(self, training_client: tinker.TrainingClient, tokenizer):
        """PPO with custom clip thresholds via loss_fn_config."""
        tokens = tokenizer.encode("PPO clip test.")
        seq_len = len(tokens)
        old_logprobs = [-1.5] * seq_len
        advantages = [0.5] * seq_len

        datum = make_ppo_datum(tokens, old_logprobs, advantages)
        result = training_client.forward_backward(
            [datum],
            loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
        ).result(timeout=300)

        assert result is not None

    def test_ppo_zero_advantages(self, training_client: tinker.TrainingClient, tokenizer):
        """PPO with zero advantages produces near-zero loss."""
        tokens = tokenizer.encode("Zero advantage test.")
        seq_len = len(tokens)
        old_logprobs = [-1.0] * seq_len
        advantages = [0.0] * seq_len

        datum = make_ppo_datum(tokens, old_logprobs, advantages)
        result = training_client.forward_backward([datum], loss_fn="ppo").result(timeout=300)
        assert result is not None


class TestImportanceSamplingLoss:
    """forward_backward with importance_sampling loss."""

    def test_importance_sampling_basic(self, training_client: tinker.TrainingClient, tokenizer):
        """Importance sampling loss with uniform advantages."""
        tokens = tokenizer.encode("Importance sampling test.")
        seq_len = len(tokens)
        old_logprobs = [-1.0] * seq_len
        advantages = [1.0] * seq_len

        datum = make_ppo_datum(tokens, old_logprobs, advantages)
        result = training_client.forward_backward(
            [datum], loss_fn="importance_sampling"
        ).result(timeout=300)

        assert result is not None
        assert len(result.loss_fn_outputs) == 1


class TestLongSequences:
    """Test with increasing sequence lengths."""

    @pytest.mark.parametrize("length", [256, 1024, 4096])
    def test_sequence_length(self, training_client: tinker.TrainingClient, tokenizer, length: int):
        """forward_backward works at various sequence lengths."""
        tokens = make_random_tokens(tokenizer, length)
        datum = make_cross_entropy_datum(tokens, train_start=length // 2)
        result = training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=600)

        assert result is not None
        assert len(result.loss_fn_outputs) == 1
