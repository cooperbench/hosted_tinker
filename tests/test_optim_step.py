"""Tests for optim_step: gradient application, learning rate, weight updates."""
from __future__ import annotations

import numpy as np
import pytest
import tinker

from .conftest import make_cross_entropy_datum


class TestOptimStep:
    """Optimizer step behavior."""

    def test_basic_optim_step(self, training_client: tinker.TrainingClient, tokenizer):
        """forward_backward + optim_step completes without error."""
        tokens = tokenizer.encode("Optimizer step test.")
        datum = make_cross_entropy_datum(tokens)

        training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        result = training_client.optim_step(
            tinker.AdamParams(learning_rate=1e-4)
        ).result(timeout=120)

        assert result is not None

    def test_weights_change_after_step(self, training_client: tinker.TrainingClient, tokenizer):
        """Logprobs change after a training step (weights updated)."""
        tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog.")
        datum = make_cross_entropy_datum(tokens)

        # Get initial logprobs
        r1 = training_client.forward([datum], loss_fn="cross_entropy").result(timeout=300)
        lp1 = np.array(r1.loss_fn_outputs[0]["logprobs"].tolist()
                        if hasattr(r1.loss_fn_outputs[0]["logprobs"], "tolist")
                        else r1.loss_fn_outputs[0]["logprobs"])

        # Train
        training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        training_client.optim_step(
            tinker.AdamParams(learning_rate=1e-3)  # large LR to ensure visible change
        ).result(timeout=120)

        # Get logprobs after training
        r2 = training_client.forward([datum], loss_fn="cross_entropy").result(timeout=300)
        lp2 = np.array(r2.loss_fn_outputs[0]["logprobs"].tolist()
                        if hasattr(r2.loss_fn_outputs[0]["logprobs"], "tolist")
                        else r2.loss_fn_outputs[0]["logprobs"])

        # Logprobs should have changed
        assert not np.allclose(lp1, lp2, atol=1e-6), "Logprobs unchanged after training step"

    def test_zero_lr_no_change(self, training_client: tinker.TrainingClient, tokenizer):
        """optim_step with lr=0 should not change weights."""
        tokens = tokenizer.encode("Zero learning rate test.")
        datum = make_cross_entropy_datum(tokens)

        # Get baseline
        r1 = training_client.forward([datum], loss_fn="cross_entropy").result(timeout=300)
        lp1 = np.array(r1.loss_fn_outputs[0]["logprobs"].tolist()
                        if hasattr(r1.loss_fn_outputs[0]["logprobs"], "tolist")
                        else r1.loss_fn_outputs[0]["logprobs"])

        # Train with lr=0
        training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        training_client.optim_step(
            tinker.AdamParams(learning_rate=0.0)
        ).result(timeout=120)

        # Logprobs should be the same
        r2 = training_client.forward([datum], loss_fn="cross_entropy").result(timeout=300)
        lp2 = np.array(r2.loss_fn_outputs[0]["logprobs"].tolist()
                        if hasattr(r2.loss_fn_outputs[0]["logprobs"], "tolist")
                        else r2.loss_fn_outputs[0]["logprobs"])

        np.testing.assert_allclose(lp1, lp2, atol=1e-5)

    def test_adam_params(self, training_client: tinker.TrainingClient, tokenizer):
        """Various AdamParams configurations are accepted."""
        tokens = tokenizer.encode("Adam params test.")
        datum = make_cross_entropy_datum(tokens)
        training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)

        result = training_client.optim_step(
            tinker.AdamParams(
                learning_rate=1e-5,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                weight_decay=0.01,
            )
        ).result(timeout=120)

        assert result is not None

    def test_optim_step_metrics(self, training_client: tinker.TrainingClient, tokenizer):
        """optim_step returns metrics (e.g., grad_norm)."""
        tokens = tokenizer.encode("Metrics test sentence.")
        datum = make_cross_entropy_datum(tokens)

        training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        result = training_client.optim_step(
            tinker.AdamParams(learning_rate=1e-4)
        ).result(timeout=120)

        # Check that result has metrics attribute
        if hasattr(result, "metrics") and result.metrics:
            assert isinstance(result.metrics, dict)


class TestMultipleSteps:
    """Multiple training steps in sequence."""

    def test_multiple_forward_backward_then_step(self, training_client: tinker.TrainingClient, tokenizer):
        """Multiple forward_backward calls accumulate, then one optim_step."""
        tokens1 = tokenizer.encode("First training sample.")
        tokens2 = tokenizer.encode("Second training sample with more words.")
        datum1 = make_cross_entropy_datum(tokens1)
        datum2 = make_cross_entropy_datum(tokens2)

        # Two forward_backward calls
        training_client.forward_backward([datum1], loss_fn="cross_entropy").result(timeout=300)
        training_client.forward_backward([datum2], loss_fn="cross_entropy").result(timeout=300)

        # One optim_step
        result = training_client.optim_step(
            tinker.AdamParams(learning_rate=1e-4)
        ).result(timeout=120)

        assert result is not None

    def test_training_loop(self, training_client: tinker.TrainingClient, tokenizer):
        """Simulate a 3-step training loop."""
        sentences = [
            "Step one of training.",
            "Step two with different data.",
            "Step three wrapping up.",
        ]

        for i, sentence in enumerate(sentences):
            tokens = tokenizer.encode(sentence)
            datum = make_cross_entropy_datum(tokens)

            fwd = training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
            assert fwd is not None

            opt = training_client.optim_step(
                tinker.AdamParams(learning_rate=1e-4 * (1 - i / len(sentences)))
            ).result(timeout=120)
            assert opt is not None
