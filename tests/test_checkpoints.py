"""Tests for checkpoint save/load: state persistence and weight restoration."""
from __future__ import annotations

import numpy as np
import pytest
import tinker

from .conftest import make_cross_entropy_datum


class TestSaveState:
    """Saving training state (weights + optimizer)."""

    def test_save_state(self, training_client: tinker.TrainingClient, tokenizer):
        """save_state returns a checkpoint path."""
        # Do at least one training step first
        tokens = tokenizer.encode("Save state test.")
        datum = make_cross_entropy_datum(tokens)
        training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        training_client.optim_step(tinker.AdamParams(learning_rate=1e-4)).result(timeout=120)

        result = training_client.save_state("test_checkpoint_001").result(timeout=300)
        assert result is not None
        # Result should contain a path or identifier
        assert hasattr(result, "path") or hasattr(result, "checkpoint_path")

    def test_save_state_multiple(self, training_client: tinker.TrainingClient, tokenizer):
        """Can save multiple checkpoints."""
        tokens = tokenizer.encode("Multi-checkpoint test.")
        datum = make_cross_entropy_datum(tokens)

        for i in range(2):
            training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
            training_client.optim_step(tinker.AdamParams(learning_rate=1e-4)).result(timeout=120)
            result = training_client.save_state(f"multi_ckpt_{i:03d}").result(timeout=300)
            assert result is not None


class TestLoadState:
    """Loading training state from checkpoint."""

    def test_save_and_load_restores_weights(self, training_client: tinker.TrainingClient, tokenizer):
        """Weights are identical after save → train → load."""
        tokens = tokenizer.encode("Round-trip checkpoint test with enough tokens for signal.")
        datum = make_cross_entropy_datum(tokens)

        # Get initial logprobs
        r_init = training_client.forward([datum], loss_fn="cross_entropy").result(timeout=300)
        lp_init = np.array(r_init.loss_fn_outputs[0]["logprobs"].tolist()
                           if hasattr(r_init.loss_fn_outputs[0]["logprobs"], "tolist")
                           else r_init.loss_fn_outputs[0]["logprobs"])

        # Save checkpoint
        save_result = training_client.save_state("restore_test").result(timeout=300)
        ckpt_path = save_result.path if hasattr(save_result, "path") else str(save_result)

        # Train (change weights)
        training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        training_client.optim_step(tinker.AdamParams(learning_rate=1e-2)).result(timeout=120)

        # Verify weights changed
        r_changed = training_client.forward([datum], loss_fn="cross_entropy").result(timeout=300)
        lp_changed = np.array(r_changed.loss_fn_outputs[0]["logprobs"].tolist()
                              if hasattr(r_changed.loss_fn_outputs[0]["logprobs"], "tolist")
                              else r_changed.loss_fn_outputs[0]["logprobs"])
        assert not np.allclose(lp_init, lp_changed, atol=1e-5), "Weights should have changed"

        # Load checkpoint (restore original weights)
        training_client.load_state(ckpt_path).result(timeout=300)

        # Verify weights restored
        r_restored = training_client.forward([datum], loss_fn="cross_entropy").result(timeout=300)
        lp_restored = np.array(r_restored.loss_fn_outputs[0]["logprobs"].tolist()
                               if hasattr(r_restored.loss_fn_outputs[0]["logprobs"], "tolist")
                               else r_restored.loss_fn_outputs[0]["logprobs"])

        np.testing.assert_allclose(lp_init, lp_restored, atol=1e-4,
                                   err_msg="Logprobs should match after checkpoint restore")


class TestSaveWeightsForSampler:
    """Saving weights for inference via SamplingClient."""

    def test_save_weights_for_sampler(self, training_client: tinker.TrainingClient, tokenizer):
        """save_weights_for_sampler returns a valid result."""
        tokens = tokenizer.encode("Sampler weights test.")
        datum = make_cross_entropy_datum(tokens)

        training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        training_client.optim_step(tinker.AdamParams(learning_rate=1e-4)).result(timeout=120)

        result = training_client.save_weights_for_sampler("sampler_ckpt_001").result(timeout=300)
        assert result is not None

    def test_save_weights_and_get_sampling_client(self, training_client: tinker.TrainingClient, tokenizer):
        """save_weights_and_get_sampling_client returns a SamplingClient."""
        tokens = tokenizer.encode("Get sampling client test.")
        datum = make_cross_entropy_datum(tokens)

        training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        training_client.optim_step(tinker.AdamParams(learning_rate=1e-4)).result(timeout=120)

        sampling_client = training_client.save_weights_and_get_sampling_client()
        assert sampling_client is not None
