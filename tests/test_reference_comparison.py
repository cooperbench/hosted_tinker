"""Compare self-hosted backend against official Tinker API (cloud).

Verifies that forward_backward, optim_step, and logprob outputs are
numerically identical (within tolerance) between the two backends.

This is the key validation test: if the self-hosted backend produces
the same gradients as the official Tinker API, the training is correct.

Usage:
    TINKER_API_KEY=tml-xxx TINKER_CLOUD_URL=https://api.tinker.live \
    TINKER_BASE_URL=http://localhost:8000 \
    pytest tests/test_reference_comparison.py -v

Environment variables:
    TINKER_CLOUD_URL - Official Tinker API URL (required)
    TINKER_BASE_URL  - Self-hosted server URL (default: http://localhost:8000)
    TINKER_API_KEY   - API key for cloud Tinker (required)
    TINKER_MODEL     - Model name (must be available on both)
    TINKER_LORA_RANK - LoRA rank (default: 32)
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import tinker

from .conftest import make_cross_entropy_datum, make_ppo_datum, make_random_tokens

CLOUD_URL = os.environ.get("TINKER_CLOUD_URL", "default")
_CLOUD_API_KEY = os.environ.get("TINKER_API_KEY")
if not _CLOUD_API_KEY or _CLOUD_API_KEY == "tml-dummy":
    pytestmark = pytest.mark.skip(reason="No real TINKER_API_KEY set — skipping reference comparison tests")


# ---------------------------------------------------------------------------
# Fixtures: two parallel clients (cloud + self-hosted)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cloud_service() -> tinker.ServiceClient:
    if CLOUD_URL and CLOUD_URL != "default":
        return tinker.ServiceClient(base_url=CLOUD_URL)
    return tinker.ServiceClient()  # uses official API with default URL


@pytest.fixture(scope="module")
def local_service(base_url) -> tinker.ServiceClient:
    return tinker.ServiceClient(base_url=base_url)


@pytest.fixture(scope="module")
def paired_clients(cloud_service, local_service, model_name, lora_rank):
    """Create paired training clients with identical config on both backends.

    Both start from the same fresh LoRA initialization (seed=0).
    """
    cloud_tc = cloud_service.create_lora_training_client(
        base_model=model_name, rank=lora_rank, seed=0,
    )
    local_tc = local_service.create_lora_training_client(
        base_model=model_name, rank=lora_rank, seed=0,
    )
    return cloud_tc, local_tc


def _extract_logprobs(result, idx: int = 0) -> np.ndarray:
    """Extract logprobs array from a ForwardBackwardOutput."""
    lp = result.loss_fn_outputs[idx]["logprobs"]
    return np.array(lp.tolist() if hasattr(lp, "tolist") else lp, dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests: compare logprobs from forward pass
# ---------------------------------------------------------------------------

class TestForwardLogprobComparison:
    """Compare forward pass logprobs between cloud and self-hosted."""

    def test_initial_logprobs_match(self, paired_clients, tokenizer):
        """Fresh model (no training): logprobs should be identical."""
        cloud_tc, local_tc = paired_clients

        tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog.")
        datum = make_cross_entropy_datum(tokens)

        cloud_result = cloud_tc.forward([datum], loss_fn="cross_entropy").result(timeout=300)
        local_result = local_tc.forward([datum], loss_fn="cross_entropy").result(timeout=300)

        cloud_lp = _extract_logprobs(cloud_result)
        local_lp = _extract_logprobs(local_result)

        assert cloud_lp.shape == local_lp.shape, (
            f"Shape mismatch: cloud={cloud_lp.shape}, local={local_lp.shape}"
        )

        # bf16 precision: ~1e-3 relative tolerance
        np.testing.assert_allclose(
            cloud_lp, local_lp, rtol=1e-2, atol=1e-3,
            err_msg="Initial logprobs differ between cloud and self-hosted",
        )

    def test_logprobs_batch_match(self, paired_clients, tokenizer):
        """Batch of 3 datums: logprobs should match per-datum."""
        cloud_tc, local_tc = paired_clients

        datums = [
            make_cross_entropy_datum(tokenizer.encode("First sentence.")),
            make_cross_entropy_datum(tokenizer.encode("Second sentence here.")),
            make_cross_entropy_datum(tokenizer.encode("Third sentence is longer than the others.")),
        ]

        cloud_result = cloud_tc.forward(datums, loss_fn="cross_entropy").result(timeout=300)
        local_result = local_tc.forward(datums, loss_fn="cross_entropy").result(timeout=300)

        assert len(cloud_result.loss_fn_outputs) == len(local_result.loss_fn_outputs) == 3

        for i in range(3):
            cloud_lp = _extract_logprobs(cloud_result, i)
            local_lp = _extract_logprobs(local_result, i)
            np.testing.assert_allclose(
                cloud_lp, local_lp, rtol=1e-2, atol=1e-3,
                err_msg=f"Logprobs differ for datum {i}",
            )


class TestForwardBackwardComparison:
    """Compare forward_backward (logprobs + gradient effect) between backends."""

    def test_cross_entropy_logprobs_match(self, paired_clients, tokenizer):
        """cross_entropy forward_backward: logprobs should match."""
        cloud_tc, local_tc = paired_clients

        tokens = tokenizer.encode("Cross entropy comparison test with enough tokens.")
        datum = make_cross_entropy_datum(tokens, train_start=len(tokens) // 2)

        cloud_result = cloud_tc.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        local_result = local_tc.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)

        cloud_lp = _extract_logprobs(cloud_result)
        local_lp = _extract_logprobs(local_result)

        np.testing.assert_allclose(
            cloud_lp, local_lp, rtol=1e-2, atol=1e-3,
            err_msg="cross_entropy forward_backward logprobs differ",
        )

    def test_ppo_logprobs_match(self, paired_clients, tokenizer):
        """PPO forward_backward: logprobs should match."""
        cloud_tc, local_tc = paired_clients

        tokens = tokenizer.encode("PPO comparison test sentence here.")
        seq_len = len(tokens)
        old_logprobs = [-1.5] * seq_len
        advantages = [0.5] * seq_len

        datum = make_ppo_datum(tokens, old_logprobs, advantages)

        cloud_result = cloud_tc.forward_backward([datum], loss_fn="ppo").result(timeout=300)
        local_result = local_tc.forward_backward([datum], loss_fn="ppo").result(timeout=300)

        cloud_lp = _extract_logprobs(cloud_result)
        local_lp = _extract_logprobs(local_result)

        np.testing.assert_allclose(
            cloud_lp, local_lp, rtol=1e-2, atol=1e-3,
            err_msg="PPO forward_backward logprobs differ",
        )


class TestOptimStepComparison:
    """Compare weight updates: after identical training, logprobs should match."""

    def test_one_step_logprobs_match(self, cloud_service, local_service, model_name, lora_rank, tokenizer):
        """After 1 training step with identical data and lr, logprobs should match."""
        # Fresh clients with same seed
        cloud_tc = cloud_service.create_lora_training_client(
            base_model=model_name, rank=lora_rank, seed=0,
        )
        local_tc = local_service.create_lora_training_client(
            base_model=model_name, rank=lora_rank, seed=0,
        )

        # Identical training data
        tokens = tokenizer.encode("Training step comparison: the model should learn this sentence.")
        datum = make_cross_entropy_datum(tokens, train_start=len(tokens) // 2)

        # Train on both
        cloud_tc.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        local_tc.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)

        adam = tinker.AdamParams(learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)
        cloud_tc.optim_step(adam).result(timeout=120)
        local_tc.optim_step(adam).result(timeout=120)

        # Compare logprobs AFTER training
        eval_tokens = tokenizer.encode("Evaluate after one step of training.")
        eval_datum = make_cross_entropy_datum(eval_tokens)

        cloud_result = cloud_tc.forward([eval_datum], loss_fn="cross_entropy").result(timeout=300)
        local_result = local_tc.forward([eval_datum], loss_fn="cross_entropy").result(timeout=300)

        cloud_lp = _extract_logprobs(cloud_result)
        local_lp = _extract_logprobs(local_result)

        np.testing.assert_allclose(
            cloud_lp, local_lp, rtol=5e-2, atol=5e-3,
            err_msg="Post-training logprobs differ after 1 step",
        )

    def test_three_steps_logprobs_match(self, cloud_service, local_service, model_name, lora_rank, tokenizer):
        """After 3 training steps, logprobs should still match (drift check)."""
        cloud_tc = cloud_service.create_lora_training_client(
            base_model=model_name, rank=lora_rank, seed=0,
        )
        local_tc = local_service.create_lora_training_client(
            base_model=model_name, rank=lora_rank, seed=0,
        )

        adam = tinker.AdamParams(learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)

        sentences = [
            "First step training data for comparison.",
            "Second step with different tokens.",
            "Third and final step of this comparison.",
        ]

        for sentence in sentences:
            tokens = tokenizer.encode(sentence)
            datum = make_cross_entropy_datum(tokens, train_start=len(tokens) // 3)

            cloud_tc.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
            local_tc.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)

            cloud_tc.optim_step(adam).result(timeout=120)
            local_tc.optim_step(adam).result(timeout=120)

        # Compare final logprobs
        eval_tokens = tokenizer.encode("Final evaluation after three training steps.")
        eval_datum = make_cross_entropy_datum(eval_tokens)

        cloud_result = cloud_tc.forward([eval_datum], loss_fn="cross_entropy").result(timeout=300)
        local_result = local_tc.forward([eval_datum], loss_fn="cross_entropy").result(timeout=300)

        cloud_lp = _extract_logprobs(cloud_result)
        local_lp = _extract_logprobs(local_result)

        # Allow slightly more tolerance for accumulated drift over 3 steps
        np.testing.assert_allclose(
            cloud_lp, local_lp, rtol=0.1, atol=1e-2,
            err_msg="Post-training logprobs diverged after 3 steps",
        )

    def test_ppo_training_match(self, cloud_service, local_service, model_name, lora_rank, tokenizer):
        """PPO training step produces same weight update on both backends."""
        cloud_tc = cloud_service.create_lora_training_client(
            base_model=model_name, rank=lora_rank, seed=0,
        )
        local_tc = local_service.create_lora_training_client(
            base_model=model_name, rank=lora_rank, seed=0,
        )

        tokens = tokenizer.encode("PPO training comparison data for gradient validation.")
        seq_len = len(tokens)
        old_logprobs = [-1.5] * seq_len
        advantages = [1.0] * (seq_len // 2) + [-1.0] * (seq_len - seq_len // 2)
        datum = make_ppo_datum(tokens, old_logprobs, advantages, train_start=seq_len // 3)

        # Train
        cloud_tc.forward_backward([datum], loss_fn="ppo").result(timeout=300)
        local_tc.forward_backward([datum], loss_fn="ppo").result(timeout=300)

        adam = tinker.AdamParams(learning_rate=1e-3)
        cloud_tc.optim_step(adam).result(timeout=120)
        local_tc.optim_step(adam).result(timeout=120)

        # Compare
        eval_datum = make_cross_entropy_datum(tokenizer.encode("Eval after PPO."))
        cloud_result = cloud_tc.forward([eval_datum], loss_fn="cross_entropy").result(timeout=300)
        local_result = local_tc.forward([eval_datum], loss_fn="cross_entropy").result(timeout=300)

        cloud_lp = _extract_logprobs(cloud_result)
        local_lp = _extract_logprobs(local_result)

        np.testing.assert_allclose(
            cloud_lp, local_lp, rtol=5e-2, atol=5e-3,
            err_msg="Post-PPO logprobs differ between backends",
        )


class TestLongSequenceComparison:
    """Compare on longer sequences (stress test for numerical precision)."""

    @pytest.mark.parametrize("length", [512, 2048, 8192, 32768])
    def test_long_sequence_logprobs_match(self, paired_clients, tokenizer, length):
        """Logprobs match on longer sequences including 32K."""
        cloud_tc, local_tc = paired_clients

        tokens = make_random_tokens(tokenizer, length, seed=42)
        datum = make_cross_entropy_datum(tokens, train_start=length // 2)

        cloud_result = cloud_tc.forward([datum], loss_fn="cross_entropy").result(timeout=1200)
        local_result = local_tc.forward([datum], loss_fn="cross_entropy").result(timeout=1200)

        cloud_lp = _extract_logprobs(cloud_result)
        local_lp = _extract_logprobs(local_result)

        # Report statistics
        diff = np.abs(cloud_lp - local_lp)
        print(f"\n  length={length}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}, "
              f"cloud_mean_lp={cloud_lp.mean():.4f}, local_mean_lp={local_lp.mean():.4f}")

        np.testing.assert_allclose(
            cloud_lp, local_lp, rtol=1e-2, atol=1e-3,
            err_msg=f"Logprobs differ at length={length}",
        )


class TestTrainingAt32K:
    """Train on 32K sequences and compare gradient descent results."""

    def test_32k_forward_backward_logprobs_match(self, cloud_service, local_service, model_name, lora_rank, tokenizer):
        """forward_backward at 32K tokens: logprobs match between backends."""
        cloud_tc = cloud_service.create_lora_training_client(
            base_model=model_name, rank=lora_rank, seed=0,
        )
        local_tc = local_service.create_lora_training_client(
            base_model=model_name, rank=lora_rank, seed=0,
        )

        tokens = make_random_tokens(tokenizer, 32768, seed=99)
        datum = make_cross_entropy_datum(tokens, train_start=len(tokens) // 2)

        cloud_result = cloud_tc.forward_backward([datum], loss_fn="cross_entropy").result(timeout=1200)
        local_result = local_tc.forward_backward([datum], loss_fn="cross_entropy").result(timeout=1200)

        cloud_lp = _extract_logprobs(cloud_result)
        local_lp = _extract_logprobs(local_result)

        diff = np.abs(cloud_lp - local_lp)
        print(f"\n  32K forward_backward: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")

        np.testing.assert_allclose(
            cloud_lp, local_lp, rtol=1e-2, atol=1e-3,
            err_msg="32K forward_backward logprobs differ",
        )

    def test_32k_training_step_match(self, cloud_service, local_service, model_name, lora_rank, tokenizer):
        """After training on 32K tokens, weights match (measured via logprobs on eval data)."""
        cloud_tc = cloud_service.create_lora_training_client(
            base_model=model_name, rank=lora_rank, seed=0,
        )
        local_tc = local_service.create_lora_training_client(
            base_model=model_name, rank=lora_rank, seed=0,
        )

        # Train on 32K sequence
        tokens = make_random_tokens(tokenizer, 32768, seed=77)
        datum = make_cross_entropy_datum(tokens, train_start=len(tokens) // 2)

        cloud_tc.forward_backward([datum], loss_fn="cross_entropy").result(timeout=1200)
        local_tc.forward_backward([datum], loss_fn="cross_entropy").result(timeout=1200)

        adam = tinker.AdamParams(learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)
        cloud_tc.optim_step(adam).result(timeout=120)
        local_tc.optim_step(adam).result(timeout=120)

        # Evaluate on short sequence (fast) — if weights match, logprobs match
        eval_tokens = tokenizer.encode("Evaluate parameters after training on 32K tokens.")
        eval_datum = make_cross_entropy_datum(eval_tokens)

        cloud_result = cloud_tc.forward([eval_datum], loss_fn="cross_entropy").result(timeout=300)
        local_result = local_tc.forward([eval_datum], loss_fn="cross_entropy").result(timeout=300)

        cloud_lp = _extract_logprobs(cloud_result)
        local_lp = _extract_logprobs(local_result)

        diff = np.abs(cloud_lp - local_lp)
        print(f"\n  32K train+eval: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")

        # After training on 32K, allow slightly more tolerance for accumulated FP precision
        np.testing.assert_allclose(
            cloud_lp, local_lp, rtol=5e-2, atol=5e-3,
            err_msg="Post-32K-training logprobs differ — parameters diverged",
        )
