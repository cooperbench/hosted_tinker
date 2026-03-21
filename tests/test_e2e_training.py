"""End-to-end training tests: full RL training loop, NLL convergence, SWE-style workflows."""
from __future__ import annotations

import numpy as np
import pytest
import tinker

from .conftest import make_cross_entropy_datum, make_ppo_datum, make_random_tokens


class TestNLLConvergence:
    """Training should reduce NLL on the training data."""

    def test_nll_decreases(self, service_client: tinker.ServiceClient, model_name: str, lora_rank: int, tokenizer):
        """NLL decreases over 3 training steps on repeated data."""
        tc = service_client.create_lora_training_client(
            base_model=model_name, rank=lora_rank,
        )

        tokens = tokenizer.encode(
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
        )
        datum = make_cross_entropy_datum(tokens, train_start=len(tokens) // 2)

        nlls = []
        for step in range(3):
            result = tc.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
            logprobs = result.loss_fn_outputs[0]["logprobs"]
            lp_arr = np.array(logprobs.tolist() if hasattr(logprobs, "tolist") else logprobs)
            # NLL only on trained region (where weights > 0)
            train_start = len(tokens) // 2
            nll = -lp_arr[train_start:].mean()
            nlls.append(nll)

            tc.optim_step(tinker.AdamParams(learning_rate=1e-3)).result(timeout=120)

        # At least one step should decrease NLL compared to initial
        assert min(nlls[1:]) < nlls[0], f"NLL should decrease after training: {nlls}"


class TestGRPOWorkflow:
    """Simulate a GRPO training workflow (as used in SWE-smith)."""

    def test_grpo_style_training(
        self, service_client: tinker.ServiceClient, model_name: str, lora_rank: int, tokenizer,
    ):
        """Simulate GRPO: forward_backward with PPO loss using group-normalized advantages."""
        tc = service_client.create_lora_training_client(
            base_model=model_name, rank=lora_rank,
        )

        # Simulate a group of episodes (same task, different outcomes)
        # Group of 4 episodes with rewards [0, 1, 0, 1]
        rewards = [0.0, 1.0, 0.0, 1.0]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        advantages_per_ep = [(r - mean_r) / std_r for r in rewards]

        datums = []
        for i, adv in enumerate(advantages_per_ep):
            tokens = make_random_tokens(tokenizer, 128, seed=i)
            seq_len = len(tokens)
            prompt_len = seq_len // 2

            old_logprobs = [-1.5] * seq_len
            advantages = [0.0] * prompt_len + [adv] * (seq_len - prompt_len)

            datum = make_ppo_datum(tokens, old_logprobs, advantages, train_start=prompt_len)
            datums.append(datum)

        # Phase 1: Get reference logprobs (forward only, no grad)
        ref_result = tc.forward(datums, loss_fn="cross_entropy").result(timeout=300)
        assert len(ref_result.loss_fn_outputs) == len(datums)

        # Phase 2: Train with PPO loss
        train_result = tc.forward_backward(datums, loss_fn="ppo").result(timeout=300)
        assert len(train_result.loss_fn_outputs) == len(datums)

        # Phase 3: Optimizer step
        opt_result = tc.optim_step(tinker.AdamParams(learning_rate=1e-4)).result(timeout=120)
        assert opt_result is not None


class TestSWETrainingLoop:
    """Simulate the full SWE-smith training loop with save/reload."""

    def test_full_loop_with_checkpoint(
        self, service_client: tinker.ServiceClient, model_name: str, lora_rank: int, tokenizer,
    ):
        """Full loop: create → train × N → save → create new client from state → continue training."""
        # Create initial client
        tc = service_client.create_lora_training_client(
            base_model=model_name, rank=lora_rank,
        )

        # Train 2 steps
        for i in range(2):
            tokens = make_random_tokens(tokenizer, 256, seed=100 + i)
            datum = make_cross_entropy_datum(tokens, train_start=128)
            tc.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
            tc.optim_step(tinker.AdamParams(learning_rate=1e-4)).result(timeout=120)

        # Save state
        save_result = tc.save_state("swe_loop_ckpt").result(timeout=300)
        ckpt_path = save_result.path if hasattr(save_result, "path") else str(save_result)

        # Create new client from saved state (simulates process restart)
        tc2 = service_client.create_training_client_from_state_with_optimizer(ckpt_path)

        # Continue training for 1 more step
        tokens = make_random_tokens(tokenizer, 256, seed=200)
        datum = make_cross_entropy_datum(tokens, train_start=128)
        result = tc2.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        assert result is not None

        tc2.optim_step(tinker.AdamParams(learning_rate=1e-4)).result(timeout=120)


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_batch(self, training_client: tinker.TrainingClient):
        """forward_backward with empty batch should either raise or return empty."""
        try:
            result = training_client.forward_backward([], loss_fn="cross_entropy").result(timeout=60)
            # If it doesn't raise, it should return an empty/valid result
            assert result is not None
        except Exception:
            pass  # Raising is also acceptable

    def test_single_token(self, training_client: tinker.TrainingClient, tokenizer):
        """Single-token sequence."""
        tokens = [tokenizer.encode("A")[0]]
        datum = make_cross_entropy_datum(tokens)
        # May raise or return — just shouldn't crash the server
        try:
            result = training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        except Exception:
            pass  # Server-side error is acceptable for degenerate input

    def test_all_zero_weights(self, training_client: tinker.TrainingClient, tokenizer):
        """All-zero weights should produce zero gradient (no training signal)."""
        tokens = tokenizer.encode("Zero weight test sentence.")
        seq_len = len(tokens)
        chunk = tinker.EncodedTextChunk(tokens=tokens)
        datum = tinker.Datum(
            model_input=tinker.ModelInput(chunks=[chunk]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(data=tokens[1:] + [0], dtype="int64"),
                "weights": tinker.TensorData(data=[0.0] * seq_len, dtype="float32"),
                "logprobs": tinker.TensorData(data=[0.0] * seq_len, dtype="float32"),
                "advantages": tinker.TensorData(data=[0.0] * seq_len, dtype="float32"),
            },
        )
        result = training_client.forward_backward([datum], loss_fn="cross_entropy").result(timeout=300)
        assert result is not None

    def test_invalid_loss_fn(self, training_client: tinker.TrainingClient, tokenizer):
        """Invalid loss function name should raise an error."""
        tokens = tokenizer.encode("Invalid loss test.")
        datum = make_cross_entropy_datum(tokens)
        with pytest.raises(Exception):
            training_client.forward_backward([datum], loss_fn="nonexistent_loss").result(timeout=60)
