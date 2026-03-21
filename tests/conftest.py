"""Shared fixtures for hosted-tinker test suite.

Configure via environment variables:
    TINKER_BASE_URL  - Server URL (default: http://localhost:8000)
    TINKER_API_KEY   - API key (default: tml-dummy)
    TINKER_MODEL     - Model to test (default: Qwen/Qwen3-30B-A3B)
    TINKER_LORA_RANK - LoRA rank (default: 32)
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import tinker


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@pytest.fixture(scope="session")
def base_url() -> str:
    return _env("TINKER_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def model_name() -> str:
    return _env("TINKER_MODEL", "Qwen/Qwen3-30B-A3B")


@pytest.fixture(scope="session")
def lora_rank() -> int:
    return int(_env("TINKER_LORA_RANK", "32"))


@pytest.fixture(scope="session")
def service_client(base_url: str) -> tinker.ServiceClient:
    return tinker.ServiceClient(base_url=base_url)


@pytest.fixture(scope="module")
def training_client(
    service_client: tinker.ServiceClient,
    model_name: str,
    lora_rank: int,
) -> tinker.TrainingClient:
    """Create a fresh training client for each test module."""
    return service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank,
    )


@pytest.fixture(scope="session")
def tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Datum helpers
# ---------------------------------------------------------------------------

def make_cross_entropy_datum(
    tokens: list[int],
    train_start: int | None = None,
) -> tinker.Datum:
    """Create a Datum for cross_entropy loss.

    The Tinker API expects:
    - model_input: token sequence (the full input including the token to predict)
    - target_tokens: shifted by 1 (tokens[1:] + [0]) — what to predict at each position
    - weights: loss mask (0 for prefix, 1 for training region)
    - logprobs: old log-probs (zeros for cross_entropy, used for PPO)
    - advantages: advantages (zeros for cross_entropy, used for PPO)
    """
    seq_len = len(tokens)
    if train_start is None:
        train_start = 0
    weights = [0.0] * train_start + [1.0] * (seq_len - train_start)
    # Target tokens: shifted by 1 (predict next token)
    target_tokens = tokens[1:] + [0]
    # Zeros for unused fields
    zeros = [0.0] * seq_len

    chunk = tinker.EncodedTextChunk(tokens=tokens)
    return tinker.Datum(
        model_input=tinker.ModelInput(chunks=[chunk]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=target_tokens, dtype="int64"),
            "weights": tinker.TensorData(data=weights, dtype="float32"),
            "logprobs": tinker.TensorData(data=zeros, dtype="float32"),
            "advantages": tinker.TensorData(data=zeros, dtype="float32"),
        },
    )


def make_ppo_datum(
    tokens: list[int],
    old_logprobs: list[float],
    advantages: list[float],
    train_start: int = 0,
) -> tinker.Datum:
    """Create a Datum for PPO loss."""
    seq_len = len(tokens)
    weights = [0.0] * train_start + [1.0] * (seq_len - train_start)
    target_tokens = tokens[1:] + [0]

    chunk = tinker.EncodedTextChunk(tokens=tokens)
    return tinker.Datum(
        model_input=tinker.ModelInput(chunks=[chunk]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=target_tokens, dtype="int64"),
            "weights": tinker.TensorData(data=weights, dtype="float32"),
            "logprobs": tinker.TensorData(data=old_logprobs, dtype="float32"),
            "advantages": tinker.TensorData(data=advantages, dtype="float32"),
        },
    )


def make_random_tokens(tokenizer, length: int, seed: int = 42) -> list[int]:
    """Generate a random token sequence of approximately the given length."""
    rng = np.random.RandomState(seed)
    vocab_size = tokenizer.vocab_size
    return rng.randint(100, vocab_size, size=length).tolist()
