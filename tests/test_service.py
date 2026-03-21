"""Tests for ServiceClient: server capabilities, model creation, health checks."""
from __future__ import annotations

import pytest
import tinker


class TestHealthAndCapabilities:
    """Server health and capability queries."""

    def test_server_reachable(self, base_url: str):
        """Server responds to health check."""
        import httpx
        r = httpx.get(f"{base_url}/api/v1/healthz", timeout=10)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_get_server_capabilities(self, service_client: tinker.ServiceClient):
        """Server reports supported models."""
        caps = service_client.get_server_capabilities()
        assert hasattr(caps, "supported_models")
        assert len(caps.supported_models) > 0

    def test_supported_model_name(self, service_client: tinker.ServiceClient, model_name: str):
        """The configured test model is in the supported list."""
        caps = service_client.get_server_capabilities()
        model_names = [m.model_name for m in caps.supported_models]
        assert model_name in model_names, f"{model_name} not in {model_names}"


class TestCreateTrainingClient:
    """Creating and managing training clients."""

    def test_create_lora_training_client(self, service_client: tinker.ServiceClient, model_name: str, lora_rank: int):
        """Can create a LoRA training client."""
        tc = service_client.create_lora_training_client(
            base_model=model_name,
            rank=lora_rank,
        )
        assert tc is not None

    def test_get_info(self, training_client: tinker.TrainingClient):
        """TrainingClient.get_info() returns model metadata."""
        info = training_client.get_info()
        assert hasattr(info, "model_data")
        assert info.model_data.model_name is not None

    def test_get_tokenizer(self, training_client: tinker.TrainingClient):
        """TrainingClient.get_tokenizer() returns a working tokenizer."""
        tok = training_client.get_tokenizer()
        encoded = tok.encode("Hello world")
        assert len(encoded) > 0
        decoded = tok.decode(encoded)
        assert "Hello" in decoded

    def test_create_with_custom_lora_config(self, service_client: tinker.ServiceClient, model_name: str):
        """LoRA config options (rank, train_mlp, train_attn) are accepted."""
        tc = service_client.create_lora_training_client(
            base_model=model_name,
            rank=16,
            train_mlp=True,
            train_attn=True,
            train_unembed=False,
        )
        assert tc is not None
