"""Abstract backend interface for TinkerEngine.

Backends handle all model state, adapter management, and computation.
The engine handles database operations and scheduling.

Standalone copy — no skyrl dependency.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from hosted_tinker import types


class AbstractBackend(ABC):
    """Abstract base class for TinkerEngine backends.

    Backends handle computation and model state manipulation.
    Database operations are handled by TinkerEngine.
    """

    @abstractmethod
    def __init__(self, base_model: str, config: BaseModel):
        """Initialize the backend."""
        pass

    @abstractmethod
    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        """Create a new model in the backend."""
        pass

    @abstractmethod
    def forward_backward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Run forward and backward pass on a batch."""
        pass

    @abstractmethod
    def forward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Run forward-only pass on a batch (no gradient computation)."""
        pass

    @abstractmethod
    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        """Apply an optimizer step using accumulated gradients."""
        pass

    @abstractmethod
    def sample(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Generate samples for a batch of requests."""
        pass

    @abstractmethod
    def save_checkpoint(self, output_path, model_id: str) -> None:
        """Save training checkpoint to disk."""
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path, model_id: str) -> None:
        """Load training checkpoint from disk."""
        pass

    @abstractmethod
    def save_sampler_checkpoint(self, output_path, model_id: str, persist: bool = True) -> None:
        """Prepare model weights for sampling and optionally save to disk."""
        pass

    @abstractmethod
    def has_model(self, model_id: str) -> bool:
        """Check if a model is registered with the backend."""
        pass

    @abstractmethod
    def delete_model(self, model_id: str) -> None:
        """Delete a model and free all associated resources."""
        pass
