"""Configuration for the Tinker engine."""

import argparse
import json
import os
from pathlib import Path

from cloudpathlib import AnyPath
from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """Configuration for the Tinker engine."""

    base_model: str = Field(..., description="Base model name (e.g., Qwen/Qwen3-0.6B)")
    backend: str = Field(default="jax", description="Backend to use for training and inference")
    backend_config: dict = Field(
        default_factory=dict,
        description="Backend-specific configuration as JSON string",
        json_schema_extra={"argparse_type": json.loads},
    )
    checkpoints_base: AnyPath = Field(
        default=AnyPath("/tmp/skyrl_checkpoints"),
        description="Base path where checkpoints will be stored",
    )
    database_url: str = Field(
        default=f'sqlite:///{Path(__file__).parent / "tinker.db"}',
        description="Database URL (e.g., postgresql://user:password@localhost:5432/tinker). If not set, uses SKYRL_DATABASE_URL env var or defaults to SQLite",
        json_schema_extra={"argparse_type": str, "env_var": "SKYRL_DATABASE_URL"},
    )
    external_inference_url: str | None = Field(
        default=None,
        description="URL of the external inference engine. If set, sample requests will be sent to the external engine instead (currently only VLLM is supported).",
        json_schema_extra={"argparse_type": str},
    )
    external_inference_api_key: str = Field(
        default="EMPTY",
        description="API key for an external inference engine. If not provided will use vLLM 'EMPTY' key convention",
    )
    external_inference_lora_base: Path = Field(
        default=Path("/tmp/lora_models"),
        description="Directory where LoRA models will be extracted for external inference engines",
    )
    session_cleanup_interval_sec: int = Field(
        default=60,
        description="How often to check for stale sessions (seconds). Set to -1 to disable cleanup.",
    )
    session_timeout_sec: int = Field(
        default=300,
        description="Seconds without heartbeat before session is considered stale. Set to -1 to disable cleanup.",
    )
    # vLLM inference server (auto-launched for split-GPU mode)
    vllm_gpus: str = Field(
        default="",
        description="GPU IDs for vLLM inference (e.g., '4,5,6,7'). If set, auto-launches vLLM.",
    )
    vllm_port: int = Field(default=8001, description="Port for vLLM inference server")
    vllm_tp: int = Field(default=1, description="vLLM tensor parallel size")
    vllm_max_model_len: int = Field(default=32768, description="Max model length for vLLM")
    vllm_max_num_seqs: int = Field(default=16, description="Max concurrent sequences for vLLM")
    vllm_gpu_mem: float = Field(default=0.90, description="vLLM GPU memory utilization")
    vllm_max_lora_rank: int = Field(default=32, description="Max LoRA rank for vLLM")


def convert_env_var(env_name: str, env_value: str, expected_type: type):
    """Convert environment variable to expected type."""
    if expected_type is bool:
        if env_value not in ("0", "1"):
            raise ValueError(
                f"Environment variable '{env_name}' for a boolean flag must be '0' or '1', but got '{env_value}'."
            )
        return env_value == "1"
    else:
        return env_value


def add_model(parser: argparse.ArgumentParser, model: type[BaseModel]) -> None:
    """Add Pydantic model fields to an ArgumentParser."""
    for name, field in model.model_fields.items():
        arg_name = name.replace("_", "-")
        kwargs = {
            "help": field.description,
        }

        default_value = field.default
        if field.json_schema_extra and "env_var" in field.json_schema_extra:
            env_name = field.json_schema_extra["env_var"]
            if env_value := os.environ.get(env_name):
                default_value = convert_env_var(env_name, env_value, field.annotation)

        if field.annotation is bool:
            kwargs = {**kwargs, "action": argparse.BooleanOptionalAction, "dest": name, "default": default_value}
        else:
            argparse_type = field.json_schema_extra.get("argparse_type") if field.json_schema_extra else None
            if argparse_type is not None:
                kwargs["type"] = argparse_type
            elif field.annotation is not None:
                kwargs["type"] = field.annotation

            if field.is_required():
                kwargs["required"] = True
            else:
                kwargs["default"] = default_value

        parser.add_argument(f"--{arg_name}", **kwargs)


def config_to_argv(cfg: BaseModel) -> list[str]:
    """This should 'unparse' a config parsed by an ArgumentParser constructed by add_model."""
    argv = []
    for field_name, value in cfg.model_dump().items():
        field = cfg.model_fields[field_name]
        arg_name = field_name.replace("_", "-")

        if field.annotation is bool:
            argv.append(f"--{arg_name}" if value else f"--no-{arg_name}")
        elif field.annotation is dict:
            if value:
                argv.append(f"--{arg_name}")
                argv.append(json.dumps(value))
        else:
            if value is not None:
                argv.append(f"--{arg_name}")
                argv.append(str(value))
    return argv
