"""Logging utilities (inlined from skyrl.utils.log)."""

import logging

from rich.logging import RichHandler

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"

RICH_HANDLER_KWARGS = {
    "show_time": False,
    "show_level": False,
    "show_path": False,
    "markup": False,
    "rich_tracebacks": True,
}


def _create_rich_handler() -> RichHandler:
    """Create a RichHandler with consistent configuration."""
    handler = RichHandler(**RICH_HANDLER_KWARGS)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def _setup_root_logger() -> None:
    _logger = logging.getLogger("hosted_tinker")
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False
    _logger.addHandler(_create_rich_handler())


def get_uvicorn_log_config() -> dict:
    """Get uvicorn logging config that uses the same RichHandler."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": LOG_FORMAT,
            },
        },
        "handlers": {
            "default": {
                "()": RichHandler,
                **RICH_HANDLER_KWARGS,
                "formatter": "default",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }


_setup_root_logger()
logger = logging.getLogger("hosted_tinker")

__all__ = ["logger", "get_uvicorn_log_config"]
