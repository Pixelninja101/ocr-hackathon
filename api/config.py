"""
API Configuration Layer for Document Verification Engine.

Reads settings from environment variables with defensive parsing, bounds checking,
and safe defaults suitable for development and production deployments.
"""

from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger("api.config")

# -----------------------------------------------------------------------------
# Default Constants
# -----------------------------------------------------------------------------

DEFAULT_MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
DEFAULT_RATE_LIMIT_REQUESTS: int = 60                   # 60 requests
DEFAULT_RATE_LIMIT_WINDOW_SECONDS: int = 60            # per 60 seconds
DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_ENVIRONMENT: str = "development"
DEFAULT_ALLOWED_ORIGINS: str = (
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
)

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
VALID_ENVIRONMENTS = {"development", "staging", "production", "test"}


# -----------------------------------------------------------------------------
# Safe Environment Parsers
# -----------------------------------------------------------------------------

def _parse_int(key: str, default: int, min_val: int = 1, max_val: int = 100_000_000) -> int:
    """Defensively parses integer environment variables with bounds checking."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw.strip())
        if min_val <= val <= max_val:
            return val
        logger.warning(
            "Config %s=%s out of allowed range [%d, %d]. Falling back to default %d.",
            key, raw, min_val, max_val, default
        )
        return default
    except (ValueError, TypeError):
        logger.warning(
            "Invalid integer format for config %s='%s'. Falling back to default %d.",
            key, raw, default
        )
        return default


def _parse_str(key: str, default: str, valid_options: set[str] | None = None) -> str:
    """Defensively parses string environment variables with optional enum validation."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    val = raw.strip()
    if valid_options and val.upper() not in valid_options and val.lower() not in valid_options:
        logger.warning(
            "Config %s='%s' not in valid options %s. Falling back to default '%s'.",
            key, raw, valid_options, default
        )
        return default
    return val


def _parse_origins(key: str, default_str: str) -> List[str]:
    """Parses comma-separated allowed CORS origins list."""
    raw = os.getenv(key, default_str)
    if not raw or not raw.strip():
        return []
    return [orig.strip() for orig in raw.split(",") if orig.strip()]


# -----------------------------------------------------------------------------
# Configuration Container
# -----------------------------------------------------------------------------

class APIConfig:
    """
    Centralized, dynamic API configuration container.
    Properties evaluate lazily to support dynamic environment variable overrides in tests.
    """

    @property
    def MAX_UPLOAD_SIZE_BYTES(self) -> int:
        return _parse_int("MAX_UPLOAD_SIZE_BYTES", DEFAULT_MAX_UPLOAD_SIZE_BYTES, min_val=1024)

    @property
    def RATE_LIMIT_REQUESTS(self) -> int:
        return _parse_int("RATE_LIMIT_REQUESTS", DEFAULT_RATE_LIMIT_REQUESTS, min_val=1, max_val=100_000)

    @property
    def RATE_LIMIT_WINDOW_SECONDS(self) -> int:
        return _parse_int("RATE_LIMIT_WINDOW_SECONDS", DEFAULT_RATE_LIMIT_WINDOW_SECONDS, min_val=1, max_val=86_400)

    @property
    def LOG_LEVEL(self) -> str:
        parsed = _parse_str("LOG_LEVEL", DEFAULT_LOG_LEVEL, VALID_LOG_LEVELS).upper()
        return parsed if parsed in VALID_LOG_LEVELS else DEFAULT_LOG_LEVEL

    @property
    def ENVIRONMENT(self) -> str:
        parsed = _parse_str("ENVIRONMENT", DEFAULT_ENVIRONMENT, VALID_ENVIRONMENTS).lower()
        return parsed if parsed in VALID_ENVIRONMENTS else DEFAULT_ENVIRONMENT

    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        return _parse_origins("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)

    def validate(self) -> bool:
        """
        Validates configuration consistency at application startup.
        Returns True if valid; logs warnings for non-fatal irregularities.
        """
        if self.MAX_UPLOAD_SIZE_BYTES < 1024:
            raise ValueError("MAX_UPLOAD_SIZE_BYTES must be at least 1024 bytes.")
        if self.RATE_LIMIT_REQUESTS < 1:
            raise ValueError("RATE_LIMIT_REQUESTS must be at least 1.")
        if self.RATE_LIMIT_WINDOW_SECONDS < 1:
            raise ValueError("RATE_LIMIT_WINDOW_SECONDS must be at least 1.")
        return True


# Global singleton instance
api_config = APIConfig()
