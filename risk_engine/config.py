"""
Centralized Configuration for Risk Engine & Document Verification API.
Provides validated, environment-variable-backed settings with robust fallbacks.
"""

from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger("risk_engine.config")

# -----------------------------------------------------------------------------
# Default Constants
# -----------------------------------------------------------------------------

DEFAULT_MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
DEFAULT_MAX_RAW_SCORE: int = 200
DEFAULT_RISK_LOW_THRESHOLD: int = 30     # 0–29 -> LOW
DEFAULT_RISK_HIGH_THRESHOLD: int = 60    # 30–59 -> MEDIUM, 60–100 -> HIGH

DEFAULT_NAME_SIMILARITY_THRESHOLD: float = 0.85
DEFAULT_DOC_LOW_CONFIDENCE_THRESHOLD: float = 0.80
DEFAULT_OCR_LOW_CONFIDENCE_THRESHOLD: float = 0.70
DEFAULT_FIELD_LOW_CONFIDENCE_THRESHOLD: float = 0.60

DEFAULT_ALLOWED_ORIGINS: str = (
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
)


# -----------------------------------------------------------------------------
# Safe Environment Variable Parsers
# -----------------------------------------------------------------------------

def _parse_int_env(key: str, default: int, min_val: int = 1, max_val: int = 100000000) -> int:
    """Safely parses an integer from environment variables with bounds checking."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw.strip())
        if min_val <= val <= max_val:
            return val
        logger.warning(
            "Config %s=%s out of bounds [%d, %d]. Using default: %d",
            key, raw, min_val, max_val, default
        )
        return default
    except (ValueError, TypeError):
        logger.warning("Invalid integer for config %s: '%s'. Using default: %d", key, raw, default)
        return default


def _parse_float_env(key: str, default: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Safely parses a float from environment variables with bounds checking."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        val = float(raw.strip())
        if min_val <= val <= max_val:
            return val
        logger.warning(
            "Config %s=%s out of bounds [%f, %f]. Using default: %f",
            key, raw, min_val, max_val, default
        )
        return default
    except (ValueError, TypeError):
        logger.warning("Invalid float for config %s: '%s'. Using default: %f", key, raw, default)
        return default


def _parse_origins_env(key: str, default_str: str) -> List[str]:
    """Safely parses a comma-separated list of allowed CORS origins."""
    raw = os.getenv(key, default_str)
    if not raw or not raw.strip():
        return []
    return [orig.strip() for orig in raw.split(",") if orig.strip()]


# -----------------------------------------------------------------------------
# Configuration Properties
# -----------------------------------------------------------------------------

class RiskEngineConfig:
    """Centralized configuration container with dynamic environment support."""

    @property
    def MAX_UPLOAD_SIZE_BYTES(self) -> int:
        return _parse_int_env("MAX_UPLOAD_SIZE_BYTES", DEFAULT_MAX_UPLOAD_SIZE_BYTES, min_val=1024)

    @property
    def MAX_RAW_SCORE(self) -> int:
        return _parse_int_env("MAX_RAW_SCORE", DEFAULT_MAX_RAW_SCORE, min_val=10, max_val=1000)

    @property
    def RISK_LOW_THRESHOLD(self) -> int:
        low = _parse_int_env("RISK_LOW_THRESHOLD", DEFAULT_RISK_LOW_THRESHOLD, min_val=1, max_val=99)
        high = self.RISK_HIGH_THRESHOLD
        if low >= high:
            logger.warning("RISK_LOW_THRESHOLD (%d) >= RISK_HIGH_THRESHOLD (%d). Using default %d", low, high, DEFAULT_RISK_LOW_THRESHOLD)
            return DEFAULT_RISK_LOW_THRESHOLD
        return low

    @property
    def RISK_HIGH_THRESHOLD(self) -> int:
        return _parse_int_env("RISK_HIGH_THRESHOLD", DEFAULT_RISK_HIGH_THRESHOLD, min_val=2, max_val=100)

    @property
    def NAME_SIMILARITY_THRESHOLD(self) -> float:
        return _parse_float_env("NAME_SIMILARITY_THRESHOLD", DEFAULT_NAME_SIMILARITY_THRESHOLD)

    @property
    def DOC_LOW_CONFIDENCE_THRESHOLD(self) -> float:
        return _parse_float_env("DOC_LOW_CONFIDENCE_THRESHOLD", DEFAULT_DOC_LOW_CONFIDENCE_THRESHOLD)

    @property
    def OCR_LOW_CONFIDENCE_THRESHOLD(self) -> float:
        return _parse_float_env("OCR_LOW_CONFIDENCE_THRESHOLD", DEFAULT_OCR_LOW_CONFIDENCE_THRESHOLD)

    @property
    def FIELD_LOW_CONFIDENCE_THRESHOLD(self) -> float:
        return _parse_float_env("FIELD_LOW_CONFIDENCE_THRESHOLD", DEFAULT_FIELD_LOW_CONFIDENCE_THRESHOLD)

    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        return _parse_origins_env("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)


# Singleton configuration instance
config = RiskEngineConfig()

# Expose direct module-level exports matching existing names for backwards compatibility
MAX_UPLOAD_SIZE_BYTES = config.MAX_UPLOAD_SIZE_BYTES
MAX_RAW_SCORE = config.MAX_RAW_SCORE
LOW_RISK_THRESHOLD = config.RISK_LOW_THRESHOLD
HIGH_RISK_THRESHOLD = config.RISK_HIGH_THRESHOLD
NAME_SIMILARITY_THRESHOLD = config.NAME_SIMILARITY_THRESHOLD
DOC_LOW_CONFIDENCE_THRESHOLD = config.DOC_LOW_CONFIDENCE_THRESHOLD
OCR_LOW_CONFIDENCE_THRESHOLD = config.OCR_LOW_CONFIDENCE_THRESHOLD
FIELD_LOW_CONFIDENCE_THRESHOLD = config.FIELD_LOW_CONFIDENCE_THRESHOLD
ALLOWED_ORIGINS = config.ALLOWED_ORIGINS
