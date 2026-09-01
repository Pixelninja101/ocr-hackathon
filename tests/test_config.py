"""
Tests for Centralized Configuration & Environment Overrides (tests/test_config.py).
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from risk_engine.config import (
    DEFAULT_MAX_RAW_SCORE,
    DEFAULT_MAX_UPLOAD_SIZE_BYTES,
    DEFAULT_NAME_SIMILARITY_THRESHOLD,
    DEFAULT_RISK_HIGH_THRESHOLD,
    DEFAULT_RISK_LOW_THRESHOLD,
    RiskEngineConfig,
)


class TestConfig(unittest.TestCase):
    """Test suite for RiskEngineConfig property parsing and validation."""

    def test_01_default_configuration_values(self) -> None:
        """Test 1: Default configuration values match system specifications."""
        cfg = RiskEngineConfig()
        self.assertEqual(cfg.MAX_UPLOAD_SIZE_BYTES, DEFAULT_MAX_UPLOAD_SIZE_BYTES)
        self.assertEqual(cfg.MAX_RAW_SCORE, DEFAULT_MAX_RAW_SCORE)
        self.assertEqual(cfg.RISK_LOW_THRESHOLD, DEFAULT_RISK_LOW_THRESHOLD)
        self.assertEqual(cfg.RISK_HIGH_THRESHOLD, DEFAULT_RISK_HIGH_THRESHOLD)
        self.assertEqual(cfg.NAME_SIMILARITY_THRESHOLD, DEFAULT_NAME_SIMILARITY_THRESHOLD)

    def test_02_environment_variable_overrides(self) -> None:
        """Test 2: Environment variables correctly override defaults within valid bounds."""
        env_vars = {
            "MAX_UPLOAD_SIZE_BYTES": "5242880",  # 5 MB
            "MAX_RAW_SCORE": "250",
            "RISK_LOW_THRESHOLD": "25",
            "RISK_HIGH_THRESHOLD": "70",
            "NAME_SIMILARITY_THRESHOLD": "0.90",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            cfg = RiskEngineConfig()
            self.assertEqual(cfg.MAX_UPLOAD_SIZE_BYTES, 5242880)
            self.assertEqual(cfg.MAX_RAW_SCORE, 250)
            self.assertEqual(cfg.RISK_LOW_THRESHOLD, 25)
            self.assertEqual(cfg.RISK_HIGH_THRESHOLD, 70)
            self.assertEqual(cfg.NAME_SIMILARITY_THRESHOLD, 0.90)

    def test_03_invalid_environment_variable_fallbacks(self) -> None:
        """Test 3: Non-numeric or negative values fall back safely to defaults."""
        invalid_env = {
            "MAX_UPLOAD_SIZE_BYTES": "-100",  # Negative
            "MAX_RAW_SCORE": "not_a_number",  # Non-numeric
            "RISK_LOW_THRESHOLD": "999",       # Out of bounds
            "NAME_SIMILARITY_THRESHOLD": "5.5", # Out of bounds float
        }
        with patch.dict(os.environ, invalid_env, clear=False):
            cfg = RiskEngineConfig()
            self.assertEqual(cfg.MAX_UPLOAD_SIZE_BYTES, DEFAULT_MAX_UPLOAD_SIZE_BYTES)
            self.assertEqual(cfg.MAX_RAW_SCORE, DEFAULT_MAX_RAW_SCORE)
            self.assertEqual(cfg.NAME_SIMILARITY_THRESHOLD, DEFAULT_NAME_SIMILARITY_THRESHOLD)

    def test_04_inverted_threshold_validation_fallback(self) -> None:
        """Test 4: Setting RISK_LOW_THRESHOLD >= RISK_HIGH_THRESHOLD falls back safely."""
        inverted_env = {
            "RISK_LOW_THRESHOLD": "80",
            "RISK_HIGH_THRESHOLD": "50",
        }
        with patch.dict(os.environ, inverted_env, clear=False):
            cfg = RiskEngineConfig()
            # LOW should fall back to default (30) when low >= high
            self.assertEqual(cfg.RISK_LOW_THRESHOLD, DEFAULT_RISK_LOW_THRESHOLD)

    def test_05_allowed_origins_parsing(self) -> None:
        """Test 5: Comma-separated ALLOWED_ORIGINS string parses into a clean list."""
        origins_env = {
            "ALLOWED_ORIGINS": "https://app.example.com, https://admin.example.com "
        }
        with patch.dict(os.environ, origins_env, clear=False):
            cfg = RiskEngineConfig()
            self.assertEqual(cfg.ALLOWED_ORIGINS, ["https://app.example.com", "https://admin.example.com"])


if __name__ == "__main__":
    unittest.main()
