"""
Test fixtures package for deterministic scenario validation and demonstrations.
"""

from tests.fixtures.verification_cases import (
    CASE_CLEAN,
    CASE_DOB_MISMATCH,
    CASE_INVALID_CHECKSUM,
    CASE_NAME_MISMATCH,
    CASE_NON_AADHAAR,
    CASE_QR_UNAVAILABLE,
    DEMO_CASES,
)

__all__ = [
    "CASE_CLEAN",
    "CASE_INVALID_CHECKSUM",
    "CASE_NAME_MISMATCH",
    "CASE_DOB_MISMATCH",
    "CASE_QR_UNAVAILABLE",
    "CASE_NON_AADHAAR",
    "DEMO_CASES",
]
