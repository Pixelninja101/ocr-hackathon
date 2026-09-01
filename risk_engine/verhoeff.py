"""
Verhoeff Checksum Algorithm Implementation for Aadhaar Number Validation.

Security & Correctness:
- Verhoeff validation checks digit structure and transposition errors.
- It does NOT prove document authenticity or UIDAI issuance.
- Full Aadhaar numbers are never logged, stored, or exposed in return values.
"""

from __future__ import annotations

import re
from typing import Optional, Union

# Verhoeff Multiplication Table (dihedral group D5)
_VERHOEFF_D: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

# Verhoeff Permutation Table
_VERHOEFF_P: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)

# Verhoeff Inverse Table
_VERHOEFF_INV: tuple[int, ...] = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def calculate_verhoeff_check_digit(first_11_digits: Union[str, int]) -> int:
    """
    Calculates the 12th Verhoeff check digit for an 11-digit Aadhaar prefix.

    Parameters:
        first_11_digits (str | int): Exactly 11 decimal digits (spaces/hyphens permitted).

    Returns:
        int: Calculated single check digit (0-9).

    Raises:
        ValueError: If the normalized input does not contain exactly 11 numeric digits.
    """
    if first_11_digits is None:
        raise ValueError("Input for Verhoeff check digit calculation cannot be None.")

    raw_str = str(first_11_digits).strip()
    clean = "".join(ch for ch in raw_str if ch.isdigit())

    if len(clean) != 11:
        raise ValueError(
            f"Expected exactly 11 digits for check digit calculation, got {len(clean)}."
        )

    c = 0
    # In Verhoeff check digit generation, position index starts from 1 (the check digit will be at index 0)
    for i, ch in enumerate(reversed(clean), start=1):
        digit = int(ch)
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][digit]]

    return _VERHOEFF_INV[c]


def validate_aadhaar_checksum(aadhaar_number: Optional[Union[str, int]]) -> bool:
    """
    Validates whether a 12-digit Aadhaar number satisfies the Verhoeff checksum.

    Parameters:
        aadhaar_number (str | int | None): Complete 12-digit Aadhaar number
                                           (spaces and hyphens are ignored).

    Returns:
        bool: True if checksum is valid, False otherwise (including for malformed/masked inputs).
    """
    if aadhaar_number is None:
        return False

    raw_str = str(aadhaar_number).strip()
    if not raw_str:
        return False

    # Check if value contains non-numeric masked characters like 'X'
    if "X" in raw_str.upper():
        return False

    # Extract digits only
    clean = "".join(ch for ch in raw_str if ch.isdigit())

    # Must be exactly 12 numeric digits
    if len(clean) != 12:
        return False

    # Compute Verhoeff checksum accumulator
    c = 0
    for i, ch in enumerate(reversed(clean)):
        digit = int(ch)
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][digit]]

    return c == 0
