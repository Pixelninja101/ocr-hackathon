"""
Normalization helpers for OCR and QR fields (Names, Dates of Birth, Genders).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


def normalize_name(name: Optional[str]) -> str:
    """
    Normalizes person names for matching:
    - Trims whitespace
    - Lowercases text
    - Collapses repeated whitespace
    - Normalizes common punctuation differences (removes noise characters like dots, commas, titles)
    """
    if not name:
        return ""

    # Replace common punctuation with space
    cleaned = re.sub(r"[.,_\\/|:\-\–—*#;]", " ", str(name))
    # Collapse multiple spaces and strip
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def normalize_gender(gender_val: Optional[str]) -> Optional[str]:
    """
    Normalizes gender representations across English and Hindi variations.
    Maps to canonical: 'MALE', 'FEMALE', 'TRANSGENDER', or None.
    """
    if not gender_val:
        return None

    cleaned = str(gender_val).strip().lower()

    # Hindi normalization
    if any(k in cleaned for k in ["पुरुष", "पु०", "पु.", "मर्द", "नर"]):
        return "MALE"
    if any(k in cleaned for k in ["महिला", "स्त्री", "स्त्री.", "औरत", "मादा"]):
        return "FEMALE"
    if any(k in cleaned for k in ["तृतीय लिंग", "किन्नर", "ट्रांसजेंडर"]):
        return "TRANSGENDER"

    # English normalization
    if cleaned in ("m", "male", "m.") or cleaned.startswith("male"):
        return "MALE"
    if cleaned in ("f", "female", "f.") or cleaned.startswith("female"):
        return "FEMALE"
    if cleaned in ("t", "transgender", "other", "trans"):
        return "TRANSGENDER"

    return None


def is_valid_calendar_date(day: int, month: int, year: int) -> bool:
    """Validates calendar date ranges (day, month, year)."""
    if not (1900 <= year <= 2030):
        return False
    if not (1 <= month <= 12):
        return False
    # Check day boundaries
    days_in_month = [31, 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if not (1 <= day <= days_in_month[month - 1]):
        return False
    return True


def normalize_dob(dob_str: Optional[str]) -> Optional[dict[str, Any]]:
    """
    Normalizes date of birth or year of birth into a precision-aware structure:
    {
        "year": int,
        "month": int | None,
        "day": int | None,
        "precision": "year" | "full",
        "iso": str | None  # "YYYY-MM-DD"
    }
    """
    if not dob_str:
        return None

    cleaned = str(dob_str).strip()

    # Check for Full Date: DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    full_date_match = re.search(
        r"\b(0?[1-9]|[12][0-9]|3[01])[\/\-\.\s](0?[1-9]|1[012])[\/\-\.\s](19\d\d|20\d\d)\b",
        cleaned,
    )
    if full_date_match:
        day = int(full_date_match.group(1))
        month = int(full_date_match.group(2))
        year = int(full_date_match.group(3))
        if is_valid_calendar_date(day, month, year):
            return {
                "year": year,
                "month": month,
                "day": day,
                "precision": "full",
                "iso": f"{year:04d}-{month:02d}-{day:02d}",
            }

    # Check for ISO format: YYYY-MM-DD
    iso_date_match = re.search(
        r"\b(19\d\d|20\d\d)[\/\-\.](0?[1-9]|1[012])[\/\-\.](0?[1-9]|[12][0-9]|3[01])\b",
        cleaned,
    )
    if iso_date_match:
        year = int(iso_date_match.group(1))
        month = int(iso_date_match.group(2))
        day = int(iso_date_match.group(3))
        if is_valid_calendar_date(day, month, year):
            return {
                "year": year,
                "month": month,
                "day": day,
                "precision": "full",
                "iso": f"{year:04d}-{month:02d}-{day:02d}",
            }

    # Check for Year-Only: e.g. "2002" or "1995"
    year_match = re.search(r"\b(19\d\d|20\d\d)\b", cleaned)
    if year_match:
        year = int(year_match.group(1))
        if 1900 <= year <= 2030:
            return {
                "year": year,
                "month": None,
                "day": None,
                "precision": "year",
                "iso": None,
            }

    return None


def format_dob_for_display(dob_dict: Optional[dict[str, Any]]) -> str:
    """Formats normalized DOB dictionary into human-readable string (DD/MM/YYYY or YYYY)."""
    if not dob_dict:
        return ""
    if dob_dict.get("precision") == "full" and dob_dict.get("day") and dob_dict.get("month"):
        return f"{dob_dict['day']:02d}/{dob_dict['month']:02d}/{dob_dict['year']}"
    if dob_dict.get("year"):
        return str(dob_dict["year"])
    return ""
