"""
Cross-validation and fuzzy matching between OCR-extracted data and QR-extracted data.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, Optional, Tuple

from document_processor.config import NAME_SIMILARITY_THRESHOLD
from document_processor.ocr.normalization import (
    normalize_dob,
    normalize_gender,
    normalize_name,
)


def calculate_string_similarity(str1: str, str2: str) -> float:
    """
    Computes Levenshtein / SequenceMatcher similarity ratio [0.0, 1.0].
    """
    if not str1 or not str2:
        return 0.0
    matcher = difflib.SequenceMatcher(None, str1, str2)
    return round(matcher.ratio(), 2)


def match_names(
    ocr_name: Optional[str],
    qr_name: Optional[str],
    threshold: float = NAME_SIMILARITY_THRESHOLD,
) -> Optional[dict[str, Any]]:
    """
    Performs fuzzy matching between OCR name and QR name.
    """
    if not ocr_name or not qr_name:
        return None

    norm_ocr = normalize_name(ocr_name)
    norm_qr = normalize_name(qr_name)

    similarity = calculate_string_similarity(norm_ocr, norm_qr)
    is_match = similarity >= threshold

    return {
        "similarity": similarity,
        "match": is_match,
    }


def match_dob(
    ocr_dob: Optional[dict[str, Any] | str],
    qr_dob: Optional[dict[str, Any] | str],
) -> Optional[dict[str, Any]]:
    """
    Compares Date/Year of Birth handling different precision levels:
    - Full Date vs Full Date (precision: full)
    - Full Date vs Year-Only (precision: year)
    - Year-Only vs Year-Only (precision: year)
    """
    if ocr_dob is None or qr_dob is None:
        return None

    # Ensure dictionaries
    if isinstance(ocr_dob, str):
        ocr_dict = normalize_dob(ocr_dob)
    else:
        ocr_dict = ocr_dob

    if isinstance(qr_dob, str):
        qr_dict = normalize_dob(qr_dob)
    else:
        qr_dict = qr_dob

    if not ocr_dict or not qr_dict or not ocr_dict.get("year") or not qr_dict.get("year"):
        return None

    ocr_year = int(ocr_dict["year"])
    qr_year = int(qr_dict["year"])

    ocr_prec = ocr_dict.get("precision", "year")
    qr_prec = qr_dict.get("precision", "year")

    # If both have full date precision, compare day, month, and year
    if (
        ocr_prec == "full"
        and qr_prec == "full"
        and ocr_dict.get("day") is not None
        and ocr_dict.get("month") is not None
        and qr_dict.get("day") is not None
        and qr_dict.get("month") is not None
    ):
        full_match = (
            ocr_year == qr_year
            and int(ocr_dict["month"]) == int(qr_dict["month"])
            and int(ocr_dict["day"]) == int(qr_dict["day"])
        )
        return {
            "match": full_match,
            "comparison": "full",
        }

    # Otherwise compare year-level precision
    year_match = ocr_year == qr_year
    return {
        "match": year_match,
        "comparison": "year",
    }


def match_gender(
    ocr_gender: Optional[str | dict[str, Any]],
    qr_gender: Optional[str],
) -> Optional[dict[str, Any]]:
    """
    Compares normalized gender representations.
    """
    if ocr_gender is None or qr_gender is None:
        return None

    if isinstance(ocr_gender, dict):
        ocr_val = ocr_gender.get("value")
    else:
        ocr_val = ocr_gender

    norm_ocr = normalize_gender(ocr_val)
    norm_qr = normalize_gender(qr_gender)

    if not norm_ocr or not norm_qr:
        return None

    return {
        "match": (norm_ocr == norm_qr)
    }


def cross_validate_ocr_and_qr(
    ocr_fields: dict[str, Any],
    qr_fields: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """
    Executes complete cross-validation between OCR and QR fields.
    Returns comparison dictionary or None if QR fields are missing.
    """
    if not qr_fields:
        return None

    cross_val: dict[str, Any] = {}

    # 1. Name Match
    ocr_name_dict = ocr_fields.get("name")
    ocr_name_val = ocr_name_dict.get("value") if isinstance(ocr_name_dict, dict) else ocr_name_dict
    qr_name_val = qr_fields.get("name")
    name_res = match_names(ocr_name_val, qr_name_val)
    if name_res:
        cross_val["name"] = name_res

    # 2. DOB Match
    ocr_dob_dict = ocr_fields.get("dob")
    qr_dob_val = qr_fields.get("dob")
    dob_res = match_dob(ocr_dob_dict, qr_dob_val)
    if dob_res:
        cross_val["dob"] = dob_res

    # 3. Gender Match
    ocr_gender_dict = ocr_fields.get("gender")
    qr_gender_val = qr_fields.get("gender")
    gender_res = match_gender(ocr_gender_dict, qr_gender_val)
    if gender_res:
        cross_val["gender"] = gender_res

    return cross_val if cross_val else None
