"""
Synthetic risk scenario builder and demonstrator for Risk Engine.
Provides deterministic scenario payloads covering scenarios A through I.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure risk_engine package root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from risk_engine import assess_document


def get_scenario_a_clean_aadhaar() -> Dict[str, Any]:
    """Scenario A: Clean Aadhaar document with good OCR and decoded QR."""
    return {
        "success": True,
        "document": {"type": "aadhaar", "confidence": 0.98},
        "ocr": {
            "language": "eng+hin",
            "confidence": 0.95,
            "fields": {
                "name": {"value": "ANANYA SHARMA", "confidence": 0.96},
                "dob": {"year": 1995, "month": 6, "day": 20, "precision": "full", "confidence": 0.94},
                "gender": {"value": "FEMALE", "confidence": 0.98},
                "aadhaar_number": {"value": "XXXX XXXX 4321", "confidence": 0.95},
                "address": {"value": "45 Park Street, Kolkata", "confidence": 0.90},
            },
        },
        "qr": {
            "detected": True,
            "decoded": True,
            "verified": False,
            "format": "xml",
            "fields": {
                "name": "ANANYA SHARMA",
                "dob": "20/06/1995",
                "gender": "FEMALE",
                "aadhaar_number": "XXXX XXXX 4321",
            },
        },
        "cross_validation": {
            "name": {"similarity": 1.0, "match": True},
            "dob": {"match": True, "comparison": "full"},
            "gender": {"match": True},
        },
        "warnings": [],
    }


def get_scenario_b_low_ocr_quality() -> Dict[str, Any]:
    """Scenario B: Low OCR confidence and noisy individual fields."""
    return {
        "success": True,
        "document": {"type": "aadhaar", "confidence": 0.88},
        "ocr": {
            "language": "eng",
            "confidence": 0.58,  # Triggers LOW_OCR_CONFIDENCE (15 pts)
            "fields": {
                "name": {"value": "ANANYA SHARMA", "confidence": 0.55},  # Low field conf (5 pts)
                "dob": {"year": 1995, "month": 6, "day": 20, "precision": "full", "confidence": 0.52},  # Low field conf (5 pts)
                "gender": {"value": "FEMALE", "confidence": 0.85},
                "aadhaar_number": {"value": "XXXX XXXX 4321", "confidence": 0.82},
                "address": {"value": "45 Park Street", "confidence": 0.70},
            },
        },
        "qr": {
            "detected": True,
            "decoded": True,
            "verified": False,
            "format": "xml",
            "fields": {
                "name": "ANANYA SHARMA",
                "dob": "20/06/1995",
                "gender": "FEMALE",
            },
        },
        "cross_validation": {
            "name": {"similarity": 1.0, "match": True},
            "dob": {"match": True, "comparison": "full"},
            "gender": {"match": True},
        },
        "warnings": [],
    }


def get_scenario_c_qr_not_detected() -> Dict[str, Any]:
    """Scenario C: Good OCR but QR code was not detected in the image."""
    return {
        "success": True,
        "document": {"type": "aadhaar", "confidence": 0.96},
        "ocr": {
            "language": "eng+hin",
            "confidence": 0.92,
            "fields": {
                "name": {"value": "VIKRAM SINGH", "confidence": 0.94},
                "dob": {"year": 1988, "month": 11, "day": 5, "precision": "full", "confidence": 0.91},
                "gender": {"value": "MALE", "confidence": 0.95},
                "aadhaar_number": {"value": "XXXX XXXX 9988", "confidence": 0.93},
                "address": {"value": "12 Sector 17, Chandigarh", "confidence": 0.88},
            },
        },
        "qr": {
            "detected": False,
            "decoded": False,
            "verified": False,
        },
        "warnings": ["No QR pattern located in image."],
    }


def get_scenario_d_qr_detected_not_decoded() -> Dict[str, Any]:
    """Scenario D: QR code was detected in the document but decoding failed."""
    return {
        "success": True,
        "document": {"type": "aadhaar", "confidence": 0.95},
        "ocr": {
            "language": "eng+hin",
            "confidence": 0.91,
            "fields": {
                "name": {"value": "VIKRAM SINGH", "confidence": 0.93},
                "dob": {"year": 1988, "month": 11, "day": 5, "precision": "full", "confidence": 0.90},
                "gender": {"value": "MALE", "confidence": 0.94},
                "aadhaar_number": {"value": "XXXX XXXX 9988", "confidence": 0.92},
                "address": {"value": "12 Sector 17, Chandigarh", "confidence": 0.85},
            },
        },
        "qr": {
            "detected": True,
            "decoded": False,
            "verified": False,
            "error": "QR_DECODE_FAILED",
        },
        "warnings": ["QR code detected but damaged; decode failed."],
    }


def get_scenario_e_name_mismatch() -> Dict[str, Any]:
    """Scenario E: OCR name disagrees with decoded QR name."""
    return {
        "success": True,
        "document": {"type": "aadhaar", "confidence": 0.97},
        "ocr": {
            "language": "eng",
            "confidence": 0.93,
            "fields": {
                "name": {"value": "SURESH MEHTA", "confidence": 0.95},
                "dob": {"year": 1990, "month": 3, "day": 15, "precision": "full", "confidence": 0.92},
                "gender": {"value": "MALE", "confidence": 0.96},
                "aadhaar_number": {"value": "XXXX XXXX 7766", "confidence": 0.94},
                "address": {"value": "MG Road, Pune", "confidence": 0.87},
            },
        },
        "qr": {
            "detected": True,
            "decoded": True,
            "verified": False,
            "format": "xml",
            "fields": {
                "name": "RAMESH CHANDRA",
                "dob": "15/03/1990",
                "gender": "MALE",
            },
        },
        "cross_validation": {
            "name": {"similarity": 0.35, "match": False},
            "dob": {"match": True, "comparison": "full"},
            "gender": {"match": True},
        },
        "warnings": ["Cross-validation observed name discrepancy."],
    }


def get_scenario_f_dob_mismatch() -> Dict[str, Any]:
    """Scenario F: OCR DOB disagrees with decoded QR DOB."""
    return {
        "success": True,
        "document": {"type": "aadhaar", "confidence": 0.97},
        "ocr": {
            "language": "eng",
            "confidence": 0.93,
            "fields": {
                "name": {"value": "SURESH MEHTA", "confidence": 0.95},
                "dob": {"year": 1990, "month": 3, "day": 15, "precision": "full", "confidence": 0.92},
                "gender": {"value": "MALE", "confidence": 0.96},
                "aadhaar_number": {"value": "XXXX XXXX 7766", "confidence": 0.94},
                "address": {"value": "MG Road, Pune", "confidence": 0.87},
            },
        },
        "qr": {
            "detected": True,
            "decoded": True,
            "verified": False,
            "format": "xml",
            "fields": {
                "name": "SURESH MEHTA",
                "dob": "25/12/1982",
                "gender": "MALE",
            },
        },
        "cross_validation": {
            "name": {"similarity": 1.0, "match": True},
            "dob": {"match": False, "comparison": "full"},
            "gender": {"match": True},
        },
        "warnings": ["Cross-validation observed DOB discrepancy."],
    }


def get_scenario_g_gender_mismatch() -> Dict[str, Any]:
    """Scenario G: OCR gender disagrees with decoded QR gender."""
    return {
        "success": True,
        "document": {"type": "aadhaar", "confidence": 0.97},
        "ocr": {
            "language": "eng",
            "confidence": 0.93,
            "fields": {
                "name": {"value": "SURESH MEHTA", "confidence": 0.95},
                "dob": {"year": 1990, "month": 3, "day": 15, "precision": "full", "confidence": 0.92},
                "gender": {"value": "MALE", "confidence": 0.96},
                "aadhaar_number": {"value": "XXXX XXXX 7766", "confidence": 0.94},
                "address": {"value": "MG Road, Pune", "confidence": 0.87},
            },
        },
        "qr": {
            "detected": True,
            "decoded": True,
            "verified": False,
            "format": "xml",
            "fields": {
                "name": "SURESH MEHTA",
                "dob": "15/03/1990",
                "gender": "FEMALE",
            },
        },
        "cross_validation": {
            "name": {"similarity": 1.0, "match": True},
            "dob": {"match": True, "comparison": "full"},
            "gender": {"match": False},
        },
        "warnings": ["Cross-validation observed gender discrepancy."],
    }


def get_scenario_h_non_aadhaar() -> Dict[str, Any]:
    """Scenario H: Uploaded file is not an Aadhaar card (e.g. utility bill / PAN / receipt)."""
    return {
        "success": True,
        "document": {"type": "not_aadhaar", "confidence": 0.85},
        "ocr": {
            "language": "eng",
            "confidence": 0.90,
            "fields": {
                "name": None,
                "dob": None,
                "gender": None,
                "aadhaar_number": None,
                "address": None,
            },
        },
        "qr": {
            "detected": False,
            "decoded": False,
            "verified": False,
        },
        "warnings": ["Document failed Aadhaar pattern recognition checks."],
    }


def get_scenario_i_ocr_failure() -> Dict[str, Any]:
    """Scenario I: Upstream document processing failed (corrupted or unreadable file)."""
    return {
        "success": False,
        "error": {
            "code": "CORRUPTED_OR_INVALID_FILE",
            "message": "The file cannot be read as a valid image or PDF.",
        },
        "warnings": ["Header validation failed on input stream."],
    }


ALL_SCENARIOS: List[Tuple[str, str, Any]] = [
    ("Scenario A", "Clean Aadhaar Document", get_scenario_a_clean_aadhaar),
    ("Scenario B", "Low OCR Quality", get_scenario_b_low_ocr_quality),
    ("Scenario C", "QR Code Not Detected", get_scenario_c_qr_not_detected),
    ("Scenario D", "QR Detected but Decode Failed", get_scenario_d_qr_detected_not_decoded),
    ("Scenario E", "Cross-Validation Name Mismatch", get_scenario_e_name_mismatch),
    ("Scenario F", "Cross-Validation DOB Mismatch", get_scenario_f_dob_mismatch),
    ("Scenario G", "Cross-Validation Gender Mismatch", get_scenario_g_gender_mismatch),
    ("Scenario H", "Non-Aadhaar Document", get_scenario_h_non_aadhaar),
    ("Scenario I", "Upstream OCR Failure", get_scenario_i_ocr_failure),
]


def run_all_scenarios() -> None:
    """Executes each scenario through assess_document and prints a comparison summary."""
    print("=" * 85)
    print(f"{'SCENARIO':<14} | {'NAME':<32} | {'SCORE':<6} | {'LEVEL':<8} | {'DECISION':<8}")
    print("=" * 85)

    for code, name, factory in ALL_SCENARIOS:
        payload = factory()
        res = assess_document(payload)
        risk = res.get("risk", {})
        score_val = risk.get("score")
        score_str = str(score_val) if score_val is not None else "N/A"
        level_str = str(risk.get("level", "UNKNOWN"))
        decision_str = str(risk.get("decision", "REVIEW"))
        flags_str = ", ".join(res.get("flags", [])) or "None"

        print(f"{code:<14} | {name:<32} | {score_str:<6} | {level_str:<8} | {decision_str:<8}")
        print(f"   Flags: {flags_str}")
        print("-" * 85)


if __name__ == "__main__":
    run_all_scenarios()
