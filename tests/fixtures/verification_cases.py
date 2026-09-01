"""
Deterministic synthetic verification test fixtures for demonstrations and regression testing.
All records use synthetic dummy data with zero real PII.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Valid Aadhaar number passing Verhoeff (Prefix "23456789012" -> Check digit 4)
SYNTHETIC_VALID_AADHAAR = "2345 6789 0124"
# Invalid Aadhaar number failing Verhoeff (Final digit mutated to 5)
SYNTHETIC_INVALID_AADHAAR = "2345 6789 0125"


# Case 1 — Clean Document (LOW risk, PASS)
CASE_CLEAN: Dict[str, Any] = {
    "name": "Clean Document",
    "description": "High-quality Aadhaar with matching OCR and QR, valid checksum.",
    "payload": {
        "success": True,
        "document": {
            "type": "aadhaar",
            "confidence": 0.98,
        },
        "ocr": {
            "language": "eng+hin",
            "confidence": 0.95,
            "fields": {
                "name": {"value": "ARJUN MEHTA", "confidence": 0.96},
                "dob": {"year": 1990, "month": 6, "day": 15, "precision": "full", "confidence": 0.94},
                "gender": {"value": "MALE", "confidence": 0.98},
                "aadhaar_number": {"value": SYNTHETIC_VALID_AADHAAR, "confidence": 0.96},
                "address": {"value": "Flat 101, Palm Grove, Mumbai", "confidence": 0.90},
            },
        },
        "qr": {
            "detected": True,
            "decoded": True,
            "verified": False,
            "format": "xml",
            "fields": {
                "name": "ARJUN MEHTA",
                "dob": "15/06/1990",
                "gender": "MALE",
                "aadhaar_number": SYNTHETIC_VALID_AADHAAR,
            },
        },
        "cross_validation": {
            "name": {"similarity": 1.0, "match": True},
            "dob": {"match": True, "comparison": "full"},
            "gender": {"match": True},
        },
        "warnings": [],
    },
    "expected_level": "LOW",
    "expected_decision": "PASS",
    "expected_override": False,
}

# Case 2 — Checksum Invalid (LOW risk, PASS, finding AADHAAR_CHECKSUM_INVALID)
CASE_INVALID_CHECKSUM: Dict[str, Any] = {
    "name": "Invalid Checksum",
    "description": "Aadhaar number fails Verhoeff checksum calculation (e.g. OCR single-digit error).",
    "payload": {
        "success": True,
        "document": {
            "type": "aadhaar",
            "confidence": 0.96,
        },
        "ocr": {
            "language": "eng+hin",
            "confidence": 0.92,
            "fields": {
                "name": {"value": "ARJUN MEHTA", "confidence": 0.95},
                "dob": {"year": 1990, "month": 6, "day": 15, "precision": "full", "confidence": 0.94},
                "gender": {"value": "MALE", "confidence": 0.98},
                "aadhaar_number": {"value": SYNTHETIC_INVALID_AADHAAR, "confidence": 0.91},
                "address": {"value": "Flat 101, Palm Grove, Mumbai", "confidence": 0.90},
            },
        },
        "qr": {
            "detected": True,
            "decoded": True,
            "verified": False,
            "format": "xml",
            "fields": {
                "name": "ARJUN MEHTA",
                "dob": "15/06/1990",
                "gender": "MALE",
            },
        },
        "cross_validation": {
            "name": {"similarity": 1.0, "match": True},
            "dob": {"match": True, "comparison": "full"},
            "gender": {"match": True},
        },
        "warnings": [],
    },
    "expected_level": "LOW",
    "expected_decision": "PASS",
    "expected_override": False,
}

# Case 3 — Name Mismatch (HIGH risk, REVIEW, override NAME_MISMATCH)
CASE_NAME_MISMATCH: Dict[str, Any] = {
    "name": "Name Mismatch",
    "description": "Visual OCR name does not match the encrypted/encoded QR name.",
    "payload": {
        "success": True,
        "document": {
            "type": "aadhaar",
            "confidence": 0.98,
        },
        "ocr": {
            "language": "eng+hin",
            "confidence": 0.95,
            "fields": {
                "name": {"value": "DEVENDRA SHARMA", "confidence": 0.96},
                "dob": {"year": 1990, "month": 6, "day": 15, "precision": "full", "confidence": 0.94},
                "gender": {"value": "MALE", "confidence": 0.98},
                "aadhaar_number": {"value": SYNTHETIC_VALID_AADHAAR, "confidence": 0.96},
                "address": {"value": "Flat 101, Palm Grove, Mumbai", "confidence": 0.90},
            },
        },
        "qr": {
            "detected": True,
            "decoded": True,
            "verified": False,
            "format": "xml",
            "fields": {
                "name": "ARJUN MEHTA",
                "dob": "15/06/1990",
                "gender": "MALE",
                "aadhaar_number": SYNTHETIC_VALID_AADHAAR,
            },
        },
        "cross_validation": {
            "name": {"similarity": 0.35, "match": False},
            "dob": {"match": True, "comparison": "full"},
            "gender": {"match": True},
        },
        "warnings": [],
    },
    "expected_level": "HIGH",
    "expected_decision": "REVIEW",
    "expected_override": True,
}

# Case 4 — DOB Mismatch (HIGH risk, REVIEW, override DOB_MISMATCH)
CASE_DOB_MISMATCH: Dict[str, Any] = {
    "name": "DOB Mismatch",
    "description": "Visual OCR date/year of birth disagrees with the QR code payload.",
    "payload": {
        "success": True,
        "document": {
            "type": "aadhaar",
            "confidence": 0.98,
        },
        "ocr": {
            "language": "eng+hin",
            "confidence": 0.95,
            "fields": {
                "name": {"value": "ARJUN MEHTA", "confidence": 0.96},
                "dob": {"year": 1982, "month": 1, "day": 1, "precision": "full", "confidence": 0.94},
                "gender": {"value": "MALE", "confidence": 0.98},
                "aadhaar_number": {"value": SYNTHETIC_VALID_AADHAAR, "confidence": 0.96},
                "address": {"value": "Flat 101, Palm Grove, Mumbai", "confidence": 0.90},
            },
        },
        "qr": {
            "detected": True,
            "decoded": True,
            "verified": False,
            "format": "xml",
            "fields": {
                "name": "ARJUN MEHTA",
                "dob": "15/06/1990",
                "gender": "MALE",
                "aadhaar_number": SYNTHETIC_VALID_AADHAAR,
            },
        },
        "cross_validation": {
            "name": {"similarity": 1.0, "match": True},
            "dob": {"match": False, "comparison": "full"},
            "gender": {"match": True},
        },
        "warnings": [],
    },
    "expected_level": "HIGH",
    "expected_decision": "REVIEW",
    "expected_override": True,
}

# Case 5 — QR Unavailable (LOW risk, PASS)
CASE_QR_UNAVAILABLE: Dict[str, Any] = {
    "name": "QR Not Detected",
    "description": "Document does not have a detectable QR barcode (e.g. cropped image).",
    "payload": {
        "success": True,
        "document": {
            "type": "aadhaar",
            "confidence": 0.95,
        },
        "ocr": {
            "language": "eng+hin",
            "confidence": 0.92,
            "fields": {
                "name": {"value": "ARJUN MEHTA", "confidence": 0.95},
                "dob": {"year": 1990, "month": 6, "day": 15, "precision": "full", "confidence": 0.94},
                "gender": {"value": "MALE", "confidence": 0.98},
                "aadhaar_number": {"value": SYNTHETIC_VALID_AADHAAR, "confidence": 0.96},
                "address": {"value": "Flat 101, Palm Grove, Mumbai", "confidence": 0.90},
            },
        },
        "qr": {
            "detected": False,
            "decoded": False,
            "verified": False,
        },
        "warnings": [],
    },
    "expected_level": "LOW",
    "expected_decision": "PASS",
    "expected_override": False,
}

# Case 6 — Non-Aadhaar Document (HIGH risk, REVIEW, override DOCUMENT_NOT_IDENTIFIED)
CASE_NON_AADHAAR: Dict[str, Any] = {
    "name": "Non-Aadhaar Document",
    "description": "Uploaded image is not an Aadhaar card (e.g. electricity bill or blank page).",
    "payload": {
        "success": True,
        "document": {
            "type": "not_aadhaar",
            "confidence": 0.85,
        },
        "ocr": {
            "language": "eng",
            "confidence": 0.80,
            "fields": {},
        },
        "qr": {
            "detected": False,
            "decoded": False,
            "verified": False,
        },
        "warnings": [],
    },
    "expected_level": "HIGH",
    "expected_decision": "REVIEW",
    "expected_override": True,
}

DEMO_CASES: List[Dict[str, Any]] = [
    CASE_CLEAN,
    CASE_INVALID_CHECKSUM,
    CASE_NAME_MISMATCH,
    CASE_DOB_MISMATCH,
    CASE_QR_UNAVAILABLE,
    CASE_NON_AADHAAR,
]
