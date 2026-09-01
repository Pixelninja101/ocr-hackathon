"""
End-to-end PII and privacy verification tests for Risk Engine.
Ensures that no raw Aadhaar numbers, raw QR payloads, or sensitive PII leak into serialized assessment outputs.
"""

from __future__ import annotations

import copy
import json
import re
import unittest
from typing import Any, Dict

from risk_engine import assess_document


class TestEndToEndPrivacy(unittest.TestCase):
    """Verifies that complete serialized assessment output contains zero unprotected PII."""

    def setUp(self) -> None:
        """Create test payload with intentional sensitive values."""
        self.sensitive_aadhaar = "987654321098"
        self.sensitive_name = "CONFIDENTIAL PERSON NAME"
        self.sensitive_address = "Flat 402, Highly Secret Colony, City-560001"
        self.raw_qr_payload = f"<PrintLetterBarcodeData uid='{self.sensitive_aadhaar}' name='{self.sensitive_name}' dob='01/01/1990' />"

        self.sensitive_payload: Dict[str, Any] = {
            "success": True,
            "document": {"type": "aadhaar", "confidence": 0.99},
            "ocr": {
                "language": "eng",
                "confidence": 0.95,
                "fields": {
                    "name": {"value": self.sensitive_name, "confidence": 0.96},
                    "dob": {"year": 1990, "month": 1, "day": 1, "precision": "full", "confidence": 0.94},
                    "gender": {"value": "MALE", "confidence": 0.97},
                    "aadhaar_number": {"value": self.sensitive_aadhaar, "confidence": 0.95},
                    "address": {"value": self.sensitive_address, "confidence": 0.90},
                },
            },
            "qr": {
                "detected": True,
                "decoded": True,
                "verified": False,
                "raw_payload": self.raw_qr_payload,
                "fields": {
                    "name": self.sensitive_name,
                    "dob": "01/01/1990",
                    "gender": "MALE",
                    "aadhaar_number": self.sensitive_aadhaar,
                },
            },
            "cross_validation": {
                "name": {"similarity": 1.0, "match": True},
                "dob": {"match": True, "comparison": "full"},
                "gender": {"match": True},
            },
            "warnings": [f"Scanned UID {self.sensitive_aadhaar} against local index."],
        }

    def test_01_raw_aadhaar_number_never_leaks_in_json(self) -> None:
        """Test 1: Unmasked 12-digit UID never appears anywhere in the serialized JSON output."""
        result = assess_document(self.sensitive_payload)
        serialized = json.dumps(result)

        self.assertNotIn(self.sensitive_aadhaar, serialized)
        # Check that no other 12-digit number sequence exists
        twelve_digit_pattern = r"\b\d{12}\b"
        self.assertFalse(
            bool(re.search(twelve_digit_pattern, serialized)),
            f"Found unmasked 12-digit number in serialized output: {serialized}",
        )

    def test_02_raw_qr_payload_never_leaks_in_json(self) -> None:
        """Test 2: Raw QR barcode payload structure is excluded from assessment output."""
        result = assess_document(self.sensitive_payload)
        serialized = json.dumps(result)

        self.assertNotIn("<PrintLetterBarcodeData", serialized)
        self.assertNotIn("raw_payload", serialized)

    def test_03_finding_evidence_excludes_raw_pii(self) -> None:
        """Test 3: Finding reasons and evidence contain only metadata and scores, never raw PII strings."""
        mismatch_payload = copy.deepcopy(self.sensitive_payload)
        mismatch_payload["cross_validation"]["name"] = {"similarity": 0.2, "match": False}
        mismatch_payload["cross_validation"]["dob"] = {"match": False, "comparison": "full"}
        mismatch_payload["cross_validation"]["gender"] = {"match": False}

        result = assess_document(mismatch_payload)
        findings = result.get("findings", [])

        for finding in findings:
            evidence_str = json.dumps(finding.get("evidence", {}))
            reason_str = finding.get("reason", "")

            self.assertNotIn(self.sensitive_aadhaar, evidence_str)
            self.assertNotIn(self.sensitive_name, evidence_str)
            self.assertNotIn(self.sensitive_address, evidence_str)

            self.assertNotIn(self.sensitive_aadhaar, reason_str)
            self.assertNotIn(self.sensitive_name, reason_str)
            self.assertNotIn(self.sensitive_address, reason_str)

    def test_04_input_immutability_under_sensitive_processing(self) -> None:
        """Test 4: assess_document does not mutate the sensitive input dictionary in place."""
        original = copy.deepcopy(self.sensitive_payload)
        input_copy = copy.deepcopy(self.sensitive_payload)

        assess_document(input_copy)

        self.assertEqual(input_copy, original)


if __name__ == "__main__":
    unittest.main()
