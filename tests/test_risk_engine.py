"""
Unit tests for Risk Engine public API (assess_document).
"""

from __future__ import annotations

import copy
import json
import unittest

from risk_engine import assess_document


class TestRiskEnginePublicAPI(unittest.TestCase):
    """Test suite covering the public API contract and boundary cases of assess_document."""

    def setUp(self) -> None:
        """Set up standard mock fixtures based on actual OCR output contract."""
        self.valid_result = {
            "success": True,
            "document": {
                "type": "aadhaar",
                "confidence": 0.99,
            },
            "ocr": {
                "language": "eng+hin",
                "confidence": 0.94,
                "fields": {
                    "name": {
                        "value": "RAHUL KUMAR",
                        "confidence": 0.95,
                    },
                    "dob": {
                        "year": 2002,
                        "month": 4,
                        "day": 12,
                        "precision": "full",
                        "confidence": 0.90,
                    },
                    "gender": {
                        "value": "MALE",
                        "confidence": 0.96,
                    },
                    "aadhaar_number": {
                        "value": "XXXX XXXX 1098",
                        "confidence": 0.95,
                    },
                    "address": {
                        "value": "123 MG Road, Bengaluru",
                        "confidence": 0.85,
                    },
                },
            },
            "qr": {
                "detected": True,
                "decoded": True,
                "verified": False,
                "format": "xml",
                "fields": {
                    "name": "RAHUL KUMAR",
                    "dob": "12/04/2002",
                    "gender": "MALE",
                    "aadhaar_number": "XXXX XXXX 1098",
                    "masked_aadhaar": "XXXX XXXX 1098",
                },
            },
            "cross_validation": {
                "name": {
                    "similarity": 1.0,
                    "match": True,
                },
                "dob": {
                    "match": True,
                    "comparison": "full",
                },
                "gender": {
                    "match": True,
                },
            },
            "warnings": [],
        }

        self.failed_result = {
            "success": False,
            "error": {
                "code": "FILE_TOO_LARGE",
                "message": "File exceeds maximum allowable size of 10MB.",
            },
            "warnings": ["Upload rejected during initial validation."],
        }

    def test_valid_processing_result(self) -> None:
        """Test assessment on a completely valid OCR and QR processing result."""
        result = assess_document(self.valid_result)

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("risk", result)
        self.assertEqual(result["risk"]["level"], "LOW")
        self.assertEqual(result["risk"]["decision"], "PASS")
        self.assertIn("signals", result)
        self.assertEqual(result["signals"]["document"]["type"], "aadhaar")
        self.assertEqual(result["signals"]["document"]["confidence"], 0.99)
        self.assertEqual(result["signals"]["ocr"]["confidence"], 0.94)
        self.assertTrue(result["signals"]["qr"]["detected"])
        self.assertTrue(result["signals"]["qr"]["decoded"])
        self.assertFalse(result["signals"]["qr"]["verified"])
        self.assertTrue(result["signals"]["cross_validation"]["available"])
        self.assertIsNone(result["error"])

    def test_failed_processing_result(self) -> None:
        """Test safe handling when upstream document processing returned success=False."""
        result = assess_document(self.failed_result)

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "PROCESSING_FAILED")
        self.assertIn("risk", result)
        self.assertIsNone(result["risk"]["score"])
        self.assertEqual(result["risk"]["level"], "UNKNOWN")
        self.assertEqual(result["risk"]["decision"], "REVIEW")
        self.assertIn("DOCUMENT_PROCESSING_FAILED", result["flags"])
        self.assertIsNotNone(result["error"])
        self.assertEqual(result["error"]["code"], "FILE_TOO_LARGE")
        self.assertIn("Upload rejected during initial validation.", result["warnings"])

    def test_missing_cross_validation(self) -> None:
        """Test processing result where cross_validation key is completely omitted (e.g. no QR)."""
        input_data = copy.deepcopy(self.valid_result)
        input_data.pop("cross_validation", None)
        input_data["qr"] = {
            "detected": False,
            "decoded": False,
            "verified": False,
        }

        result = assess_document(input_data)

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result["signals"]["cross_validation"]["available"])
        self.assertFalse(result["signals"]["qr"]["detected"])

    def test_missing_qr_fields(self) -> None:
        """Test processing result where QR was detected but decode failed or fields are missing."""
        input_data = copy.deepcopy(self.valid_result)
        input_data.pop("cross_validation", None)
        input_data["qr"] = {
            "detected": True,
            "decoded": False,
            "verified": False,
            "error": "QR_DECODE_FAILED",
        }

        result = assess_document(input_data)

        self.assertTrue(result["success"])
        self.assertTrue(result["signals"]["qr"]["detected"])
        self.assertFalse(result["signals"]["qr"]["decoded"])
        self.assertFalse(result["signals"]["cross_validation"]["available"])

    def test_missing_ocr_fields(self) -> None:
        """Test processing result where all or some OCR fields are None."""
        input_data = copy.deepcopy(self.valid_result)
        input_data["ocr"]["fields"] = {
            "name": None,
            "dob": None,
            "gender": None,
            "aadhaar_number": None,
            "address": None,
        }

        result = assess_document(input_data)

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result["signals"]["ocr"]["fields"]["name"]["available"])
        self.assertFalse(result["signals"]["ocr"]["fields"]["dob"]["available"])

    def test_malformed_input(self) -> None:
        """Test handling of unexpected types or incomplete dictionaries."""
        malformed_inputs = [
            {},
            {"random_key": 123},
            {"success": "not_a_bool"},
            {"document": "not_a_dict"},
            [],
            "string_input",
            12345,
        ]

        for item in malformed_inputs:
            with self.subTest(item=item):
                result = assess_document(item)  # type: ignore[arg-type]
                self.assertIsInstance(result, dict)
                # Must not raise an exception, must return JSON-serializable dict
                json_str = json.dumps(result)
                self.assertIsInstance(json_str, str)

    def test_none_input(self) -> None:
        """Test graceful rejection when input is None."""
        result = assess_document(None)

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "INVALID_INPUT")
        self.assertIn("risk", result)
        self.assertIsNone(result["risk"]["score"])
        self.assertEqual(result["risk"]["level"], "UNKNOWN")
        self.assertEqual(result["risk"]["decision"], "REVIEW")
        self.assertIn("INVALID_INPUT", result["flags"])
        self.assertEqual(result["error"]["code"], "INVALID_INPUT")

    def test_input_immutability(self) -> None:
        """Verify that the input dictionary is never mutated by assess_document."""
        original_input = copy.deepcopy(self.valid_result)
        input_to_pass = copy.deepcopy(self.valid_result)

        assess_document(input_to_pass)

        self.assertEqual(input_to_pass, original_input)

    def test_json_serializability(self) -> None:
        """Verify that the output of assess_document is strictly JSON-serializable."""
        result = assess_document(self.valid_result)
        try:
            serialized = json.dumps(result)
            deserialized = json.loads(serialized)
            self.assertEqual(deserialized["status"], "SUCCESS")
            self.assertEqual(deserialized["risk"]["decision"], "PASS")
        except TypeError as err:
            self.fail(f"assess_document result is not JSON-serializable: {err}")

    def test_no_raw_aadhaar_pii_leakage(self) -> None:
        """Ensure no unmasked 12-digit numbers are exposed in the result output."""
        sensitive_input = copy.deepcopy(self.valid_result)
        # Even if unmasked numbers somehow exist in raw warnings or error message:
        sensitive_input["warnings"] = ["Processing check with raw UID 9876 5432 1098"]

        result = assess_document(sensitive_input)
        serialized = json.dumps(result)

        # Raw continuous 12 digits should never appear
        self.assertNotIn("987654321098", serialized)


if __name__ == "__main__":
    unittest.main()
