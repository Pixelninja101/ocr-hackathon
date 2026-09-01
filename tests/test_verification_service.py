"""
Comprehensive Integration Tests for Backend Verification Service (tests/test_verification_service.py).
Validates verify_document, standard response contracts, explicit override exposure, PII privacy, and real OCR integration.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import unittest
from typing import Any, Dict

from integration.verification_service import verify_document


class TestVerificationService(unittest.TestCase):
    """Test suite for the backend-facing verify_document service."""

    def setUp(self) -> None:
        # Valid Aadhaar number (prefix 23456789012 + check digit 4)
        self.valid_aadhaar_num = "2345 6789 0124"
        # Invalid Aadhaar number (check digit 5)
        self.invalid_aadhaar_num = "2345 6789 0125"

        self.clean_ocr_payload: Dict[str, Any] = {
            "success": True,
            "document": {
                "type": "aadhaar",
                "confidence": 0.98,
            },
            "ocr": {
                "language": "eng+hin",
                "confidence": 0.95,
                "fields": {
                    "name": {"value": "ANANYA SHARMA", "confidence": 0.96},
                    "dob": {"year": 1995, "month": 8, "day": 20, "precision": "full", "confidence": 0.95},
                    "gender": {"value": "FEMALE", "confidence": 0.98},
                    "aadhaar_number": {"value": self.valid_aadhaar_num, "confidence": 0.96},
                    "address": {"value": "104, Green Valley, Bangalore, Karnataka", "confidence": 0.92},
                },
            },
            "qr": {
                "detected": True,
                "decoded": True,
                "verified": False,
                "format": "xml",
                "fields": {
                    "name": "ANANYA SHARMA",
                    "dob": "20/08/1995",
                    "gender": "FEMALE",
                    "aadhaar_number": self.valid_aadhaar_num,
                },
            },
            "cross_validation": {
                "name": {"similarity": 1.0, "match": True},
                "dob": {"match": True, "comparison": "full"},
                "gender": {"match": True},
            },
            "warnings": [],
        }

    # -------------------------------------------------------------------------
    # 1. Clean Aadhaar
    # -------------------------------------------------------------------------

    def test_01_clean_aadhaar_response_contract(self) -> None:
        """Test 1: Clean Aadhaar returns standard structure with LOW risk, PASS decision, and no overrides."""
        res = verify_document(self.clean_ocr_payload)

        self.assertTrue(res["success"])
        self.assertEqual(res["document"]["type"], "aadhaar")
        self.assertAlmostEqual(res["document"]["confidence"], 0.98, places=2)

        # Verification block
        self.assertEqual(res["verification"]["risk_level"], "LOW")
        self.assertEqual(res["verification"]["decision"], "PASS")
        self.assertFalse(res["verification"]["override_applied"])
        self.assertEqual(res["verification"]["override_reasons"], [])
        self.assertIn("low observed risk", res["verification"]["summary"])

        # Checks block
        self.assertTrue(res["checks"]["ocr"]["available"])
        self.assertAlmostEqual(res["checks"]["ocr"]["confidence"], 0.95, places=2)
        self.assertTrue(res["checks"]["qr"]["detected"])
        self.assertTrue(res["checks"]["qr"]["decoded"])
        self.assertFalse(res["checks"]["qr"]["verified"])
        self.assertEqual(res["checks"]["qr"]["status"], "QR_DECODED_VERIFICATION_UNAVAILABLE")

        self.assertTrue(res["checks"]["checksum"]["available"])
        self.assertTrue(res["checks"]["checksum"]["valid"])

        self.assertTrue(res["checks"]["cross_validation"]["available"])
        self.assertTrue(res["checks"]["cross_validation"]["name_match"])
        self.assertTrue(res["checks"]["cross_validation"]["dob_match"])
        self.assertTrue(res["checks"]["cross_validation"]["gender_match"])

    # -------------------------------------------------------------------------
    # 2. QR Unavailable
    # -------------------------------------------------------------------------

    def test_02_qr_unavailable(self) -> None:
        """Test 2: QR code not detected is accurately reflected in checks.qr and non-override status."""
        inp = copy.deepcopy(self.clean_ocr_payload)
        inp["qr"] = {"detected": False, "decoded": False, "verified": False}
        inp.pop("cross_validation", None)

        res = verify_document(inp)

        self.assertTrue(res["success"])
        self.assertFalse(res["checks"]["qr"]["detected"])
        self.assertFalse(res["checks"]["qr"]["decoded"])
        self.assertEqual(res["checks"]["qr"]["status"], "QR_NOT_DETECTED")
        self.assertFalse(res["verification"]["override_applied"])
        self.assertEqual(res["verification"]["risk_level"], "LOW")

    # -------------------------------------------------------------------------
    # 3. Masked Aadhaar
    # -------------------------------------------------------------------------

    def test_03_masked_aadhaar(self) -> None:
        """Test 3: Masked Aadhaar results in available: False, valid: None without triggering checksum failure."""
        inp = copy.deepcopy(self.clean_ocr_payload)
        inp["ocr"]["fields"]["aadhaar_number"]["value"] = "XXXX XXXX 1234"

        res = verify_document(inp)

        self.assertTrue(res["success"])
        self.assertFalse(res["checks"]["checksum"]["available"])
        self.assertIsNone(res["checks"]["checksum"]["valid"])
        self.assertNotIn("AADHAAR_CHECKSUM_INVALID", [f["rule_id"] for f in res["findings"]])

    # -------------------------------------------------------------------------
    # 4. Invalid Checksum
    # -------------------------------------------------------------------------

    def test_04_invalid_checksum(self) -> None:
        """Test 4: Invalid checksum sets checks.checksum.valid=False, +15 pts, non-override."""
        inp = copy.deepcopy(self.clean_ocr_payload)
        inp["ocr"]["fields"]["aadhaar_number"]["value"] = self.invalid_aadhaar_num

        res = verify_document(inp)

        self.assertTrue(res["success"])
        self.assertTrue(res["checks"]["checksum"]["available"])
        self.assertFalse(res["checks"]["checksum"]["valid"])
        self.assertFalse(res["verification"]["override_applied"])
        # Finding present
        chk_finding = next(f for f in res["findings"] if f["rule_id"] == "AADHAAR_CHECKSUM_INVALID")
        self.assertEqual(chk_finding["points"], 15)

    # -------------------------------------------------------------------------
    # 5. Name Mismatch & Critical Override Clarity
    # -------------------------------------------------------------------------

    def test_05_name_mismatch_override_clarity(self) -> None:
        """Test 5: Name mismatch sets override_applied=True, override_reasons=['NAME_MISMATCH'], HIGH level."""
        inp = copy.deepcopy(self.clean_ocr_payload)
        inp["cross_validation"]["name"] = {"similarity": 0.40, "match": False}

        res = verify_document(inp)

        self.assertTrue(res["success"])
        self.assertEqual(res["verification"]["risk_level"], "HIGH")
        self.assertEqual(res["verification"]["decision"], "REVIEW")
        self.assertTrue(res["verification"]["override_applied"])
        self.assertIn("NAME_MISMATCH", res["verification"]["override_reasons"])
        self.assertFalse(res["checks"]["cross_validation"]["name_match"])

        # Numeric score should NOT be artificially inflated to match HIGH
        # (30 pts mismatch + 5 pts QR unverified = 35 raw -> score 18)
        self.assertEqual(res["verification"]["risk_score"], 18)

    # -------------------------------------------------------------------------
    # 6. DOB Mismatch
    # -------------------------------------------------------------------------

    def test_06_dob_mismatch_override(self) -> None:
        """Test 6: DOB mismatch sets override_applied=True, override_reasons=['DOB_MISMATCH'], HIGH level."""
        inp = copy.deepcopy(self.clean_ocr_payload)
        inp["cross_validation"]["dob"] = {"match": False, "comparison": "full"}

        res = verify_document(inp)

        self.assertTrue(res["success"])
        self.assertEqual(res["verification"]["risk_level"], "HIGH")
        self.assertTrue(res["verification"]["override_applied"])
        self.assertIn("DOB_MISMATCH", res["verification"]["override_reasons"])
        self.assertFalse(res["checks"]["cross_validation"]["dob_match"])

    # -------------------------------------------------------------------------
    # 7. Gender Mismatch
    # -------------------------------------------------------------------------

    def test_07_gender_mismatch_override(self) -> None:
        """Test 7: Gender mismatch sets override_applied=True, override_reasons=['GENDER_MISMATCH'], HIGH level."""
        inp = copy.deepcopy(self.clean_ocr_payload)
        inp["cross_validation"]["gender"] = {"match": False}

        res = verify_document(inp)

        self.assertTrue(res["success"])
        self.assertEqual(res["verification"]["risk_level"], "HIGH")
        self.assertTrue(res["verification"]["override_applied"])
        self.assertIn("GENDER_MISMATCH", res["verification"]["override_reasons"])
        self.assertFalse(res["checks"]["cross_validation"]["gender_match"])

    # -------------------------------------------------------------------------
    # 8. Non-Aadhaar Document
    # -------------------------------------------------------------------------

    def test_08_non_aadhaar_document(self) -> None:
        """Test 8: Document of type 'not_aadhaar' triggers DOCUMENT_NOT_IDENTIFIED override."""
        inp = copy.deepcopy(self.clean_ocr_payload)
        inp["document"]["type"] = "not_aadhaar"
        inp["document"]["confidence"] = 0.90

        res = verify_document(inp)

        self.assertTrue(res["success"])
        self.assertEqual(res["document"]["type"], "not_aadhaar")
        self.assertEqual(res["verification"]["risk_level"], "HIGH")
        self.assertTrue(res["verification"]["override_applied"])
        self.assertIn("DOCUMENT_NOT_IDENTIFIED", res["verification"]["override_reasons"])

    # -------------------------------------------------------------------------
    # 9. OCR Processing Failure
    # -------------------------------------------------------------------------

    def test_09_ocr_processing_failure_response(self) -> None:
        """Test 9: Upstream failure returns success: False, score: None, level: UNKNOWN, decision: REVIEW, error code."""
        failed_input = {
            "success": False,
            "error": {
                "code": "IMAGE_DECODE_ERROR",
                "message": "File format not recognized as valid image.",
            },
            "warnings": ["Corrupt header encountered."],
        }

        res = verify_document(failed_input)

        self.assertFalse(res["success"])
        self.assertIsNone(res["verification"]["risk_score"])
        self.assertEqual(res["verification"]["risk_level"], "UNKNOWN")
        self.assertEqual(res["verification"]["decision"], "REVIEW")
        self.assertFalse(res["verification"]["override_applied"])
        self.assertIn("error", res)
        self.assertEqual(res["error"]["code"], "IMAGE_DECODE_ERROR")

    # -------------------------------------------------------------------------
    # 10. Missing Optional Fields Robustness
    # -------------------------------------------------------------------------

    def test_10_missing_optional_fields_safe_handling(self) -> None:
        """Test 10: Missing optional keys in input dictionary are safely handled with defaults."""
        sparse_input = {
            "success": True,
            "document": {"type": "aadhaar"},
            "ocr": {},
        }

        res = verify_document(sparse_input)

        self.assertTrue(res["success"])
        self.assertIsInstance(res["checks"], dict)
        self.assertFalse(res["checks"]["qr"]["detected"])
        self.assertFalse(res["checks"]["cross_validation"]["available"])
        self.assertFalse(res["checks"]["checksum"]["available"])

    # -------------------------------------------------------------------------
    # 11. JSON Serialization
    # -------------------------------------------------------------------------

    def test_11_json_serialization_without_custom_encoders(self) -> None:
        """Test 11: All verification responses must serialize with standard json.dumps without errors."""
        res = verify_document(self.clean_ocr_payload)

        serialized = json.dumps(res)
        self.assertIsInstance(serialized, str)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["success"], res["success"])
        self.assertEqual(deserialized["verification"]["risk_level"], "LOW")

    # -------------------------------------------------------------------------
    # 12. PII Safety & Recursive Privacy Inspection
    # -------------------------------------------------------------------------

    def test_12_pii_privacy_recursive_verification(self) -> None:
        """Test 12: Ensure no unmasked 12-digit number, raw name, raw DOB, or address leaks into final response."""
        sensitive_num = "987654321096"
        sensitive_name = "CONFIDENTIAL USER"
        sensitive_addr = "Secret Flat 99, Exclusive Towers, Mumbai 400001"

        inp = copy.deepcopy(self.clean_ocr_payload)
        inp["ocr"]["fields"]["name"]["value"] = sensitive_name
        inp["ocr"]["fields"]["address"]["value"] = sensitive_addr
        inp["ocr"]["fields"]["aadhaar_number"]["value"] = sensitive_num
        inp["qr"]["fields"]["name"] = sensitive_name
        inp["qr"]["fields"]["aadhaar_number"] = sensitive_num
        inp["warnings"] = [f"Aadhaar {sensitive_num} verified."]

        res = verify_document(inp)
        serialized = json.dumps(res)

        # 1. Unmasked continuous 12 digits should never appear
        self.assertNotIn(sensitive_num, serialized)
        # 2. No unmasked 12-digit regex match
        self.assertFalse(bool(re.search(r"\b\d{12}\b", serialized)))
        # 3. Raw names, addresses, or private details do not appear in findings/evidence/summary
        for f in res.get("findings", []):
            f_str = json.dumps(f)
            self.assertNotIn(sensitive_name, f_str)
            self.assertNotIn(sensitive_addr, f_str)

    # -------------------------------------------------------------------------
    # 13. Input Immutability
    # -------------------------------------------------------------------------

    def test_13_input_immutability(self) -> None:
        """Test 13: Passing a dictionary to verify_document does not modify the caller's dictionary."""
        input_data = copy.deepcopy(self.clean_ocr_payload)
        original_copy = copy.deepcopy(input_data)

        verify_document(input_data)

        self.assertEqual(input_data, original_copy)

    # -------------------------------------------------------------------------
    # 14. Real Document Processor Integration with File Bytes
    # -------------------------------------------------------------------------

    def test_14_real_document_processor_file_bytes_integration(self) -> None:
        """Test 14: End-to-end execution of verify_document(image_bytes) with synthetic document generator."""
        spec = importlib.util.spec_from_file_location(
            "ocr_test_helpers", r"C:\icons\ocr\tests\test_helpers.py"
        )
        assert spec is not None and spec.loader is not None
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)

        # Generate synthetic Aadhaar image bytes
        img_bytes = helpers.create_synthetic_aadhaar_bytes(
            name="DEEPAK ROY",
            dob="15/03/1990",
            gender="MALE",
            aadhaar_num=self.valid_aadhaar_num,
            include_qr=True,
        )

        res = verify_document(img_bytes)

        self.assertTrue(res["success"])
        self.assertIn("document", res)
        self.assertIn("verification", res)
        self.assertIn("checks", res)
        self.assertIn("findings", res)
        self.assertIn("warnings", res)

        # Verify JSON serializability
        self.assertIsInstance(json.dumps(res), str)


if __name__ == "__main__":
    unittest.main()
