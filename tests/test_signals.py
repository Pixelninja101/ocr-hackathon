"""
Unit & integration tests for Stage 2 Signal Extraction (risk_engine.signals).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from typing import Any, Dict

from risk_engine.models import QRVerificationStatus
from risk_engine.signals import extract_normalized_signals, extract_signals


class TestSignalExtraction(unittest.TestCase):
    """Test suite validating signal extraction and normalization adhering to all PRD rules."""

    def setUp(self) -> None:
        """Create baseline mock processing results."""
        self.complete_ocr_result: Dict[str, Any] = {
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

    def test_01_complete_successful_ocr_result(self) -> None:
        """Test 1: Complete successful OCR result extraction."""
        signals = extract_signals(self.complete_ocr_result)

        # Document signals
        self.assertEqual(signals["document"]["type"], "aadhaar")
        self.assertEqual(signals["document"]["confidence"], 0.99)

        # OCR signals
        self.assertEqual(signals["ocr"]["confidence"], 0.94)
        self.assertEqual(signals["ocr"]["language"], "eng+hin")
        self.assertTrue(signals["ocr"]["fields"]["name"]["available"])
        self.assertEqual(signals["ocr"]["fields"]["name"]["confidence"], 0.95)
        self.assertTrue(signals["ocr"]["fields"]["dob"]["available"])
        self.assertEqual(signals["ocr"]["fields"]["dob"]["precision"], "full")
        self.assertTrue(signals["ocr"]["fields"]["gender"]["available"])
        self.assertTrue(signals["ocr"]["fields"]["aadhaar_number"]["available"])
        self.assertTrue(signals["ocr"]["fields"]["address"]["available"])

        # QR signals
        self.assertTrue(signals["qr"]["detected"])
        self.assertTrue(signals["qr"]["decoded"])
        self.assertFalse(signals["qr"]["verified"])
        self.assertEqual(
            signals["qr"]["verification_status"],
            QRVerificationStatus.QR_DECODED_VERIFICATION_UNAVAILABLE.value,
        )

        # Cross-validation signals
        self.assertTrue(signals["cross_validation"]["available"])
        self.assertTrue(signals["cross_validation"]["name"]["available"])
        self.assertEqual(signals["cross_validation"]["name"]["similarity"], 1.0)
        self.assertTrue(signals["cross_validation"]["name"]["match"])
        self.assertTrue(signals["cross_validation"]["dob"]["match"])
        self.assertEqual(signals["cross_validation"]["dob"]["comparison"], "full")
        self.assertTrue(signals["cross_validation"]["gender"]["match"])

    def test_02_missing_cross_validation(self) -> None:
        """Test 2: Missing cross_validation must yield available=False without error."""
        input_data = copy.deepcopy(self.complete_ocr_result)
        input_data.pop("cross_validation", None)

        signals = extract_signals(input_data)

        self.assertFalse(signals["cross_validation"]["available"])
        self.assertFalse(signals["cross_validation"]["name"]["available"])
        self.assertIsNone(signals["cross_validation"]["name"]["similarity"])
        self.assertIsNone(signals["cross_validation"]["name"]["match"])
        self.assertFalse(signals["cross_validation"]["dob"]["available"])
        self.assertIsNone(signals["cross_validation"]["dob"]["match"])
        self.assertIsNone(signals["cross_validation"]["dob"]["comparison"])
        self.assertFalse(signals["cross_validation"]["gender"]["available"])
        self.assertIsNone(signals["cross_validation"]["gender"]["match"])

    def test_03_qr_detected_not_decoded(self) -> None:
        """Test 3: QR detected but decode failed -> QR_DETECTED_NOT_DECODED."""
        input_data = copy.deepcopy(self.complete_ocr_result)
        input_data.pop("cross_validation", None)
        input_data["qr"] = {
            "detected": True,
            "decoded": False,
            "verified": False,
            "error": "QR_DECODE_FAILED",
        }

        signals = extract_signals(input_data)

        self.assertTrue(signals["qr"]["detected"])
        self.assertFalse(signals["qr"]["decoded"])
        self.assertFalse(signals["qr"]["verified"])
        self.assertEqual(
            signals["qr"]["verification_status"],
            QRVerificationStatus.QR_DETECTED_NOT_DECODED.value,
        )

    def test_04_qr_decoded_verification_unavailable(self) -> None:
        """Test 4: QR decoded but verified=False must NOT become verification failed."""
        input_data = copy.deepcopy(self.complete_ocr_result)
        input_data["qr"] = {
            "detected": True,
            "decoded": True,
            "verified": False,
        }

        signals = extract_signals(input_data)

        self.assertEqual(
            signals["qr"]["verification_status"],
            QRVerificationStatus.QR_DECODED_VERIFICATION_UNAVAILABLE.value,
        )
        self.assertNotEqual(signals["qr"]["verification_status"], "QR_VERIFICATION_FAILED")
        self.assertNotEqual(signals["qr"]["verification_status"], "verification failed")

    def test_05_explicit_name_mismatch(self) -> None:
        """Test 5: Explicit name mismatch preserves similarity and match=False."""
        input_data = copy.deepcopy(self.complete_ocr_result)
        input_data["cross_validation"]["name"] = {
            "similarity": 0.42,
            "match": False,
        }

        signals = extract_signals(input_data)

        self.assertTrue(signals["cross_validation"]["name"]["available"])
        self.assertEqual(signals["cross_validation"]["name"]["similarity"], 0.42)
        self.assertIs(signals["cross_validation"]["name"]["match"], False)

    def test_06_name_similarity_without_match_field(self) -> None:
        """Test 6: Name similarity available but match field omitted -> derives match accurately."""
        input_data = copy.deepcopy(self.complete_ocr_result)
        input_data["cross_validation"]["name"] = {
            "similarity": 0.92,
        }

        signals = extract_signals(input_data)

        self.assertTrue(signals["cross_validation"]["name"]["available"])
        self.assertEqual(signals["cross_validation"]["name"]["similarity"], 0.92)
        self.assertIs(signals["cross_validation"]["name"]["match"], True)

        # Also test low similarity without match field
        input_data["cross_validation"]["name"] = {
            "similarity": 0.60,
        }
        signals_low = extract_signals(input_data)
        self.assertIs(signals_low["cross_validation"]["name"]["match"], False)

    def test_07_dob_full_precision_match(self) -> None:
        """Test 7: DOB full precision match."""
        input_data = copy.deepcopy(self.complete_ocr_result)
        input_data["cross_validation"]["dob"] = {
            "match": True,
            "comparison": "full",
        }

        signals = extract_signals(input_data)

        self.assertTrue(signals["cross_validation"]["dob"]["available"])
        self.assertIs(signals["cross_validation"]["dob"]["match"], True)
        self.assertEqual(signals["cross_validation"]["dob"]["comparison"], "full")

    def test_08_dob_year_only_match(self) -> None:
        """Test 8: DOB year-only match remains distinguishable from full comparison."""
        input_data = copy.deepcopy(self.complete_ocr_result)
        input_data["cross_validation"]["dob"] = {
            "match": True,
            "comparison": "year",
        }

        signals = extract_signals(input_data)

        self.assertTrue(signals["cross_validation"]["dob"]["available"])
        self.assertIs(signals["cross_validation"]["dob"]["match"], True)
        self.assertEqual(signals["cross_validation"]["dob"]["comparison"], "year")

    def test_09_explicit_dob_mismatch(self) -> None:
        """Test 9: Explicit DOB mismatch preserves match=False (distinct from None)."""
        input_data = copy.deepcopy(self.complete_ocr_result)
        input_data["cross_validation"]["dob"] = {
            "match": False,
            "comparison": "full",
        }

        signals = extract_signals(input_data)

        self.assertTrue(signals["cross_validation"]["dob"]["available"])
        self.assertIs(signals["cross_validation"]["dob"]["match"], False)
        self.assertEqual(signals["cross_validation"]["dob"]["comparison"], "full")

    def test_10_explicit_gender_mismatch(self) -> None:
        """Test 10: Explicit gender mismatch preserves match=False."""
        input_data = copy.deepcopy(self.complete_ocr_result)
        input_data["cross_validation"]["gender"] = {
            "match": False,
        }

        signals = extract_signals(input_data)

        self.assertTrue(signals["cross_validation"]["gender"]["available"])
        self.assertIs(signals["cross_validation"]["gender"]["match"], False)

    def test_11_missing_ocr_fields(self) -> None:
        """Test 11: Missing OCR fields gracefully extract as available=False and confidence=None."""
        input_data = copy.deepcopy(self.complete_ocr_result)
        input_data["ocr"]["fields"] = {
            "name": None,
            "dob": None,
            "gender": None,
            "aadhaar_number": None,
            "address": None,
        }

        signals = extract_signals(input_data)

        for field_name in ("name", "gender", "aadhaar_number", "address"):
            self.assertFalse(signals["ocr"]["fields"][field_name]["available"])
            self.assertIsNone(signals["ocr"]["fields"][field_name]["confidence"])

        self.assertFalse(signals["ocr"]["fields"]["dob"]["available"])
        self.assertIsNone(signals["ocr"]["fields"]["dob"]["confidence"])
        self.assertIsNone(signals["ocr"]["fields"]["dob"]["precision"])

    def test_12_failed_ocr_processing_result(self) -> None:
        """Test 12: Failed upstream processing result returns safe empty signals without error."""
        failed_result = {
            "success": False,
            "error": {
                "code": "CORRUPTED_OR_INVALID_FILE",
                "message": "File is corrupted.",
            },
            "warnings": ["Corrupted header found."],
        }

        signals = extract_signals(failed_result)

        self.assertEqual(signals["document"]["type"], "unknown")
        self.assertEqual(signals["document"]["confidence"], 0.0)
        self.assertIsNone(signals["ocr"]["confidence"])
        self.assertFalse(signals["qr"]["detected"])
        self.assertFalse(signals["cross_validation"]["available"])
        self.assertIn("Corrupted header found.", signals["warnings"])

    def test_13_malformed_processing_result(self) -> None:
        """Test 13: Malformed inputs (None, empty dict, primitive types) return structured signal dict."""
        malformed_cases = [
            None,
            {},
            {"document": "not_a_dict"},
            {"ocr": {"fields": "not_a_dict"}},
            {"qr": []},
            {"cross_validation": 123},
            "string_payload",
            [1, 2, 3],
        ]

        for item in malformed_cases:
            with self.subTest(item=item):
                signals = extract_signals(item)  # type: ignore[arg-type]
                self.assertIsInstance(signals, dict)
                self.assertIn("document", signals)
                self.assertIn("ocr", signals)
                self.assertIn("qr", signals)
                self.assertIn("cross_validation", signals)
                self.assertIn("warnings", signals)
                # Must be serializable to JSON
                json_str = json.dumps(signals)
                self.assertIsInstance(json_str, str)

    def test_14_pii_safety_unmasked_aadhaar(self) -> None:
        """Test 14: PII safety: synthetic 12-digit Aadhaar number never appears in output."""
        sensitive_result = copy.deepcopy(self.complete_ocr_result)
        # Place synthetic 12-digit number in OCR field and QR fields
        sensitive_result["ocr"]["fields"]["aadhaar_number"]["value"] = "987654321098"
        sensitive_result["qr"]["fields"]["aadhaar_number"] = "987654321098"
        sensitive_result["warnings"] = ["Checked ID 9876 5432 1098 in database."]

        signals = extract_signals(sensitive_result)
        serialized = json.dumps(signals)

        # Verify unmasked digits are not in serialized output
        self.assertNotIn("987654321098", serialized)
        # Verify aadhaar_number signal only has metadata (available, confidence)
        self.assertTrue(signals["ocr"]["fields"]["aadhaar_number"]["available"])
        self.assertEqual(signals["ocr"]["fields"]["aadhaar_number"]["confidence"], 0.95)
        self.assertNotIn("value", signals["ocr"]["fields"]["aadhaar_number"])

    def test_15_input_immutability(self) -> None:
        """Test 15: Signal extraction never modifies the original OCR result dictionary."""
        original = copy.deepcopy(self.complete_ocr_result)
        input_to_extract = copy.deepcopy(self.complete_ocr_result)

        extract_signals(input_to_extract)

        self.assertEqual(input_to_extract, original)

    def test_16_integration_with_actual_document_processor(self) -> None:
        """Test 16: Integration with actual document_processor.process_document()."""
        from document_processor import process_document

        # Load synthetic fixture generator from OCR package via importlib
        spec = importlib.util.spec_from_file_location(
            "ocr_test_helpers", r"C:\icons\ocr\tests\test_helpers.py"
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        ocr_helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ocr_helpers)

        # Generate synthetic Aadhaar in-memory bytes
        png_bytes = ocr_helpers.create_synthetic_aadhaar_bytes(
            name="PRIYA SHARMA",
            dob="25/12/1998",
            gender="FEMALE",
            aadhaar_num="9876 5432 1098",
            include_qr=True,
        )

        # 1. Upstream document processing
        ocr_result = process_document(png_bytes)
        self.assertTrue(ocr_result["success"])

        # 2. Downstream signal extraction
        signals = extract_signals(ocr_result)

        self.assertEqual(signals["document"]["type"], "aadhaar")
        self.assertGreater(signals["document"]["confidence"], 0.60)
        self.assertTrue(signals["qr"]["detected"])
        self.assertTrue(signals["qr"]["decoded"])
        self.assertEqual(
            signals["qr"]["verification_status"],
            QRVerificationStatus.QR_DECODED_VERIFICATION_UNAVAILABLE.value,
        )
        self.assertTrue(signals["cross_validation"]["available"])
        self.assertTrue(signals["cross_validation"]["name"]["match"])
        self.assertTrue(signals["cross_validation"]["dob"]["match"])

        # Verify JSON serializability
        serialized = json.dumps(signals)
        self.assertIsInstance(serialized, str)


if __name__ == "__main__":
    unittest.main()
