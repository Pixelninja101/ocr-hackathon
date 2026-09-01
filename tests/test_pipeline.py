"""
Comprehensive test suite strictly covering the 10 PRD Section 29 test scenarios.
"""

import unittest
from unittest.mock import patch
import numpy as np

from document_processor.processor import process_document
from tests.test_helpers import (
    create_synthetic_aadhaar_bytes,
    create_synthetic_aadhaar_pdf,
)


class TestPRDScenarios(unittest.TestCase):

    def test_prd_scenario_1_clear_aadhaar(self):
        """Test 1 — Clear Aadhaar: Aadhaar detected, OCR succeeds, QR detected, QR decodes, fields extracted."""
        png_bytes = create_synthetic_aadhaar_bytes(
            name="RAHUL KUMAR",
            dob="12/04/2002",
            gender="MALE",
            aadhaar_num="9876 5432 1098",
        )
        mock_ocr = {
            "text": "Government of India\nUnique Identification Authority of India\nRAHUL KUMAR\nDOB: 12/04/2002\nGender: MALE\n9876 5432 1098",
            "language": "eng+hin",
            "confidence": 0.96,
            "tokens": [],
            "warnings": [],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_ocr):
            result = process_document(png_bytes)

        self.assertTrue(result["success"])
        self.assertEqual(result["document"]["type"], "aadhaar")
        self.assertGreaterEqual(result["document"]["confidence"], 0.80)
        self.assertTrue(result["qr"]["detected"])
        self.assertTrue(result["qr"]["decoded"])
        self.assertEqual(result["ocr"]["fields"]["name"]["value"], "RAHUL KUMAR")
        self.assertEqual(result["ocr"]["fields"]["dob"]["year"], 2002)
        self.assertEqual(result["ocr"]["fields"]["gender"]["value"], "MALE")

    def test_prd_scenario_2_low_quality_photo(self):
        """Test 2 — Low-quality photograph: Processing completes, confidence decreases, warnings handled, does not crash."""
        png_bytes = create_synthetic_aadhaar_bytes(name="RAHUL KUMAR")
        mock_ocr_low_conf = {
            "text": "Govt of Ind\nRAHUL KUMAR\nDOB 12/04/2002",
            "language": "eng+hin",
            "confidence": 0.42,
            "tokens": [],
            "warnings": ["Low resolution image detected"],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_ocr_low_conf):
            result = process_document(png_bytes)

        self.assertTrue(result["success"])
        self.assertEqual(result["ocr"]["confidence"], 0.42)
        self.assertIn("warnings", result)

    def test_prd_scenario_3_hindi_and_english(self):
        """Test 3 — Hindi + English: eng+hin OCR runs."""
        png_bytes = create_synthetic_aadhaar_bytes()
        mock_bilingual_ocr = {
            "text": "भारत सरकार\nGovernment of India\nराहुल कुमार\nRAHUL KUMAR\nजन्म तिथि / DOB: 12/04/2002\nलिंग / Gender: पुरुष / MALE",
            "language": "eng+hin",
            "confidence": 0.94,
            "tokens": [],
            "warnings": [],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_bilingual_ocr):
            result = process_document(png_bytes)

        self.assertTrue(result["success"])
        self.assertEqual(result["ocr"]["language"], "eng+hin")
        self.assertEqual(result["ocr"]["fields"]["name"]["value"], "RAHUL KUMAR")
        self.assertEqual(result["ocr"]["fields"]["gender"]["value"], "MALE")

    def test_prd_scenario_4_ocr_typo_fuzzy_match(self):
        """Test 4 — OCR typo: OCR 'RAHUI KUMAR' vs QR 'RAHUL KUMAR' -> High fuzzy similarity -> Likely match."""
        png_bytes = create_synthetic_aadhaar_bytes(
            name="RAHUL KUMAR",
            dob="12/04/2002",
            gender="MALE",
        )
        mock_typo_ocr = {
            "text": "Government of India\nRAHUI KUMAR\nDOB: 12/04/2002\nGender: MALE",
            "language": "eng+hin",
            "confidence": 0.88,
            "tokens": [],
            "warnings": [],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_typo_ocr):
            result = process_document(png_bytes)

        self.assertTrue(result["success"])
        self.assertIn("cross_validation", result)
        self.assertTrue(result["cross_validation"]["name"]["match"])
        self.assertGreaterEqual(result["cross_validation"]["name"]["similarity"], 0.85)

    def test_prd_scenario_5_year_only_dob(self):
        """Test 5 — Year-only DOB: OCR: 2002 vs QR: 12/04/2002 -> Result: MATCH (comparison: 'year')."""
        png_bytes = create_synthetic_aadhaar_bytes(
            name="RAHUL KUMAR",
            dob="12/04/2002",
            gender="MALE",
        )
        mock_yob_ocr = {
            "text": "Government of India\nRAHUL KUMAR\nYear of Birth: 2002\nGender: MALE",
            "language": "eng+hin",
            "confidence": 0.92,
            "tokens": [],
            "warnings": [],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_yob_ocr):
            result = process_document(png_bytes)

        self.assertTrue(result["success"])
        self.assertEqual(result["ocr"]["fields"]["dob"]["precision"], "year")
        self.assertEqual(result["ocr"]["fields"]["dob"]["year"], 2002)
        self.assertTrue(result["cross_validation"]["dob"]["match"])
        self.assertEqual(result["cross_validation"]["dob"]["comparison"], "year")

    def test_prd_scenario_6_qr_mismatch(self):
        """Test 6 — QR mismatch: OCR name 'RAHUL KUMAR' vs QR name 'AMIT KUMAR' -> Name mismatch."""
        png_bytes = create_synthetic_aadhaar_bytes(
            name="AMIT KUMAR",  # QR will have AMIT KUMAR
            dob="12/04/2002",
            gender="MALE",
        )
        mock_mismatch_ocr = {
            "text": "Government of India\nRAHUL KUMAR\nDOB: 12/04/2002\nGender: MALE",
            "language": "eng+hin",
            "confidence": 0.94,
            "tokens": [],
            "warnings": [],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_mismatch_ocr):
            result = process_document(png_bytes)

        self.assertTrue(result["success"])
        self.assertIn("cross_validation", result)
        self.assertFalse(result["cross_validation"]["name"]["match"])
        self.assertLess(result["cross_validation"]["name"]["similarity"], 0.85)

    def test_prd_scenario_7_no_qr(self):
        """Test 7 — No QR: detected: false, decoded: false, pipeline continues instead of crashing."""
        png_no_qr = create_synthetic_aadhaar_bytes(include_qr=False)
        mock_ocr = {
            "text": "Government of India\nUnique Identification Authority of India\nRAHUL KUMAR\nDOB: 12/04/2002\nGender: MALE",
            "language": "eng+hin",
            "confidence": 0.92,
            "tokens": [],
            "warnings": [],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_ocr):
            result = process_document(png_no_qr)

        self.assertTrue(result["success"])
        self.assertFalse(result["qr"]["detected"])
        self.assertFalse(result["qr"]["decoded"])
        self.assertNotIn("cross_validation", result)

    def test_prd_scenario_8_unsupported_file(self):
        """Test 8 — Unsupported file: structured error."""
        # Random non-supported file buffer
        result = process_document(b"PK\x03\x04zipfilecontents")
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_prd_scenario_9_corrupted_image(self):
        """Test 9 — Corrupted image: structured error."""
        result = process_document(b"this is corrupt data not an image")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["code"], "CORRUPTED_OR_INVALID_FILE")

    def test_prd_scenario_10_oversized_file(self):
        """Test 10 — Oversized file: structured error."""
        oversized = b"%PDF-" + b"x" * (11 * 1024 * 1024)
        result = process_document(oversized)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["code"], "FILE_TOO_LARGE")

    def test_prd_scenario_11_json_serialization_and_privacy_masking(self):
        """Test 11 — JSON Serialization & Privacy: Output is JSON-serializable and Aadhaar is masked."""
        import json
        import re

        png_bytes = create_synthetic_aadhaar_bytes(
            name="RAHUL KUMAR",
            dob="12/04/2002",
            gender="MALE",
            aadhaar_num="9876 5432 1098",
            include_qr=True,
        )
        mock_ocr = {
            "text": "Government of India\nRAHUL KUMAR\nDOB: 12/04/2002\nGender: MALE\n9876 5432 1098",
            "language": "eng+hin",
            "confidence": 0.95,
            "tokens": [],
            "warnings": [],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_ocr):
            result = process_document(png_bytes)

        self.assertTrue(result["success"])
        # Must be JSON serializable
        json_output = json.dumps(result)
        self.assertIsInstance(json_output, str)

        # Aadhaar number must be masked in ocr fields
        self.assertEqual(result["ocr"]["fields"]["aadhaar_number"]["value"], "XXXX XXXX 1098")
        # No raw unmasked 12-digit number (987654321098 or 9876 5432 1098) in the serialized JSON
        self.assertNotIn("9876 5432 1098", json_output)
        self.assertNotIn("987654321098", json_output)

    def test_prd_scenario_12_non_aadhaar_document(self):
        """Test 12 — Non-Aadhaar document: Completes gracefully with unknown/non-aadhaar classification."""
        import cv2
        blank_img = np.full((300, 400, 3), 255, dtype=np.uint8)
        _, img_bytes = cv2.imencode(".png", blank_img)
        mock_non_aadhaar_ocr = {
            "text": "ELECTRICITY BILL\nAccount Number: 123456\nAmount Due: Rs 500\nDue Date: 30/11/2024",
            "language": "eng+hin",
            "confidence": 0.90,
            "tokens": [],
            "warnings": [],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_non_aadhaar_ocr), \
             patch("document_processor.processor.process_qr_code", return_value={"detected": False, "decoded": False, "verified": False}):
            result = process_document(img_bytes.tobytes())

        self.assertTrue(result["success"])
        self.assertEqual(result["document"]["type"], "unknown")
        self.assertLess(result["document"]["confidence"], 0.50)

    def test_prd_scenario_13_unreadable_qr_graceful_continuation(self):
        """Test 13 — Unreadable QR: Detected but decoding fails -> Pipeline continues without crash."""
        png_bytes = create_synthetic_aadhaar_bytes(name="RAHUL KUMAR", include_qr=False)
        mock_ocr = {
            "text": "Government of India\nRAHUL KUMAR\nDOB: 12/04/2002\nGender: MALE",
            "language": "eng+hin",
            "confidence": 0.92,
            "tokens": [],
            "warnings": [],
        }
        # Simulate blurry QR detected but not decodable
        mock_blurry_qr = {
            "detected": True,
            "decoded": False,
            "verified": False,
            "error": "QR_UNREADABLE",
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_ocr), \
             patch("document_processor.processor.process_qr_code", return_value=mock_blurry_qr):
            result = process_document(png_bytes)

        self.assertTrue(result["success"])
        self.assertTrue(result["qr"]["detected"])
        self.assertFalse(result["qr"]["decoded"])
        self.assertEqual(result["ocr"]["fields"]["name"]["value"], "RAHUL KUMAR")
        self.assertNotIn("cross_validation", result)

    def test_prd_scenario_14_multiline_address_in_pipeline_output(self):
        """Test 14 — Multiline Address: Address field properly extracted and included in pipeline result."""
        png_bytes = create_synthetic_aadhaar_bytes(include_qr=False)
        mock_ocr = {
            "text": "Address:\nC/O: Ramesh Kumar, 123 MG Road, Thatipur, Gwalior, Madhya Pradesh - 474011",
            "language": "eng+hin",
            "confidence": 0.90,
            "tokens": [],
            "warnings": [],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_ocr), \
             patch("document_processor.processor.process_qr_code", return_value={"detected": False, "decoded": False, "verified": False}):
            result = process_document(png_bytes)

        self.assertTrue(result["success"])
        self.assertIsNotNone(result["ocr"]["fields"]["address"])
        self.assertIn("123 MG Road", result["ocr"]["fields"]["address"]["value"])


if __name__ == "__main__":
    unittest.main()

