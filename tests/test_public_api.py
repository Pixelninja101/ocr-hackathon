"""
Integration tests specifically validating the primary public entry point:
`from document_processor import process_document`
"""

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Primary Public Entry Point
from document_processor import process_document
from tests.test_helpers import create_synthetic_aadhaar_bytes


class TestPublicAPI(unittest.TestCase):
    """
    Validates that callers only need `from document_processor import process_document`
    and can supply any supported input format safely.
    """

    # 1. Accepts raw bytes input
    def test_1_accepts_raw_bytes(self):
        png_bytes = create_synthetic_aadhaar_bytes(
            name="PRIYA SHARMA",
            dob="15/08/1995",
            gender="FEMALE",
            aadhaar_num="1234 5678 9012",
            include_qr=True,
        )
        mock_ocr = {
            "text": "Government of India\nPRIYA SHARMA\nDOB: 15/08/1995\nGender: FEMALE\n1234 5678 9012",
            "language": "eng+hin",
            "confidence": 0.93,
            "tokens": [],
            "warnings": [],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_ocr):
            res = process_document(png_bytes)

        self.assertTrue(res["success"])
        self.assertEqual(res["ocr"]["fields"]["name"]["value"], "PRIYA SHARMA")
        self.assertEqual(res["ocr"]["fields"]["gender"]["value"], "FEMALE")

    # 2. Accepts io.BytesIO buffer input
    def test_2_accepts_bytesio_buffer(self):
        png_bytes = create_synthetic_aadhaar_bytes(name="RAHUL KUMAR", include_qr=False)
        buffer = io.BytesIO(png_bytes)

        mock_ocr = {
            "text": "Government of India\nRAHUL KUMAR\nDOB: 12/04/2002\nGender: MALE",
            "language": "eng+hin",
            "confidence": 0.90,
            "tokens": [],
            "warnings": [],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_ocr):
            res = process_document(buffer)

        self.assertTrue(res["success"])
        self.assertEqual(res["ocr"]["fields"]["name"]["value"], "RAHUL KUMAR")

    # 3. Accepts file path string (str)
    def test_3_accepts_filepath_str(self):
        # Use existing test image if present or mock
        test_file = Path("test_document1.png")
        if test_file.exists():
            res = process_document(str(test_file))
            self.assertTrue(res["success"])
            self.assertEqual(res["document"]["type"], "aadhaar")
        else:
            self.skipTest("test_document1.png not found")

    # 4. Accepts pathlib.Path object
    def test_4_accepts_pathlib_path(self):
        test_file = Path("test_document1.png")
        if test_file.exists():
            res = process_document(test_file)
            self.assertTrue(res["success"])
            self.assertEqual(res["document"]["type"], "aadhaar")
        else:
            self.skipTest("test_document1.png not found")

    # 5. Success result is completely JSON serializable
    def test_5_result_json_serializability(self):
        png_bytes = create_synthetic_aadhaar_bytes(
            name="AMIT VERMA",
            dob="01/01/1988",
            gender="MALE",
            aadhaar_num="9876 5432 1098",
            include_qr=True,
        )
        mock_ocr = {
            "text": "Government of India\nAMIT VERMA\nDOB: 01/01/1988\nGender: MALE\n9876 5432 1098",
            "language": "eng+hin",
            "confidence": 0.95,
            "tokens": [],
            "warnings": [],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_ocr):
            res = process_document(png_bytes)

        self.assertTrue(res["success"])
        serialized = json.dumps(res)
        self.assertIsInstance(serialized, str)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["document"]["type"], "aadhaar")

    # 6. Non-existent file path returns structured error without crashing
    def test_6_file_not_found_structured_error(self):
        res = process_document("non_existent_file_path_12345.png")
        self.assertFalse(res["success"])
        self.assertEqual(res["error"]["code"], "FILE_NOT_FOUND")
        self.assertIn("message", res["error"])

    # 7. Empty input returns structured error
    def test_7_empty_input_structured_error(self):
        res = process_document(b"")
        self.assertFalse(res["success"])
        self.assertEqual(res["error"]["code"], "EMPTY_FILE")

    # 8. Corrupted input returns structured error
    def test_8_corrupted_input_structured_error(self):
        res = process_document(b"not an image or pdf")
        self.assertFalse(res["success"])
        self.assertEqual(res["error"]["code"], "CORRUPTED_OR_INVALID_FILE")

    # 9. Oversized input returns structured error
    def test_9_oversized_input_structured_error(self):
        oversized = b"%PDF-" + b"0" * (11 * 1024 * 1024)
        res = process_document(oversized)
        self.assertFalse(res["success"])
        self.assertEqual(res["error"]["code"], "FILE_TOO_LARGE")

    # 10. Privacy guarantees: Raw Aadhaar numbers are masked in all output fields
    def test_10_privacy_masking_in_public_result(self):
        png_bytes = create_synthetic_aadhaar_bytes(
            name="TEST USER",
            dob="10/10/2000",
            gender="MALE",
            aadhaar_num="4368 6864 3392",
            include_qr=True,
        )
        mock_ocr = {
            "text": "Government of India\nTEST USER\nDOB: 10/10/2000\nGender: MALE\n4368 6864 3392",
            "language": "eng+hin",
            "confidence": 0.95,
            "tokens": [],
            "warnings": [],
        }
        with patch("document_processor.processor.run_ocr", return_value=mock_ocr):
            res = process_document(png_bytes)

        self.assertTrue(res["success"])
        # OCR field masked
        self.assertEqual(res["ocr"]["fields"]["aadhaar_number"]["value"], "XXXX XXXX 3392")
        # QR field masked
        if res.get("qr", {}).get("fields", {}).get("aadhaar_number"):
            self.assertEqual(res["qr"]["fields"]["aadhaar_number"], "XXXX XXXX 3392")

        # Serialized check: Raw number 4368 6864 3392 or 436868643392 must never appear
        json_dump = json.dumps(res)
        self.assertNotIn("4368 6864 3392", json_dump)
        self.assertNotIn("436868643392", json_dump)


if __name__ == "__main__":
    unittest.main()
