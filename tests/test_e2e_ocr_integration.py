"""
End-to-end integration tests connecting real document_processor with risk_engine.
Tests multiple document conditions safely using in-memory synthetic fixtures.
"""

from __future__ import annotations

import importlib.util
import unittest
from typing import Any

import cv2
import numpy as np

from document_processor import process_document
from risk_engine import assess_document
from risk_engine.models import RiskDecision, RiskLevel


class TestE2EOCRIntegration(unittest.TestCase):
    """Verifies end-to-end processing across various synthetic document image conditions."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load synthetic generator from external OCR test helpers."""
        spec = importlib.util.spec_from_file_location(
            "ocr_test_helpers", r"C:\icons\ocr\tests\test_helpers.py"
        )
        assert spec is not None and spec.loader is not None
        cls.helpers: Any = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.helpers)

    def test_01_normal_document_condition(self) -> None:
        """Condition 1: Normal synthetic Aadhaar document processes cleanly end-to-end."""
        img_bytes = self.helpers.create_synthetic_aadhaar_bytes(
            name="ROHAN GUPTA",
            dob="10/05/1992",
            gender="MALE",
            aadhaar_num="1122 3344 5566",
            include_qr=True,
        )

        ocr_result = process_document(img_bytes)
        self.assertTrue(ocr_result["success"])

        assessment = assess_document(ocr_result)
        self.assertTrue(assessment["success"])
        self.assertIn("risk", assessment)
        self.assertIsInstance(assessment["risk"]["score"], int)
        self.assertIn(assessment["risk"]["level"], [RiskLevel.LOW.value, RiskLevel.MEDIUM.value])
        self.assertIn(assessment["risk"]["decision"], [RiskDecision.PASS.value, RiskDecision.REVIEW.value])

    def test_02_slightly_rotated_document_condition(self) -> None:
        """Condition 2: Slightly rotated (5 deg) document processes safely and identifies flags."""
        raw_img = self.helpers.create_synthetic_aadhaar_image(
            name="ROHAN GUPTA",
            dob="10/05/1992",
            gender="MALE",
            aadhaar_num="1122 3344 5566",
            include_qr=True,
        )
        h, w = raw_img.shape[:2]
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), 5, 1.0)
        rotated_img = cv2.warpAffine(raw_img, rot_mat, (w, h), borderValue=(255, 255, 255))
        _, buf = cv2.imencode(".png", rotated_img)

        ocr_result = process_document(buf.tobytes())
        self.assertTrue(ocr_result["success"])

        assessment = assess_document(ocr_result)
        self.assertTrue(assessment["success"])
        self.assertIn("risk", assessment)
        self.assertIn(assessment["risk"]["level"], [RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value])

    def test_03_low_quality_noisy_document_condition(self) -> None:
        """Condition 3: Low quality noisy image processes safely without unhandled errors."""
        raw_img = self.helpers.create_synthetic_aadhaar_image(
            name="ROHAN GUPTA",
            dob="10/05/1992",
            gender="MALE",
            aadhaar_num="1122 3344 5566",
            include_qr=True,
        )
        noise = np.random.normal(0, 35, raw_img.shape).astype(np.uint8)
        noisy_img = cv2.add(raw_img, noise)
        _, buf = cv2.imencode(".png", noisy_img)

        ocr_result = process_document(buf.tobytes())
        self.assertTrue(ocr_result["success"])

        assessment = assess_document(ocr_result)
        self.assertTrue(assessment["success"])
        self.assertIn("risk", assessment)

    def test_04_damaged_obscured_qr_condition(self) -> None:
        """Condition 4: Document with obscured QR code handles missing QR signals gracefully."""
        raw_img = self.helpers.create_synthetic_aadhaar_image(
            name="ROHAN GUPTA",
            dob="10/05/1992",
            gender="MALE",
            aadhaar_num="1122 3344 5566",
            include_qr=False,  # no QR
        )
        _, buf = cv2.imencode(".png", raw_img)

        ocr_result = process_document(buf.tobytes())
        self.assertTrue(ocr_result["success"])

        assessment = assess_document(ocr_result)
        self.assertTrue(assessment["success"])
        self.assertIn("QR_NOT_DETECTED", assessment["flags"])

    def test_05_non_aadhaar_document_condition(self) -> None:
        """Condition 5: Non-Aadhaar document receives HIGH risk and REVIEW decision."""
        non_doc = np.full((600, 800, 3), 255, dtype=np.uint8)
        cv2.putText(non_doc, "INVOICE #99881", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(non_doc, "Total Amount Due: $150.00", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        _, buf = cv2.imencode(".png", non_doc)

        ocr_result = process_document(buf.tobytes())
        self.assertTrue(ocr_result["success"])

        assessment = assess_document(ocr_result)
        self.assertTrue(assessment["success"])
        self.assertEqual(assessment["risk"]["level"], RiskLevel.HIGH.value)
        self.assertEqual(assessment["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertIn("DOCUMENT_NOT_IDENTIFIED", assessment["flags"])

    def test_06_corrupted_file_condition(self) -> None:
        """Condition 6: Corrupted/invalid file bytes return structured error without raising exception."""
        corrupt_bytes = b"CORRUPTED_FILE_DATA_HEADER_INVALID"

        ocr_result = process_document(corrupt_bytes)
        self.assertFalse(ocr_result["success"])

        assessment = assess_document(ocr_result)
        self.assertFalse(assessment["success"])
        self.assertIsNone(assessment["risk"]["score"])
        self.assertEqual(assessment["risk"]["level"], RiskLevel.UNKNOWN.value)
        self.assertEqual(assessment["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertIn("DOCUMENT_PROCESSING_FAILED", assessment["flags"])
        self.assertEqual(assessment["error"]["code"], "CORRUPTED_OR_INVALID_FILE")

    def test_07_oversized_file_condition(self) -> None:
        """Condition 7: Oversized file (>10MB) returns structured FILE_TOO_LARGE error."""
        oversized_bytes = b"A" * (11 * 1024 * 1024)

        ocr_result = process_document(oversized_bytes)
        self.assertFalse(ocr_result["success"])

        assessment = assess_document(ocr_result)
        self.assertFalse(assessment["success"])
        self.assertIsNone(assessment["risk"]["score"])
        self.assertEqual(assessment["risk"]["level"], RiskLevel.UNKNOWN.value)
        self.assertEqual(assessment["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertIn("DOCUMENT_PROCESSING_FAILED", assessment["flags"])
        self.assertEqual(assessment["error"]["code"], "FILE_TOO_LARGE")


if __name__ == "__main__":
    unittest.main()
