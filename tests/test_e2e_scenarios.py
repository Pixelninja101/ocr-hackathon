"""
Automated end-to-end scenario tests for Risk Engine (Scenarios A through I).
Validates risk levels, decision mappings, and exact rule ID flags.
"""

from __future__ import annotations

import unittest

from demo.scenarios import (
    get_scenario_a_clean_aadhaar,
    get_scenario_b_low_ocr_quality,
    get_scenario_c_qr_not_detected,
    get_scenario_d_qr_detected_not_decoded,
    get_scenario_e_name_mismatch,
    get_scenario_f_dob_mismatch,
    get_scenario_g_gender_mismatch,
    get_scenario_h_non_aadhaar,
    get_scenario_i_ocr_failure,
)
from risk_engine import assess_document
from risk_engine.models import RiskDecision, RiskLevel


class TestEndToEndScenarios(unittest.TestCase):
    """Verifies that all standard synthetic risk scenarios produce expected scores, tiers, and flags."""

    def test_scenario_a_clean_aadhaar(self) -> None:
        """Scenario A: Clean Aadhaar -> LOW risk, PASS decision."""
        result = assess_document(get_scenario_a_clean_aadhaar())

        self.assertTrue(result["success"])
        self.assertEqual(result["risk"]["level"], RiskLevel.LOW.value)
        self.assertEqual(result["risk"]["decision"], RiskDecision.PASS.value)
        self.assertLess(result["risk"]["score"], 30)
        self.assertIn("QR_VERIFICATION_UNAVAILABLE", result["flags"])
        self.assertNotIn("NAME_MISMATCH", result["flags"])
        self.assertNotIn("DOB_MISMATCH", result["flags"])
        self.assertNotIn("GENDER_MISMATCH", result["flags"])

    def test_scenario_b_low_ocr_quality(self) -> None:
        """Scenario B: Low OCR quality -> LOW or MEDIUM risk (NOT forced HIGH)."""
        result = assess_document(get_scenario_b_low_ocr_quality())

        self.assertTrue(result["success"])
        self.assertIn(result["risk"]["level"], [RiskLevel.LOW.value, RiskLevel.MEDIUM.value])
        self.assertNotEqual(result["risk"]["level"], RiskLevel.HIGH.value)
        self.assertIn("LOW_OCR_CONFIDENCE", result["flags"])
        self.assertIn("LOW_FIELD_OCR_CONFIDENCE", result["flags"])

    def test_scenario_c_qr_not_detected(self) -> None:
        """Scenario C: QR not detected -> QR_NOT_DETECTED flag, LOW risk (NOT forced HIGH or fake)."""
        result = assess_document(get_scenario_c_qr_not_detected())

        self.assertTrue(result["success"])
        self.assertEqual(result["risk"]["level"], RiskLevel.LOW.value)
        self.assertEqual(result["risk"]["decision"], RiskDecision.PASS.value)
        self.assertIn("QR_NOT_DETECTED", result["flags"])
        self.assertEqual(result["risk"]["score"], round(15 / 200 * 100))  # 8 points

    def test_scenario_d_qr_detected_not_decoded(self) -> None:
        """Scenario D: QR detected but decode failed -> QR_DETECTED_NOT_DECODED flag, LOW risk (NOT forced HIGH)."""
        result = assess_document(get_scenario_d_qr_detected_not_decoded())

        self.assertTrue(result["success"])
        self.assertEqual(result["risk"]["level"], RiskLevel.LOW.value)
        self.assertEqual(result["risk"]["decision"], RiskDecision.PASS.value)
        self.assertIn("QR_DETECTED_NOT_DECODED", result["flags"])
        self.assertEqual(result["risk"]["score"], round(20 / 200 * 100))  # 10 points

    def test_scenario_e_name_mismatch(self) -> None:
        """Scenario E: Cross-validation name mismatch -> NAME_MISMATCH flag, HIGH risk override, REVIEW decision."""
        result = assess_document(get_scenario_e_name_mismatch())

        self.assertTrue(result["success"])
        self.assertEqual(result["risk"]["level"], RiskLevel.HIGH.value)
        self.assertEqual(result["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertIn("NAME_MISMATCH", result["flags"])

    def test_scenario_f_dob_mismatch(self) -> None:
        """Scenario F: Cross-validation DOB mismatch -> DOB_MISMATCH flag, HIGH risk override, REVIEW decision."""
        result = assess_document(get_scenario_f_dob_mismatch())

        self.assertTrue(result["success"])
        self.assertEqual(result["risk"]["level"], RiskLevel.HIGH.value)
        self.assertEqual(result["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertIn("DOB_MISMATCH", result["flags"])

    def test_scenario_g_gender_mismatch(self) -> None:
        """Scenario G: Cross-validation gender mismatch -> GENDER_MISMATCH flag, HIGH risk override, REVIEW decision."""
        result = assess_document(get_scenario_g_gender_mismatch())

        self.assertTrue(result["success"])
        self.assertEqual(result["risk"]["level"], RiskLevel.HIGH.value)
        self.assertEqual(result["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertIn("GENDER_MISMATCH", result["flags"])

    def test_scenario_h_non_aadhaar_document(self) -> None:
        """Scenario H: Non-Aadhaar document -> DOCUMENT_NOT_IDENTIFIED flag, HIGH risk override, REVIEW decision."""
        result = assess_document(get_scenario_h_non_aadhaar())

        self.assertTrue(result["success"])
        self.assertEqual(result["risk"]["level"], RiskLevel.HIGH.value)
        self.assertEqual(result["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertIn("DOCUMENT_NOT_IDENTIFIED", result["flags"])
        self.assertIn("could not be confidently identified as Aadhaar", result["risk"]["summary"])

    def test_scenario_i_upstream_ocr_failure(self) -> None:
        """Scenario I: Upstream OCR failure -> score=None, level=UNKNOWN, decision=REVIEW, error preserved."""
        result = assess_document(get_scenario_i_ocr_failure())

        self.assertFalse(result["success"])
        self.assertIsNone(result["risk"]["score"])
        self.assertEqual(result["risk"]["level"], RiskLevel.UNKNOWN.value)
        self.assertEqual(result["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertIn("DOCUMENT_PROCESSING_FAILED", result["flags"])
        self.assertEqual(result["error"]["code"], "CORRUPTED_OR_INVALID_FILE")


if __name__ == "__main__":
    unittest.main()
