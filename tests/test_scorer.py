"""
Comprehensive test suite for Stage 4: Risk Scoring and Risk Classification (risk_engine.scorer).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from typing import Any, Dict, List

from risk_engine import (
    assess_document,
    calculate_raw_score,
    classify_risk_level,
    determine_decision,
    normalize_risk_score,
    score_document,
)
from risk_engine.models import RiskDecision, RiskLevel


class TestRiskScorer(unittest.TestCase):
    """Test suite verifying mathematical scoring, risk tier boundaries, decision logic, and overrides."""

    def setUp(self) -> None:
        """Create baseline OCR output fixtures."""
        self.clean_ocr_result: Dict[str, Any] = {
            "success": True,
            "document": {
                "type": "aadhaar",
                "confidence": 0.99,
            },
            "ocr": {
                "language": "eng+hin",
                "confidence": 0.95,
                "fields": {
                    "name": {"value": "RAHUL KUMAR", "confidence": 0.96},
                    "dob": {"year": 2002, "month": 4, "day": 12, "precision": "full", "confidence": 0.92},
                    "gender": {"value": "MALE", "confidence": 0.97},
                    "aadhaar_number": {"value": "XXXX XXXX 1098", "confidence": 0.95},
                    "address": {"value": "123 MG Road, Bengaluru", "confidence": 0.88},
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
                "name": {"similarity": 1.0, "match": True},
                "dob": {"match": True, "comparison": "full"},
                "gender": {"match": True},
            },
            "warnings": [],
        }

    # -------------------------------------------------------------------------
    # 1. Score Calculation Tests
    # -------------------------------------------------------------------------

    def test_01_zero_findings_score(self) -> None:
        """Test 1: Zero findings results in raw score 0, normalized score 0, LOW risk, PASS decision."""
        summary = score_document([], [])
        self.assertEqual(summary.score, 0)
        self.assertEqual(summary.level, RiskLevel.LOW.value)
        self.assertEqual(summary.decision, RiskDecision.PASS.value)
        self.assertIn("low observed risk", summary.summary)

    def test_02_single_five_point_finding(self) -> None:
        """Test 2: A single 5-point finding normalizes according to round(5 / 200 * 100)."""
        findings = [{"rule_id": "QR_VERIFICATION_UNAVAILABLE", "points": 5, "triggered": True}]
        flags = ["QR_VERIFICATION_UNAVAILABLE"]
        summary = score_document(findings, flags)
        self.assertEqual(summary.score, round(5 / 200 * 100))
        self.assertEqual(summary.level, RiskLevel.LOW.value)
        self.assertEqual(summary.decision, RiskDecision.PASS.value)

    def test_03_multiple_findings_sum(self) -> None:
        """Test 3: Multiple findings sum and normalize accurately (15 + 20 + 30 = 65 -> round(65/200*100))."""
        findings = [
            {"rule_id": "LOW_OCR_CONFIDENCE", "points": 15, "triggered": True},
            {"rule_id": "QR_DETECTED_NOT_DECODED", "points": 20, "triggered": True},
            {"rule_id": "SOME_CUSTOM_FINDING", "points": 30, "triggered": True},
        ]
        flags = ["LOW_OCR_CONFIDENCE", "QR_DETECTED_NOT_DECODED", "SOME_CUSTOM_FINDING"]
        raw = calculate_raw_score(findings)
        self.assertEqual(raw, 65)
        normalized = normalize_risk_score(raw)
        self.assertEqual(normalized, round(65 / 200 * 100))

        summary = score_document(findings, flags)
        self.assertEqual(summary.score, normalized)
        self.assertEqual(summary.level, RiskLevel.MEDIUM.value)
        self.assertEqual(summary.decision, RiskDecision.REVIEW.value)

    def test_04_score_normalization_formula(self) -> None:
        """Test 4: Validate normalization formula across representative points."""
        test_points = [
            (0, 0),
            (2, 1),
            (10, 5),
            (50, 25),
            (100, 50),
            (150, 75),
            (200, 100),
        ]
        for raw, expected in test_points:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_risk_score(raw), expected)

    def test_05_score_clamped_to_100(self) -> None:
        """Test 5: Raw points exceeding 200 are strictly clamped to 100 maximum."""
        self.assertEqual(normalize_risk_score(250), 100)
        self.assertEqual(normalize_risk_score(500), 100)
        self.assertEqual(normalize_risk_score(-50), 0)

    # -------------------------------------------------------------------------
    # 2. Risk Boundaries Tests
    # -------------------------------------------------------------------------

    def test_06_boundary_score_29_low(self) -> None:
        """Test 6: Boundary: Score 29 -> LOW."""
        self.assertEqual(classify_risk_level(29), RiskLevel.LOW.value)

    def test_07_boundary_score_30_medium(self) -> None:
        """Test 7: Boundary: Score 30 -> MEDIUM."""
        self.assertEqual(classify_risk_level(30), RiskLevel.MEDIUM.value)

    def test_08_boundary_score_59_medium(self) -> None:
        """Test 8: Boundary: Score 59 -> MEDIUM."""
        self.assertEqual(classify_risk_level(59), RiskLevel.MEDIUM.value)

    def test_09_boundary_score_60_high(self) -> None:
        """Test 9: Boundary: Score 60 -> HIGH."""
        self.assertEqual(classify_risk_level(60), RiskLevel.HIGH.value)

    def test_10_boundary_score_100_high(self) -> None:
        """Test 10: Boundary: Score 100 -> HIGH."""
        self.assertEqual(classify_risk_level(100), RiskLevel.HIGH.value)

    # -------------------------------------------------------------------------
    # 3. Override Rules Tests
    # -------------------------------------------------------------------------

    def test_11_override_name_mismatch(self) -> None:
        """Test 11: NAME_MISMATCH forces risk level to at least HIGH and decision REVIEW."""
        # 30 raw pts -> score 15 (normally LOW), but override elevates to HIGH
        findings = [{"rule_id": "NAME_MISMATCH", "points": 30, "triggered": True}]
        flags = ["NAME_MISMATCH"]
        summary = score_document(findings, flags)
        self.assertEqual(summary.score, 15)
        self.assertEqual(summary.level, RiskLevel.HIGH.value)
        self.assertEqual(summary.decision, RiskDecision.REVIEW.value)
        self.assertIn("identity-consistency risk signals", summary.summary)

    def test_12_override_dob_mismatch(self) -> None:
        """Test 12: DOB_MISMATCH forces risk level to at least HIGH and decision REVIEW."""
        findings = [{"rule_id": "DOB_MISMATCH", "points": 30, "triggered": True}]
        flags = ["DOB_MISMATCH"]
        summary = score_document(findings, flags)
        self.assertEqual(summary.level, RiskLevel.HIGH.value)
        self.assertEqual(summary.decision, RiskDecision.REVIEW.value)

    def test_13_override_gender_mismatch(self) -> None:
        """Test 13: GENDER_MISMATCH forces risk level to at least HIGH and decision REVIEW."""
        findings = [{"rule_id": "GENDER_MISMATCH", "points": 20, "triggered": True}]
        flags = ["GENDER_MISMATCH"]
        summary = score_document(findings, flags)
        self.assertEqual(summary.level, RiskLevel.HIGH.value)
        self.assertEqual(summary.decision, RiskDecision.REVIEW.value)

    def test_14_override_qr_verification_failed(self) -> None:
        """Test 14: QR_VERIFICATION_FAILED forces risk level to at least HIGH and decision REVIEW."""
        findings = [{"rule_id": "QR_VERIFICATION_FAILED", "points": 35, "triggered": True}]
        flags = ["QR_VERIFICATION_FAILED"]
        summary = score_document(findings, flags)
        self.assertEqual(summary.level, RiskLevel.HIGH.value)
        self.assertEqual(summary.decision, RiskDecision.REVIEW.value)

    def test_15_override_document_not_identified(self) -> None:
        """Test 15: DOCUMENT_NOT_IDENTIFIED forces risk level to at least HIGH and decision REVIEW."""
        findings = [{"rule_id": "DOCUMENT_NOT_IDENTIFIED", "points": 30, "triggered": True}]
        flags = ["DOCUMENT_NOT_IDENTIFIED"]
        summary = score_document(findings, flags)
        self.assertEqual(summary.level, RiskLevel.HIGH.value)
        self.assertEqual(summary.decision, RiskDecision.REVIEW.value)
        self.assertIn("could not be confidently identified as Aadhaar", summary.summary)

    # -------------------------------------------------------------------------
    # 4. Non-Overrides Tests
    # -------------------------------------------------------------------------

    def test_16_non_override_qr_not_detected(self) -> None:
        """Test 16: QR_NOT_DETECTED does NOT automatically force HIGH (remains score-based)."""
        findings = [{"rule_id": "QR_NOT_DETECTED", "points": 15, "triggered": True}]
        flags = ["QR_NOT_DETECTED"]
        summary = score_document(findings, flags)
        self.assertEqual(summary.score, round(15 / 200 * 100))
        self.assertEqual(summary.level, RiskLevel.LOW.value)
        self.assertEqual(summary.decision, RiskDecision.PASS.value)

    def test_17_non_override_qr_detected_not_decoded(self) -> None:
        """Test 17: QR_DETECTED_NOT_DECODED does NOT automatically force HIGH."""
        findings = [{"rule_id": "QR_DETECTED_NOT_DECODED", "points": 20, "triggered": True}]
        flags = ["QR_DETECTED_NOT_DECODED"]
        summary = score_document(findings, flags)
        self.assertEqual(summary.score, round(20 / 200 * 100))
        self.assertEqual(summary.level, RiskLevel.LOW.value)
        self.assertEqual(summary.decision, RiskDecision.PASS.value)

    def test_18_non_override_qr_verification_unavailable(self) -> None:
        """Test 18: QR_VERIFICATION_UNAVAILABLE does NOT automatically force HIGH."""
        findings = [{"rule_id": "QR_VERIFICATION_UNAVAILABLE", "points": 5, "triggered": True}]
        flags = ["QR_VERIFICATION_UNAVAILABLE"]
        summary = score_document(findings, flags)
        self.assertEqual(summary.score, round(5 / 200 * 100))
        self.assertEqual(summary.level, RiskLevel.LOW.value)
        self.assertEqual(summary.decision, RiskDecision.PASS.value)

    # -------------------------------------------------------------------------
    # 5. Decision Mapping Tests
    # -------------------------------------------------------------------------

    def test_19_decision_low_is_pass(self) -> None:
        """Test 19: Decision for LOW risk is PASS."""
        self.assertEqual(determine_decision(RiskLevel.LOW.value), RiskDecision.PASS.value)

    def test_20_decision_medium_is_review(self) -> None:
        """Test 20: Decision for MEDIUM risk is REVIEW."""
        self.assertEqual(determine_decision(RiskLevel.MEDIUM.value), RiskDecision.REVIEW.value)

    def test_21_decision_high_is_review(self) -> None:
        """Test 21: Decision for HIGH risk is REVIEW (never FAKE)."""
        decision = determine_decision(RiskLevel.HIGH.value)
        self.assertEqual(decision, RiskDecision.REVIEW.value)
        self.assertNotEqual(decision, "FAKE")

    # -------------------------------------------------------------------------
    # 6. Failure & Boundary Handling Tests
    # -------------------------------------------------------------------------

    def test_22_failed_upstream_ocr_handling(self) -> None:
        """Test 22: Upstream OCR failure returns structured error with score=None, level=UNKNOWN, decision=REVIEW."""
        failed_ocr = {
            "success": False,
            "error": {
                "code": "CORRUPTED_OR_INVALID_FILE",
                "message": "The file is corrupt.",
            },
            "warnings": ["Corrupt header encountered."],
        }
        res = assess_document(failed_ocr)

        self.assertFalse(res["success"])
        self.assertIn("risk", res)
        self.assertIsNone(res["risk"]["score"])
        self.assertEqual(res["risk"]["level"], RiskLevel.UNKNOWN.value)
        self.assertEqual(res["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertEqual(res["flags"], ["DOCUMENT_PROCESSING_FAILED"])
        self.assertEqual(res["error"]["code"], "CORRUPTED_OR_INVALID_FILE")

    def test_23_invalid_input_handling(self) -> None:
        """Test 23: Invalid inputs (None, strings, numbers, lists) return structured INVALID_INPUT without throwing."""
        invalid_inputs = [None, "invalid_str", 12345, [1, 2, 3], {}]

        for item in invalid_inputs:
            with self.subTest(item=item):
                res = assess_document(item)  # type: ignore[arg-type]
                self.assertIsInstance(res, dict)
                self.assertFalse(res["success"])
                self.assertIsNone(res["risk"]["score"])
                self.assertEqual(res["risk"]["level"], RiskLevel.UNKNOWN.value)
                self.assertEqual(res["risk"]["decision"], RiskDecision.REVIEW.value)
                self.assertIn("error", res)

    def test_24_missing_optional_data_safe_assessment(self) -> None:
        """Test 24: Processing result with missing cross_validation or missing QR assesses cleanly."""
        input_data = copy.deepcopy(self.clean_ocr_result)
        input_data.pop("cross_validation", None)
        input_data["qr"] = {"detected": False, "decoded": False, "verified": False}

        res = assess_document(input_data)
        self.assertTrue(res["success"])
        self.assertIn("risk", res)
        self.assertEqual(res["risk"]["score"], round(15 / 200 * 100))  # QR_NOT_DETECTED (15 pts -> 8 score)
        self.assertEqual(res["risk"]["level"], RiskLevel.LOW.value)
        self.assertEqual(res["risk"]["decision"], RiskDecision.PASS.value)
        self.assertIn("QR_NOT_DETECTED", res["flags"])

    # -------------------------------------------------------------------------
    # 7. Security, PII, and Immutability Tests
    # -------------------------------------------------------------------------

    def test_25_raw_aadhaar_number_does_not_leak(self) -> None:
        """Test 25: Unmasked 12-digit Aadhaar numbers never appear in assessment output."""
        sensitive_data = copy.deepcopy(self.clean_ocr_result)
        sensitive_data["ocr"]["fields"]["aadhaar_number"]["value"] = "987654321098"
        sensitive_data["qr"]["fields"]["aadhaar_number"] = "987654321098"
        sensitive_data["warnings"] = ["Validated UID 9876 5432 1098 against database."]

        res = assess_document(sensitive_data)
        serialized = json.dumps(res)

        self.assertNotIn("987654321098", serialized)

    def test_26_raw_qr_payload_does_not_leak(self) -> None:
        """Test 26: Raw QR XML or JSON payloads do not leak into assessment results."""
        sensitive_data = copy.deepcopy(self.clean_ocr_result)
        sensitive_data["qr"]["raw_payload"] = "<PrintLetterBarcodeData uid='123456789012' name='Secret' />"

        res = assess_document(sensitive_data)
        serialized = json.dumps(res)

        self.assertNotIn("<PrintLetterBarcodeData", serialized)

    def test_27_input_immutability(self) -> None:
        """Test 27: assess_document never modifies the input dictionary."""
        original = copy.deepcopy(self.clean_ocr_result)
        input_copy = copy.deepcopy(self.clean_ocr_result)

        assess_document(input_copy)

        self.assertEqual(input_copy, original)

    def test_28_json_serializability(self) -> None:
        """Test 28: Complete output of assess_document is strictly JSON serializable."""
        res = assess_document(self.clean_ocr_result)
        try:
            serialized = json.dumps(res)
            deserialized = json.loads(serialized)
            self.assertEqual(deserialized["risk"]["decision"], "PASS")
            self.assertEqual(deserialized["risk"]["level"], "LOW")
        except TypeError as err:
            self.fail(f"assess_document result is not JSON-serializable: {err}")

    # -------------------------------------------------------------------------
    # 8. End-to-End Integration Test
    # -------------------------------------------------------------------------

    def test_29_end_to_end_integration(self) -> None:
        """Test 29: End-to-end pipeline from process_document -> extract_signals -> evaluate_rules -> assess_document."""
        from document_processor import process_document

        # Load synthetic test fixture generator
        spec = importlib.util.spec_from_file_location(
            "ocr_test_helpers", r"C:\icons\ocr\tests\test_helpers.py"
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        ocr_helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ocr_helpers)

        # 1. Create synthetic Aadhaar document bytes
        png_bytes = ocr_helpers.create_synthetic_aadhaar_bytes(
            name="DEEPAK VERMA",
            dob="15/08/1996",
            gender="MALE",
            aadhaar_num="9876 5432 1098",
            include_qr=True,
        )

        # 2. Run OCR document processor
        ocr_result = process_document(png_bytes)
        self.assertTrue(ocr_result["success"])

        # 3. Run Risk Assessment Engine
        assessment = assess_document(ocr_result)

        self.assertTrue(assessment["success"])
        self.assertIn("risk", assessment)
        self.assertIsInstance(assessment["risk"]["score"], int)
        self.assertIn(assessment["risk"]["level"], ["LOW", "MEDIUM", "HIGH"])
        self.assertIn(assessment["risk"]["decision"], ["PASS", "REVIEW"])
        self.assertIsInstance(assessment["findings"], list)
        self.assertIsInstance(assessment["flags"], list)
        self.assertIsInstance(assessment["signals"], dict)

        # Confirm JSON serializability of end-to-end result
        serialized = json.dumps(assessment)
        self.assertIsInstance(serialized, str)


if __name__ == "__main__":
    unittest.main()
