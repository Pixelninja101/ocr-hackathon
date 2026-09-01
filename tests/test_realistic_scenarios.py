"""
Production End-to-End Test Suite for Risk Engine (tests/test_realistic_scenarios.py).
Validates Scenarios A through L, multi-signal aggregation, score math, overrides, privacy recursion, and real OCR integration.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import unittest
from typing import Any, Dict, List

from document_processor import process_document
from risk_engine import (
    assess_document,
    calculate_raw_score,
    classify_risk_level,
    determine_decision,
    normalize_risk_score,
    score_document,
    validate_aadhaar_checksum,
)
from risk_engine.models import RiskDecision, RiskLevel, RuleSeverity


class TestRealisticScenarios(unittest.TestCase):
    """Full-coverage realistic document processing scenario validation."""

    def setUp(self) -> None:
        """Create standard baseline processing results."""
        # Valid 12-digit Aadhaar number with valid Verhoeff checksum (prefix 23456789012 + check digit 4)
        self.valid_aadhaar_num = "2345 6789 0124"
        # Invalid 12-digit Aadhaar number (final digit changed from 4 to 5)
        self.invalid_aadhaar_num = "2345 6789 0125"

        self.clean_aadhaar_result: Dict[str, Any] = {
            "success": True,
            "document": {
                "type": "aadhaar",
                "confidence": 0.98,
            },
            "ocr": {
                "language": "eng+hin",
                "confidence": 0.95,
                "fields": {
                    "name": {"value": "VIKRAM ADITYA", "confidence": 0.96},
                    "dob": {"year": 1992, "month": 5, "day": 14, "precision": "full", "confidence": 0.94},
                    "gender": {"value": "MALE", "confidence": 0.98},
                    "aadhaar_number": {"value": self.valid_aadhaar_num, "confidence": 0.95},
                    "address": {"value": "Flat 302, MG Road, Pune, Maharashtra", "confidence": 0.90},
                },
            },
            "qr": {
                "detected": True,
                "decoded": True,
                "verified": False,
                "format": "xml",
                "fields": {
                    "name": "VIKRAM ADITYA",
                    "dob": "14/05/1992",
                    "gender": "MALE",
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
    # Step 2: Scenarios A through L
    # -------------------------------------------------------------------------

    def test_scenario_a_clean_aadhaar(self) -> None:
        """Scenario A: Clean Aadhaar -> LOW risk, PASS decision, no mismatch/checksum-invalid flags."""
        res = assess_document(self.clean_aadhaar_result)

        self.assertTrue(res["success"])
        self.assertEqual(res["risk"]["level"], RiskLevel.LOW.value)
        self.assertEqual(res["risk"]["decision"], RiskDecision.PASS.value)
        self.assertLess(res["risk"]["score"], 30)

        # Confirm signals
        self.assertTrue(res["signals"]["aadhaar_checksum"]["available"])
        self.assertTrue(res["signals"]["aadhaar_checksum"]["valid"])

        # Flags: only QR_VERIFICATION_UNAVAILABLE (5 pts) is present
        self.assertIn("QR_VERIFICATION_UNAVAILABLE", res["flags"])
        self.assertNotIn("AADHAAR_CHECKSUM_INVALID", res["flags"])
        self.assertNotIn("NAME_MISMATCH", res["flags"])
        self.assertNotIn("DOB_MISMATCH", res["flags"])
        self.assertNotIn("GENDER_MISMATCH", res["flags"])
        self.assertNotIn("DOCUMENT_NOT_IDENTIFIED", res["flags"])

    def test_scenario_b_aadhaar_with_checksum_failure(self) -> None:
        """Scenario B: Aadhaar with checksum failure -> AADHAAR_CHECKSUM_INVALID (+15 pts, MEDIUM), no forced HIGH."""
        input_data = copy.deepcopy(self.clean_aadhaar_result)
        input_data["ocr"]["fields"]["aadhaar_number"]["value"] = self.invalid_aadhaar_num
        original_copy = copy.deepcopy(input_data)

        res = assess_document(input_data)

        # Immutability
        self.assertEqual(input_data, original_copy)

        self.assertTrue(res["success"])
        self.assertTrue(res["signals"]["aadhaar_checksum"]["available"])
        self.assertFalse(res["signals"]["aadhaar_checksum"]["valid"])
        self.assertIn("AADHAAR_CHECKSUM_INVALID", res["flags"])

        # Check finding properties
        finding = next(f for f in res["findings"] if f["rule_id"] == "AADHAAR_CHECKSUM_INVALID")
        self.assertEqual(finding["severity"], RuleSeverity.MEDIUM.value)
        self.assertEqual(finding["points"], 15)
        self.assertEqual(finding["evidence"], {"checksum_valid": False})

        # Does NOT directly force HIGH risk (5 pts QR + 15 pts checksum = 20 raw -> score 10 -> LOW)
        self.assertEqual(res["risk"]["level"], RiskLevel.LOW.value)
        self.assertEqual(res["risk"]["decision"], RiskDecision.PASS.value)

    def test_scenario_c_aadhaar_with_name_mismatch(self) -> None:
        """Scenario C: Aadhaar with OCR name mismatch -> NAME_MISMATCH, HIGH severity, 30 pts, HIGH override, REVIEW."""
        input_data = copy.deepcopy(self.clean_aadhaar_result)
        input_data["cross_validation"]["name"] = {"similarity": 0.32, "match": False}

        res = assess_document(input_data)

        self.assertTrue(res["success"])
        self.assertEqual(res["risk"]["level"], RiskLevel.HIGH.value)
        self.assertEqual(res["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertIn("NAME_MISMATCH", res["flags"])

        finding = next(f for f in res["findings"] if f["rule_id"] == "NAME_MISMATCH")
        self.assertEqual(finding["severity"], RuleSeverity.HIGH.value)
        self.assertEqual(finding["points"], 30)
        self.assertEqual(finding["evidence"], {"similarity": 0.32})
        # Ensure names do not leak in finding reason or evidence
        self.assertNotIn("VIKRAM", finding["reason"])
        self.assertNotIn("ADITYA", json.dumps(finding["evidence"]))

    def test_scenario_d_aadhaar_with_dob_mismatch_full_and_year(self) -> None:
        """Scenario D: Aadhaar with DOB mismatch -> DOB_MISMATCH, HIGH severity, 30 pts, HIGH override, REVIEW."""
        # 1. Full DOB comparison mismatch
        input_full = copy.deepcopy(self.clean_aadhaar_result)
        input_full["cross_validation"]["dob"] = {"match": False, "comparison": "full"}
        res_full = assess_document(input_full)

        self.assertEqual(res_full["risk"]["level"], RiskLevel.HIGH.value)
        self.assertEqual(res_full["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertIn("DOB_MISMATCH", res_full["flags"])
        finding_full = next(f for f in res_full["findings"] if f["rule_id"] == "DOB_MISMATCH")
        self.assertEqual(finding_full["evidence"], {"comparison": "full"})

        # 2. Year-only comparison mismatch
        input_year = copy.deepcopy(self.clean_aadhaar_result)
        input_year["cross_validation"]["dob"] = {"match": False, "comparison": "year"}
        res_year = assess_document(input_year)

        self.assertEqual(res_year["risk"]["level"], RiskLevel.HIGH.value)
        self.assertEqual(res_year["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertIn("DOB_MISMATCH", res_year["flags"])
        finding_year = next(f for f in res_year["findings"] if f["rule_id"] == "DOB_MISMATCH")
        self.assertEqual(finding_year["evidence"], {"comparison": "year"})

    def test_scenario_e_qr_not_detected(self) -> None:
        """Scenario E: QR not detected -> QR_NOT_DETECTED (15 pts), no HIGH override solely for missing QR."""
        input_data = copy.deepcopy(self.clean_aadhaar_result)
        input_data["qr"] = {"detected": False, "decoded": False, "verified": False}
        input_data.pop("cross_validation", None)

        res = assess_document(input_data)

        self.assertTrue(res["success"])
        self.assertEqual(res["risk"]["level"], RiskLevel.LOW.value)
        self.assertEqual(res["risk"]["decision"], RiskDecision.PASS.value)
        self.assertIn("QR_NOT_DETECTED", res["flags"])
        self.assertEqual(res["risk"]["score"], round(15 / 200 * 100))

    def test_scenario_f_qr_detected_not_decoded(self) -> None:
        """Scenario F: QR detected but not decoded -> QR_DETECTED_NOT_DECODED (20 pts), no HIGH override."""
        input_data = copy.deepcopy(self.clean_aadhaar_result)
        input_data["qr"] = {"detected": True, "decoded": False, "verified": False, "error": "DECODE_FAILED"}
        input_data.pop("cross_validation", None)

        res = assess_document(input_data)

        self.assertTrue(res["success"])
        self.assertEqual(res["risk"]["level"], RiskLevel.LOW.value)
        self.assertEqual(res["risk"]["decision"], RiskDecision.PASS.value)
        self.assertIn("QR_DETECTED_NOT_DECODED", res["flags"])
        self.assertEqual(res["risk"]["score"], round(20 / 200 * 100))

    def test_scenario_g_qr_decoded_verification_unavailable(self) -> None:
        """Scenario G: QR decoded but verification unavailable -> QR_VERIFICATION_UNAVAILABLE (5 pts, LOW)."""
        input_data = copy.deepcopy(self.clean_aadhaar_result)
        # Already verification_status="QR_DECODED_VERIFICATION_UNAVAILABLE" by default
        res = assess_document(input_data)

        self.assertTrue(res["success"])
        self.assertEqual(res["risk"]["level"], RiskLevel.LOW.value)
        self.assertEqual(res["risk"]["decision"], RiskDecision.PASS.value)
        self.assertIn("QR_VERIFICATION_UNAVAILABLE", res["flags"])
        self.assertNotIn("QR_VERIFICATION_FAILED", res["flags"])

    def test_scenario_h_explicit_cryptographic_verification_failure(self) -> None:
        """Scenario H: Explicit QR verification failure -> QR_VERIFICATION_FAILED (35 pts, HIGH override, REVIEW)."""
        input_data = copy.deepcopy(self.clean_aadhaar_result)
        input_data["qr"]["verification_status"] = "FAILED"
        input_data["qr"]["error"] = "SIGNATURE_MISMATCH"

        res = assess_document(input_data)

        self.assertTrue(res["success"])
        self.assertEqual(res["risk"]["level"], RiskLevel.HIGH.value)
        self.assertEqual(res["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertIn("QR_VERIFICATION_FAILED", res["flags"])
        finding = next(f for f in res["findings"] if f["rule_id"] == "QR_VERIFICATION_FAILED")
        self.assertEqual(finding["severity"], RuleSeverity.HIGH.value)
        self.assertEqual(finding["points"], 35)

    def test_scenario_i_non_aadhaar_document(self) -> None:
        """Scenario I: Non-Aadhaar document -> DOCUMENT_NOT_IDENTIFIED (HIGH override, REVIEW)."""
        input_data = copy.deepcopy(self.clean_aadhaar_result)
        input_data["document"]["type"] = "not_aadhaar"
        input_data["document"]["confidence"] = 0.82

        res = assess_document(input_data)

        self.assertTrue(res["success"])
        self.assertEqual(res["risk"]["level"], RiskLevel.HIGH.value)
        self.assertEqual(res["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertIn("DOCUMENT_NOT_IDENTIFIED", res["flags"])
        self.assertIn("could not be confidently identified as Aadhaar", res["risk"]["summary"])

    def test_scenario_j_missing_critical_fields_scaling(self) -> None:
        """Scenario J: Missing critical fields scales points properly (1=5pts, 2=10pts, 3=20pts, 4=25pts)."""
        # 1 missing (gender missing)
        inp1 = copy.deepcopy(self.clean_aadhaar_result)
        inp1["ocr"]["fields"]["gender"] = None
        res1 = assess_document(inp1)
        f1 = next(f for f in res1["findings"] if f["rule_id"] == "MISSING_CRITICAL_FIELDS")
        self.assertEqual(f1["points"], 5)
        self.assertEqual(f1["severity"], RuleSeverity.LOW.value)

        # 2 missing (gender + dob missing)
        inp2 = copy.deepcopy(self.clean_aadhaar_result)
        inp2["ocr"]["fields"]["gender"] = None
        inp2["ocr"]["fields"]["dob"] = None
        res2 = assess_document(inp2)
        f2 = next(f for f in res2["findings"] if f["rule_id"] == "MISSING_CRITICAL_FIELDS")
        self.assertEqual(f2["points"], 10)
        self.assertEqual(f2["severity"], RuleSeverity.MEDIUM.value)

        # 3 missing (gender + dob + name missing)
        inp3 = copy.deepcopy(self.clean_aadhaar_result)
        inp3["ocr"]["fields"]["gender"] = None
        inp3["ocr"]["fields"]["dob"] = None
        inp3["ocr"]["fields"]["name"] = None
        res3 = assess_document(inp3)
        f3 = next(f for f in res3["findings"] if f["rule_id"] == "MISSING_CRITICAL_FIELDS")
        self.assertEqual(f3["points"], 20)
        self.assertEqual(f3["severity"], RuleSeverity.HIGH.value)

        # 4 missing (gender + dob + name + aadhaar_number missing)
        inp4 = copy.deepcopy(self.clean_aadhaar_result)
        inp4["ocr"]["fields"]["gender"] = None
        inp4["ocr"]["fields"]["dob"] = None
        inp4["ocr"]["fields"]["name"] = None
        inp4["ocr"]["fields"]["aadhaar_number"] = None
        res4 = assess_document(inp4)
        f4 = next(f for f in res4["findings"] if f["rule_id"] == "MISSING_CRITICAL_FIELDS")
        self.assertEqual(f4["points"], 25)
        self.assertEqual(f4["severity"], RuleSeverity.HIGH.value)

    def test_scenario_k_masked_aadhaar(self) -> None:
        """Scenario K: Masked Aadhaar (XXXX XXXX 1234) -> checksum unavailable, AADHAAR_CHECKSUM_INVALID does NOT trigger."""
        inp = copy.deepcopy(self.clean_aadhaar_result)
        inp["ocr"]["fields"]["aadhaar_number"]["value"] = "XXXX XXXX 1234"

        res = assess_document(inp)

        self.assertTrue(res["success"])
        self.assertFalse(res["signals"]["aadhaar_checksum"]["available"])
        self.assertIsNone(res["signals"]["aadhaar_checksum"]["valid"])
        self.assertNotIn("AADHAAR_CHECKSUM_INVALID", res["flags"])

    def test_scenario_l_upstream_ocr_failure(self) -> None:
        """Scenario L: Upstream OCR failure -> score=None, level=UNKNOWN, decision=REVIEW, structured error."""
        failed_input = {
            "success": False,
            "error": {
                "code": "INVALID_IMAGE_FILE",
                "message": "The file cannot be read by image decoder.",
            },
            "warnings": ["Corrupt header encountered."],
        }

        res = assess_document(failed_input)

        self.assertFalse(res["success"])
        self.assertIsNone(res["risk"]["score"])
        self.assertEqual(res["risk"]["level"], RiskLevel.UNKNOWN.value)
        self.assertEqual(res["risk"]["decision"], RiskDecision.REVIEW.value)
        self.assertEqual(res["flags"], ["DOCUMENT_PROCESSING_FAILED"])
        self.assertEqual(res["error"]["code"], "INVALID_IMAGE_FILE")

    # -------------------------------------------------------------------------
    # Step 3: Multiple Simultaneous Signals
    # -------------------------------------------------------------------------

    def test_multiple_simultaneous_signals_aggregation(self) -> None:
        """Verify multiple simultaneous signals accumulate points, clamp to 100, and preserve overrides."""
        inp = {
            "success": True,
            "document": {"type": "aadhaar", "confidence": 0.72},  # LOW_DOCUMENT_CONFIDENCE (15 pts)
            "ocr": {
                "confidence": 0.55,  # LOW_OCR_CONFIDENCE (15 pts)
                "fields": {
                    "name": {"value": "RAJESH KHANNA", "confidence": 0.52},  # low field conf (5 pts)
                    "dob": None,  # MISSING_CRITICAL_FIELDS (5 pts)
                    "gender": {"value": "MALE", "confidence": 0.50},  # low field conf (5 pts)
                    "aadhaar_number": {"value": self.invalid_aadhaar_num, "confidence": 0.85},  # AADHAAR_CHECKSUM_INVALID (15 pts)
                    "address": {"value": "Some Address", "confidence": 0.70},
                },
            },
            "qr": {
                "detected": True,
                "decoded": False,  # QR_DETECTED_NOT_DECODED (20 pts)
                "verified": False,
            },
            "warnings": [],
        }

        res = assess_document(inp)

        self.assertTrue(res["success"])
        expected_flags = [
            "LOW_DOCUMENT_CONFIDENCE",
            "LOW_OCR_CONFIDENCE",
            "MISSING_CRITICAL_FIELDS",
            "QR_DETECTED_NOT_DECODED",
            "LOW_FIELD_OCR_CONFIDENCE",
            "AADHAAR_CHECKSUM_INVALID",
        ]
        for ef in expected_flags:
            self.assertIn(ef, res["flags"])

        # Points: 15 (doc) + 15 (ocr) + 5 (missing) + 20 (qr) + 10 (2 low fields) + 15 (chk) = 80 raw pts
        # Normalized score: round(80 / 200 * 100) = 40 (MEDIUM)
        self.assertEqual(res["risk"]["score"], 40)
        self.assertEqual(res["risk"]["level"], RiskLevel.MEDIUM.value)
        self.assertEqual(res["risk"]["decision"], RiskDecision.REVIEW.value)

    # -------------------------------------------------------------------------
    # Step 4: Score Mathematics Verification
    # -------------------------------------------------------------------------

    def test_score_mathematics_and_thresholds(self) -> None:
        """Verify normalization formula, score clamping, and exact threshold boundaries."""
        # 1. Formula & clamping
        self.assertEqual(normalize_risk_score(0), 0)
        self.assertEqual(normalize_risk_score(5), 2)
        self.assertEqual(normalize_risk_score(15), 8)
        self.assertEqual(normalize_risk_score(30), 15)
        self.assertEqual(normalize_risk_score(60), 30)
        self.assertEqual(normalize_risk_score(100), 50)
        self.assertEqual(normalize_risk_score(120), 60)
        self.assertEqual(normalize_risk_score(200), 100)
        self.assertEqual(normalize_risk_score(250), 100)  # Clamped

        # 2. Level classification thresholds
        self.assertEqual(classify_risk_level(0), RiskLevel.LOW.value)
        self.assertEqual(classify_risk_level(29), RiskLevel.LOW.value)
        self.assertEqual(classify_risk_level(30), RiskLevel.MEDIUM.value)
        self.assertEqual(classify_risk_level(59), RiskLevel.MEDIUM.value)
        self.assertEqual(classify_risk_level(60), RiskLevel.HIGH.value)
        self.assertEqual(classify_risk_level(100), RiskLevel.HIGH.value)

        # 3. Decision mapping
        self.assertEqual(determine_decision(RiskLevel.LOW.value), RiskDecision.PASS.value)
        self.assertEqual(determine_decision(RiskLevel.MEDIUM.value), RiskDecision.REVIEW.value)
        self.assertEqual(determine_decision(RiskLevel.HIGH.value), RiskDecision.REVIEW.value)

    # -------------------------------------------------------------------------
    # Step 5: Override Precedence
    # -------------------------------------------------------------------------

    def test_override_precedence_rules(self) -> None:
        """Verify override elevates LOW/MEDIUM to HIGH, preserves HIGH, and non-critical QR does not override."""
        # 1. LOW score + critical override (NAME_MISMATCH 30 pts -> score 15 LOW base) -> Final HIGH
        low_with_override = score_document(
            [{"rule_id": "NAME_MISMATCH", "points": 30, "triggered": True}],
            ["NAME_MISMATCH"],
        )
        self.assertEqual(low_with_override.level, RiskLevel.HIGH.value)
        self.assertEqual(low_with_override.decision, RiskDecision.REVIEW.value)

        # 2. MEDIUM score + critical override (70 raw pts -> score 35 MEDIUM base + DOB_MISMATCH) -> Final HIGH
        med_with_override = score_document(
            [
                {"rule_id": "LOW_OCR_CONFIDENCE", "points": 40, "triggered": True},
                {"rule_id": "DOB_MISMATCH", "points": 30, "triggered": True},
            ],
            ["LOW_OCR_CONFIDENCE", "DOB_MISMATCH"],
        )
        self.assertEqual(med_with_override.level, RiskLevel.HIGH.value)

        # 3. HIGH score + no override (140 raw pts -> score 70 HIGH base) -> Final HIGH
        high_no_override = score_document(
            [{"rule_id": "CUMULATIVE_RULES", "points": 140, "triggered": True}],
            ["CUMULATIVE_RULES"],
        )
        self.assertEqual(high_no_override.level, RiskLevel.HIGH.value)

        # 4. LOW score + non-critical QR issue (QR_NOT_DETECTED 15 pts -> score 8) -> Remains LOW
        low_qr_issue = score_document(
            [{"rule_id": "QR_NOT_DETECTED", "points": 15, "triggered": True}],
            ["QR_NOT_DETECTED"],
        )
        self.assertEqual(low_qr_issue.level, RiskLevel.LOW.value)

    # -------------------------------------------------------------------------
    # Step 6: Deep Privacy & Recursive PII Protection
    # -------------------------------------------------------------------------

    def test_deep_recursive_privacy_inspection(self) -> None:
        """Recursively inspect every nested key and string in assessment result to verify zero unmasked PII."""
        sensitive_aadhaar = "987654321096"
        sensitive_name = "DEEP PRIVATE NAME"
        sensitive_address = "Sector 5, Confidential Apartment, New Delhi 110001"

        inp = copy.deepcopy(self.clean_aadhaar_result)
        inp["ocr"]["fields"]["name"]["value"] = sensitive_name
        inp["ocr"]["fields"]["address"]["value"] = sensitive_address
        inp["ocr"]["fields"]["aadhaar_number"]["value"] = sensitive_aadhaar
        inp["qr"]["fields"]["name"] = sensitive_name
        inp["qr"]["fields"]["aadhaar_number"] = sensitive_aadhaar
        inp["warnings"] = [f"Processed record with Aadhaar {sensitive_aadhaar}"]

        res = assess_document(inp)
        serialized = json.dumps(res)

        # 1. Unmasked continuous 12-digit number should never appear
        self.assertNotIn(sensitive_aadhaar, serialized)
        # 2. No unmasked 12 digits regex match
        self.assertFalse(bool(re.search(r"\b\d{12}\b", serialized)))
        # 3. Raw names, addresses, or OCR text should not appear in findings/evidence/summary
        for f in res.get("findings", []):
            f_str = json.dumps(f)
            self.assertNotIn(sensitive_name, f_str)
            self.assertNotIn(sensitive_address, f_str)

    # -------------------------------------------------------------------------
    # Step 7: Real OCR Integration Test
    # -------------------------------------------------------------------------

    def test_real_document_processor_integration(self) -> None:
        """Integration test executing document_processor.process_document() -> risk_engine.assess_document()."""
        spec = importlib.util.spec_from_file_location(
            "ocr_test_helpers", r"C:\icons\ocr\tests\test_helpers.py"
        )
        assert spec is not None and spec.loader is not None
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)

        # Generate synthetic test Aadhaar image bytes
        img_bytes = helpers.create_synthetic_aadhaar_bytes(
            name="KAVITA MENON",
            dob="22/10/1993",
            gender="FEMALE",
            aadhaar_num=self.valid_aadhaar_num,
            include_qr=True,
        )

        # 1. Run actual OCR processor
        ocr_result = process_document(img_bytes)
        self.assertTrue(ocr_result["success"])

        # 2. Run Risk Assessment engine
        assessment = assess_document(ocr_result)
        self.assertTrue(assessment["success"])
        self.assertIn("risk", assessment)
        self.assertIn("signals", assessment)
        self.assertIn("findings", assessment)
        self.assertIn("flags", assessment)

        # Output must be standard JSON serializable
        serialized = json.dumps(assessment)
        self.assertIsInstance(serialized, str)

    # -------------------------------------------------------------------------
    # Step 9: Error Handling & Malformed Input Robustness
    # -------------------------------------------------------------------------

    def test_malformed_and_edge_case_inputs(self) -> None:
        """Verify no unhandled exceptions escape assess_document for all forms of bad inputs."""
        malformed_inputs = [
            None,
            {},
            [],
            "arbitrary_string",
            12345,
            {"success": "invalid_boolean"},
            {"success": True, "document": "not_a_dict"},
            {"success": True, "ocr": {"fields": None}},
            {"success": True, "qr": {"detected": "yes", "decoded": "no"}},
            {"success": True, "cross_validation": None},
            {"success": True, "ocr": {"confidence": "invalid_conf", "fields": {"name": {"confidence": "bad"}}}},
        ]

        for item in malformed_inputs:
            with self.subTest(item=item):
                res = assess_document(item)  # type: ignore[arg-type]
                self.assertIsInstance(res, dict)
                self.assertIn("risk", res)
                self.assertIn("success", res)
                # JSON serialization must succeed
                serialized = json.dumps(res)
                self.assertIsInstance(serialized, str)


if __name__ == "__main__":
    unittest.main()
