"""
Unit tests for Stage 3 Explainable Risk Rules (risk_engine.rules).
"""

from __future__ import annotations

import copy
import json
import unittest
from typing import Any, Dict

from risk_engine.models import QRVerificationStatus, RuleSeverity
from risk_engine.rules import (
    RULES,
    evaluate_rules,
    rule_dob_mismatch,
    rule_document_not_identified,
    rule_gender_mismatch,
    rule_low_document_confidence,
    rule_low_field_ocr_confidence,
    rule_low_ocr_confidence,
    rule_missing_critical_fields,
    rule_name_mismatch,
    rule_qr_detected_not_decoded,
    rule_qr_not_detected,
    rule_qr_verification_failed,
    rule_qr_verification_unavailable,
)


class TestRiskRules(unittest.TestCase):
    """Test suite validating explainable risk rules against normalized signals."""

    def setUp(self) -> None:
        """Create baseline normalized signals for a clean valid Aadhaar document."""
        self.clean_signals: Dict[str, Any] = {
            "document": {
                "type": "aadhaar",
                "confidence": 0.98,
            },
            "ocr": {
                "confidence": 0.94,
                "language": "eng+hin",
                "fields": {
                    "name": {"available": True, "confidence": 0.95},
                    "dob": {"available": True, "confidence": 0.92, "precision": "full"},
                    "gender": {"available": True, "confidence": 0.96},
                    "aadhaar_number": {"available": True, "confidence": 0.95},
                    "address": {"available": True, "confidence": 0.88},
                },
            },
            "qr": {
                "detected": True,
                "decoded": True,
                "verified": False,
                "verification_status": QRVerificationStatus.QR_DECODED_VERIFICATION_UNAVAILABLE.value,
            },
            "cross_validation": {
                "available": True,
                "name": {"available": True, "similarity": 1.0, "match": True},
                "dob": {"available": True, "match": True, "comparison": "full"},
                "gender": {"available": True, "match": True},
            },
            "warnings": [],
        }

    def test_01_valid_aadhaar_clean_evaluation(self) -> None:
        """Test 1: Valid clean Aadhaar triggers only informational QR_VERIFICATION_UNAVAILABLE (5 pts)."""
        res = evaluate_rules(self.clean_signals)
        findings = res["findings"]
        rule_ids = [f["rule_id"] for f in findings]

        # Only the expected informational QR verification unavailable rule triggers
        self.assertEqual(rule_ids, ["QR_VERIFICATION_UNAVAILABLE"])
        self.assertEqual(res["total_points"], 5)
        self.assertEqual(findings[0]["severity"], RuleSeverity.LOW.value)

    def test_02_document_not_identified(self) -> None:
        """Test 2: Non-Aadhaar document triggers DOCUMENT_NOT_IDENTIFIED (30 pts, MEDIUM)."""
        signals = copy.deepcopy(self.clean_signals)
        signals["document"] = {"type": "not_aadhaar", "confidence": 0.10}

        finding = rule_document_not_identified(signals)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.rule_id, "DOCUMENT_NOT_IDENTIFIED")
        self.assertEqual(finding.points, 30)
        self.assertEqual(finding.severity, RuleSeverity.MEDIUM.value)
        self.assertIn("could not be confidently identified", finding.reason)
        self.assertEqual(finding.evidence["document_type"], "not_aadhaar")

        # Must not trigger when document type is aadhaar
        signals["document"] = {"type": "aadhaar", "confidence": 0.95}
        self.assertIsNone(rule_document_not_identified(signals))

    def test_03_low_document_confidence(self) -> None:
        """Test 3: Low document classification confidence (< 0.80) triggers LOW_DOCUMENT_CONFIDENCE (15 pts, MEDIUM)."""
        signals = copy.deepcopy(self.clean_signals)
        signals["document"] = {"type": "aadhaar", "confidence": 0.72}

        finding = rule_low_document_confidence(signals)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.rule_id, "LOW_DOCUMENT_CONFIDENCE")
        self.assertEqual(finding.points, 15)
        self.assertEqual(finding.severity, RuleSeverity.MEDIUM.value)
        self.assertEqual(finding.evidence["confidence"], 0.72)

        # Confidence exactly 0.80 or greater must not trigger
        signals["document"]["confidence"] = 0.80
        self.assertIsNone(rule_low_document_confidence(signals))
        signals["document"]["confidence"] = 0.95
        self.assertIsNone(rule_low_document_confidence(signals))

    def test_04_low_ocr_confidence(self) -> None:
        """Test 4: Low overall OCR confidence (< 0.70) triggers LOW_OCR_CONFIDENCE (15 pts, MEDIUM)."""
        signals = copy.deepcopy(self.clean_signals)
        signals["ocr"]["confidence"] = 0.65

        finding = rule_low_ocr_confidence(signals)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.rule_id, "LOW_OCR_CONFIDENCE")
        self.assertEqual(finding.points, 15)
        self.assertEqual(finding.severity, RuleSeverity.MEDIUM.value)
        self.assertEqual(finding.evidence["ocr_confidence"], 0.65)

        # Confidence >= 0.70 must not trigger
        signals["ocr"]["confidence"] = 0.70
        self.assertIsNone(rule_low_ocr_confidence(signals))

    def test_05_missing_one_critical_field(self) -> None:
        """Test 5: One missing critical field triggers MISSING_CRITICAL_FIELDS (5 pts, LOW)."""
        signals = copy.deepcopy(self.clean_signals)
        signals["ocr"]["fields"]["name"]["available"] = False

        finding = rule_missing_critical_fields(signals)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.rule_id, "MISSING_CRITICAL_FIELDS")
        self.assertEqual(finding.points, 5)
        self.assertEqual(finding.severity, RuleSeverity.LOW.value)
        self.assertEqual(finding.evidence["missing_fields"], ["name"])
        self.assertEqual(finding.evidence["missing_count"], 1)

    def test_06_missing_multiple_critical_fields(self) -> None:
        """Test 6: Multiple missing critical fields scale points appropriately."""
        signals = copy.deepcopy(self.clean_signals)

        # 2 missing: 10 pts, MEDIUM
        signals["ocr"]["fields"]["name"]["available"] = False
        signals["ocr"]["fields"]["dob"]["available"] = False
        f2 = rule_missing_critical_fields(signals)
        self.assertIsNotNone(f2)
        self.assertEqual(f2.points, 10)
        self.assertEqual(f2.severity, RuleSeverity.MEDIUM.value)
        self.assertEqual(f2.evidence["missing_count"], 2)

        # 3 missing: 20 pts, HIGH
        signals["ocr"]["fields"]["gender"]["available"] = False
        f3 = rule_missing_critical_fields(signals)
        self.assertIsNotNone(f3)
        self.assertEqual(f3.points, 20)
        self.assertEqual(f3.severity, RuleSeverity.HIGH.value)

        # 4 missing: 25 pts, HIGH
        signals["ocr"]["fields"]["aadhaar_number"]["available"] = False
        f4 = rule_missing_critical_fields(signals)
        self.assertIsNotNone(f4)
        self.assertEqual(f4.points, 25)
        self.assertEqual(f4.severity, RuleSeverity.HIGH.value)

        # Address missing does NOT count as a critical field
        clean = copy.deepcopy(self.clean_signals)
        clean["ocr"]["fields"]["address"]["available"] = False
        self.assertIsNone(rule_missing_critical_fields(clean))

    def test_07_qr_not_detected(self) -> None:
        """Test 7: QR not detected triggers QR_NOT_DETECTED (15 pts, MEDIUM)."""
        signals = copy.deepcopy(self.clean_signals)
        signals["qr"] = {
            "detected": False,
            "decoded": False,
            "verified": False,
            "verification_status": QRVerificationStatus.QR_NOT_DETECTED.value,
        }

        finding = rule_qr_not_detected(signals)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.rule_id, "QR_NOT_DETECTED")
        self.assertEqual(finding.points, 15)
        self.assertEqual(finding.severity, RuleSeverity.MEDIUM.value)

        # Ensure QR_DETECTED_NOT_DECODED does not trigger
        self.assertIsNone(rule_qr_detected_not_decoded(signals))

    def test_08_qr_detected_not_decoded(self) -> None:
        """Test 8: QR detected but not decoded triggers QR_DETECTED_NOT_DECODED (20 pts, MEDIUM) exclusively."""
        signals = copy.deepcopy(self.clean_signals)
        signals["qr"] = {
            "detected": True,
            "decoded": False,
            "verified": False,
            "verification_status": QRVerificationStatus.QR_DETECTED_NOT_DECODED.value,
        }

        finding = rule_qr_detected_not_decoded(signals)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.rule_id, "QR_DETECTED_NOT_DECODED")
        self.assertEqual(finding.points, 20)
        self.assertEqual(finding.severity, RuleSeverity.MEDIUM.value)

        # QR_NOT_DETECTED must NOT trigger
        self.assertIsNone(rule_qr_not_detected(signals))

    def test_09_qr_decoded_verification_unavailable(self) -> None:
        """Test 9: QR decoded with verification unavailable triggers QR_VERIFICATION_UNAVAILABLE (5 pts, LOW)."""
        signals = copy.deepcopy(self.clean_signals)
        signals["qr"] = {
            "detected": True,
            "decoded": True,
            "verified": False,
            "verification_status": QRVerificationStatus.QR_DECODED_VERIFICATION_UNAVAILABLE.value,
        }

        finding = rule_qr_verification_unavailable(signals)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.rule_id, "QR_VERIFICATION_UNAVAILABLE")
        self.assertEqual(finding.points, 5)
        self.assertEqual(finding.severity, RuleSeverity.LOW.value)
        self.assertNotIn("failed", finding.reason.lower())

        # Failed rule must NOT trigger
        self.assertIsNone(rule_qr_verification_failed(signals))

    def test_10_explicit_qr_verification_failed(self) -> None:
        """Test 10: Explicit QR verification failure triggers QR_VERIFICATION_FAILED (35 pts, HIGH)."""
        signals = copy.deepcopy(self.clean_signals)
        signals["qr"] = {
            "detected": True,
            "decoded": True,
            "verified": False,
            "verification_status": QRVerificationStatus.QR_VERIFICATION_FAILED.value,
        }

        finding = rule_qr_verification_failed(signals)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.rule_id, "QR_VERIFICATION_FAILED")
        self.assertEqual(finding.points, 35)
        self.assertEqual(finding.severity, RuleSeverity.HIGH.value)

        # Unavailable rule must NOT trigger
        self.assertIsNone(rule_qr_verification_unavailable(signals))

    def test_11_name_mismatch(self) -> None:
        """Test 11: Name mismatch triggers NAME_MISMATCH (30 pts, HIGH) with similarity evidence."""
        signals = copy.deepcopy(self.clean_signals)
        signals["cross_validation"]["name"] = {
            "available": True,
            "similarity": 0.45,
            "match": False,
        }

        finding = rule_name_mismatch(signals)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.rule_id, "NAME_MISMATCH")
        self.assertEqual(finding.points, 30)
        self.assertEqual(finding.severity, RuleSeverity.HIGH.value)
        self.assertEqual(finding.evidence["similarity"], 0.45)
        # Verify no names in evidence
        self.assertNotIn("name", finding.evidence)
        self.assertNotIn("ocr_name", finding.evidence)

    def test_12_name_unavailable_no_mismatch_rule(self) -> None:
        """Test 12: Name unavailable in cross-validation must NOT trigger NAME_MISMATCH."""
        signals = copy.deepcopy(self.clean_signals)
        signals["cross_validation"]["name"] = {
            "available": False,
            "similarity": None,
            "match": None,
        }

        self.assertIsNone(rule_name_mismatch(signals))

        # Also when entire cross_validation is unavailable
        signals["cross_validation"] = {"available": False}
        self.assertIsNone(rule_name_mismatch(signals))

    def test_13_dob_full_mismatch(self) -> None:
        """Test 13: Full DOB mismatch triggers DOB_MISMATCH (30 pts, HIGH)."""
        signals = copy.deepcopy(self.clean_signals)
        signals["cross_validation"]["dob"] = {
            "available": True,
            "match": False,
            "comparison": "full",
        }

        finding = rule_dob_mismatch(signals)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.rule_id, "DOB_MISMATCH")
        self.assertEqual(finding.points, 30)
        self.assertEqual(finding.severity, RuleSeverity.HIGH.value)
        self.assertEqual(finding.evidence["comparison"], "full")
        # Verify no DOB dates in evidence
        self.assertNotIn("dob", finding.evidence)

    def test_14_dob_year_only_mismatch(self) -> None:
        """Test 14: Year-only DOB mismatch triggers DOB_MISMATCH with comparison='year'."""
        signals = copy.deepcopy(self.clean_signals)
        signals["cross_validation"]["dob"] = {
            "available": True,
            "match": False,
            "comparison": "year",
        }

        finding = rule_dob_mismatch(signals)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.rule_id, "DOB_MISMATCH")
        self.assertEqual(finding.points, 30)
        self.assertEqual(finding.evidence["comparison"], "year")

    def test_15_dob_unavailable_no_mismatch_rule(self) -> None:
        """Test 15: DOB unavailable in cross-validation must NOT trigger DOB_MISMATCH."""
        signals = copy.deepcopy(self.clean_signals)
        signals["cross_validation"]["dob"] = {
            "available": False,
            "match": None,
            "comparison": None,
        }

        self.assertIsNone(rule_dob_mismatch(signals))

    def test_16_gender_mismatch(self) -> None:
        """Test 16: Gender mismatch triggers GENDER_MISMATCH (20 pts, MEDIUM)."""
        signals = copy.deepcopy(self.clean_signals)
        signals["cross_validation"]["gender"] = {
            "available": True,
            "match": False,
        }

        finding = rule_gender_mismatch(signals)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.rule_id, "GENDER_MISMATCH")
        self.assertEqual(finding.points, 20)
        self.assertEqual(finding.severity, RuleSeverity.MEDIUM.value)

        # Gender unavailable must not trigger
        signals["cross_validation"]["gender"] = {
            "available": False,
            "match": None,
        }
        self.assertIsNone(rule_gender_mismatch(signals))

    def test_17_low_confidence_ocr_single_field(self) -> None:
        """Test 17: Single OCR field < 0.60 triggers LOW_FIELD_OCR_CONFIDENCE (5 pts, LOW)."""
        signals = copy.deepcopy(self.clean_signals)
        signals["ocr"]["fields"]["address"]["confidence"] = 0.52

        finding = rule_low_field_ocr_confidence(signals)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.rule_id, "LOW_FIELD_OCR_CONFIDENCE")
        self.assertEqual(finding.points, 5)
        self.assertEqual(finding.severity, RuleSeverity.LOW.value)
        self.assertEqual(finding.evidence["low_confidence_fields"], ["address"])

    def test_18_multiple_low_confidence_ocr_fields_capped(self) -> None:
        """Test 18: Multiple low confidence OCR fields increment points and cap at 15."""
        signals = copy.deepcopy(self.clean_signals)
        signals["ocr"]["fields"]["name"]["confidence"] = 0.48
        signals["ocr"]["fields"]["dob"]["confidence"] = 0.50

        # 2 fields -> 10 pts
        f2 = rule_low_field_ocr_confidence(signals)
        self.assertIsNotNone(f2)
        self.assertEqual(f2.points, 10)
        self.assertEqual(len(f2.evidence["low_confidence_fields"]), 2)

        # 4 fields -> capped at 15 pts
        signals["ocr"]["fields"]["gender"]["confidence"] = 0.40
        signals["ocr"]["fields"]["address"]["confidence"] = 0.55
        f4 = rule_low_field_ocr_confidence(signals)
        self.assertIsNotNone(f4)
        self.assertEqual(f4.points, 15)
        self.assertEqual(len(f4.evidence["low_confidence_fields"]), 4)

    def test_19_malformed_and_empty_signals(self) -> None:
        """Test 19: Malformed and empty signal inputs evaluate safely without exceptions."""
        malformed_cases = [
            None,
            {},
            {"document": None},
            {"ocr": "bad_type"},
            {"qr": []},
            {"cross_validation": 1234},
        ]

        for item in malformed_cases:
            with self.subTest(item=item):
                res = evaluate_rules(item)  # type: ignore[arg-type]
                self.assertIsInstance(res, dict)
                self.assertIn("findings", res)
                self.assertIn("total_points", res)
                self.assertIsInstance(res["total_points"], int)
                # Output must be JSON serializable
                json_str = json.dumps(res)
                self.assertIsInstance(json_str, str)

    def test_20_pii_safety_in_findings(self) -> None:
        """Test 20: PII safety: no raw Aadhaar numbers, names, DOBs, or full addresses appear in findings."""
        signals = copy.deepcopy(self.clean_signals)
        # Configure several rules to trigger
        signals["cross_validation"]["name"] = {"available": True, "similarity": 0.50, "match": False}
        signals["cross_validation"]["dob"] = {"available": True, "match": False, "comparison": "full"}
        signals["cross_validation"]["gender"] = {"available": True, "match": False}
        signals["ocr"]["fields"]["name"]["confidence"] = 0.45

        res = evaluate_rules(signals)
        serialized = json.dumps(res)

        # Check that no common PII placeholders or raw values exist in rule outputs
        self.assertNotIn("RAHUL KUMAR", serialized)
        self.assertNotIn("12/04/2002", serialized)
        self.assertNotIn("123 MG Road", serialized)
        self.assertNotIn("987654321098", serialized)

    def test_21_rule_output_json_serializability(self) -> None:
        """Test 21: Complete evaluate_rules output is strictly JSON serializable."""
        signals = copy.deepcopy(self.clean_signals)
        signals["document"] = {"type": "aadhaar", "confidence": 0.75}
        signals["ocr"]["confidence"] = 0.60

        res = evaluate_rules(signals)
        try:
            serialized = json.dumps(res)
            deserialized = json.loads(serialized)
            self.assertEqual(len(deserialized["findings"]), len(res["findings"]))
            self.assertEqual(deserialized["total_points"], res["total_points"])
        except TypeError as err:
            self.fail(f"evaluate_rules output is not JSON-serializable: {err}")

    def test_22_input_immutability(self) -> None:
        """Test 22: evaluate_rules never modifies the input signals dictionary."""
        original = copy.deepcopy(self.clean_signals)
        input_to_evaluate = copy.deepcopy(self.clean_signals)

        evaluate_rules(input_to_evaluate)

        self.assertEqual(input_to_evaluate, original)


if __name__ == "__main__":
    unittest.main()
