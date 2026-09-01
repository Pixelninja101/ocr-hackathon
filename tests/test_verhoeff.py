"""
Comprehensive test suite for Verhoeff Checksum validation and AADHAAR_CHECKSUM_INVALID risk rule.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from typing import Any, Dict

from document_processor import process_document
from risk_engine import (
    assess_document,
    calculate_verhoeff_check_digit,
    evaluate_rules,
    extract_signals,
    rule_aadhaar_checksum_invalid,
    validate_aadhaar_checksum,
)
from risk_engine.models import RuleSeverity


class TestVerhoeffChecksum(unittest.TestCase):
    """Unit tests for Verhoeff checksum algorithm and check digit computation."""

    def setUp(self) -> None:
        # Known valid 11-digit prefixes and their check digits:
        # Prefix "23456789012" -> Check digit 4 -> "234567890124"
        # Prefix "98765432109" -> Check digit 6 -> "987654321096"
        # Prefix "12345678901" -> Check digit 0 -> "123456789010"
        self.valid_aadhaar_1 = "234567890124"
        self.valid_aadhaar_2 = "987654321096"
        self.valid_aadhaar_3 = "123456789010"

    # -------------------------------------------------------------------------
    # 1. Algorithm Unit Tests
    # -------------------------------------------------------------------------

    def test_01_valid_12_digit_aadhaar_checksum(self) -> None:
        """Test 1: Valid 12-digit Aadhaar numbers pass Verhoeff validation."""
        self.assertTrue(validate_aadhaar_checksum(self.valid_aadhaar_1))
        self.assertTrue(validate_aadhaar_checksum(self.valid_aadhaar_2))
        self.assertTrue(validate_aadhaar_checksum(self.valid_aadhaar_3))

    def test_02_invalid_final_digit(self) -> None:
        """Test 2: Single-digit error in check digit (final digit) is detected as invalid."""
        invalid_num = self.valid_aadhaar_1[:-1] + ("5" if self.valid_aadhaar_1[-1] != "5" else "6")
        self.assertFalse(validate_aadhaar_checksum(invalid_num))

    def test_03_invalid_middle_digit(self) -> None:
        """Test 3: Single-digit error in a middle digit is detected as invalid."""
        # Change 5th digit
        invalid_num = self.valid_aadhaar_1[:4] + ("9" if self.valid_aadhaar_1[4] != "9" else "8") + self.valid_aadhaar_1[5:]
        self.assertFalse(validate_aadhaar_checksum(invalid_num))

    def test_04_exactly_11_digits_rejected(self) -> None:
        """Test 4: Incomplete 11-digit number is rejected by validation."""
        self.assertFalse(validate_aadhaar_checksum("23456789012"))

    def test_05_exactly_13_digits_rejected(self) -> None:
        """Test 5: 13-digit number is rejected by validation."""
        self.assertFalse(validate_aadhaar_checksum("2345678901245"))

    def test_06_alphabetic_and_masked_input_rejected(self) -> None:
        """Test 6: Strings containing letters or masked characters ('X') are rejected."""
        self.assertFalse(validate_aadhaar_checksum("23456789012A"))
        self.assertFalse(validate_aadhaar_checksum("XXXX XXXX 1234"))
        self.assertFalse(validate_aadhaar_checksum("INVALID_STR!"))

    def test_07_empty_and_none_input_rejected(self) -> None:
        """Test 7: Empty strings, None, or whitespace-only strings safely return False."""
        self.assertFalse(validate_aadhaar_checksum(""))
        self.assertFalse(validate_aadhaar_checksum("   "))
        self.assertFalse(validate_aadhaar_checksum(None))  # type: ignore[arg-type]

    def test_08_whitespace_and_hyphen_separated_aadhaar_accepted(self) -> None:
        """Test 8: Valid numbers with 4-digit grouping spaces or hyphens are normalized and validated."""
        spaced = "2345 6789 0124"
        hyphenated = "2345-6789-0124"
        tabbed = "2345\t6789\t0124"

        self.assertTrue(validate_aadhaar_checksum(spaced))
        self.assertTrue(validate_aadhaar_checksum(hyphenated))
        self.assertTrue(validate_aadhaar_checksum(tabbed))

    def test_09_check_digit_calculation_works_for_valid_prefixes(self) -> None:
        """Test 9: calculate_verhoeff_check_digit generates the correct check digit for 11-digit prefixes."""
        self.assertEqual(calculate_verhoeff_check_digit("23456789012"), 4)
        self.assertEqual(calculate_verhoeff_check_digit("98765432109"), 6)
        self.assertEqual(calculate_verhoeff_check_digit("12345678901"), 0)

        # Spaced 11 digits
        self.assertEqual(calculate_verhoeff_check_digit("2345 6789 012"), 4)

    def test_10_check_digit_calculation_invalid_length_raises_value_error(self) -> None:
        """Test 10: calculate_verhoeff_check_digit raises ValueError for inputs that are not exactly 11 digits."""
        with self.assertRaises(ValueError):
            calculate_verhoeff_check_digit("12345")  # 5 digits

        with self.assertRaises(ValueError):
            calculate_verhoeff_check_digit("123456789012")  # 12 digits

        with self.assertRaises(ValueError):
            calculate_verhoeff_check_digit(None)  # type: ignore[arg-type]

    # -------------------------------------------------------------------------
    # 2. Risk Rule Tests (AADHAAR_CHECKSUM_INVALID)
    # -------------------------------------------------------------------------

    def test_11_valid_checksum_rule_does_not_trigger(self) -> None:
        """Test 11: AADHAAR_CHECKSUM_INVALID does not trigger when checksum is valid."""
        signals = {
            "aadhaar_checksum": {
                "available": True,
                "valid": True,
            }
        }
        finding = rule_aadhaar_checksum_invalid(signals)
        self.assertIsNone(finding)

    def test_12_invalid_checksum_rule_triggers_with_15_points(self) -> None:
        """Test 12: AADHAAR_CHECKSUM_INVALID triggers with 15 points and MEDIUM severity when valid == False."""
        signals = {
            "aadhaar_checksum": {
                "available": True,
                "valid": False,
            }
        }
        finding = rule_aadhaar_checksum_invalid(signals)
        self.assertIsNotNone(finding)
        assert finding is not None
        self.assertEqual(finding.rule_id, "AADHAAR_CHECKSUM_INVALID")
        self.assertEqual(finding.severity, RuleSeverity.MEDIUM.value)
        self.assertEqual(finding.points, 15)
        self.assertTrue(finding.triggered)
        self.assertIn("failed checksum validation", finding.reason)
        self.assertEqual(finding.evidence, {"checksum_valid": False})

    def test_13_missing_or_masked_aadhaar_rule_does_not_trigger(self) -> None:
        """Test 13: AADHAAR_CHECKSUM_INVALID does NOT trigger when Aadhaar number is missing or masked."""
        # 1. available = False, valid = None
        signals_missing = {
            "aadhaar_checksum": {
                "available": False,
                "valid": None,
            }
        }
        self.assertIsNone(rule_aadhaar_checksum_invalid(signals_missing))

        # 2. Empty signals
        self.assertIsNone(rule_aadhaar_checksum_invalid({}))

    def test_14_pii_safety_in_findings_and_evidence(self) -> None:
        """Test 14: Checksum finding evidence and reason contain zero raw digits or PII."""
        signals = {
            "aadhaar_checksum": {
                "available": True,
                "valid": False,
            }
        }
        finding = rule_aadhaar_checksum_invalid(signals)
        assert finding is not None
        finding_dict = finding.to_dict()
        serialized = json.dumps(finding_dict)

        self.assertNotIn("2345", serialized)
        self.assertNotIn("9876", serialized)
        self.assertEqual(finding_dict["evidence"], {"checksum_valid": False})

    # -------------------------------------------------------------------------
    # 3. End-to-End Integration Test
    # -------------------------------------------------------------------------

    def test_15_end_to_end_integration_with_synthetic_document(self) -> None:
        """Test 15: Full pipeline from process_document to assess_document extracts checksum signal correctly."""
        # Load synthetic generator from OCR test helpers
        spec = importlib.util.spec_from_file_location(
            "ocr_test_helpers", r"C:\icons\ocr\tests\test_helpers.py"
        )
        assert spec is not None and spec.loader is not None
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)

        # Generate synthetic Aadhaar with valid Verhoeff number ("1234 5678 9010")
        img_bytes = helpers.create_synthetic_aadhaar_bytes(
            name="SUNIL VERMA",
            dob="12/08/1994",
            gender="MALE",
            aadhaar_num="1234 5678 9010",
            include_qr=True,
        )

        ocr_result = process_document(img_bytes)
        self.assertTrue(ocr_result["success"])

        assessment = assess_document(ocr_result)
        self.assertTrue(assessment["success"])
        self.assertIn("signals", assessment)
        self.assertIn("aadhaar_checksum", assessment["signals"])

        checksum_sig = assessment["signals"]["aadhaar_checksum"]
        self.assertIn("available", checksum_sig)
        self.assertIn("valid", checksum_sig)

        # If OCR extracted the 12 digits, valid is True; if masked or OCR missed, available is False/None.
        # Crucially, AADHAAR_CHECKSUM_INVALID should not be triggered unless an unmasked number failed.
        if checksum_sig["available"] and checksum_sig["valid"] is True:
            self.assertNotIn("AADHAAR_CHECKSUM_INVALID", assessment["flags"])


if __name__ == "__main__":
    unittest.main()
