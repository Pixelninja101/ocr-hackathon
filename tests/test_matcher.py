"""
Tests for verification/matcher module: fuzzy name matching, precision-aware DOB matching, gender validation.
"""

import unittest
from document_processor.verification.matcher import (
    cross_validate_ocr_and_qr,
    match_dob,
    match_gender,
    match_names,
)


class TestMatcher(unittest.TestCase):

    def test_fuzzy_name_matching_ocr_typo(self):
        # PRD Test 4: OCR typo "RAHUI KUMAR" vs QR "RAHUL KUMAR"
        res = match_names("RAHUI KUMAR", "RAHUL KUMAR")
        self.assertIsNotNone(res)
        self.assertGreaterEqual(res["similarity"], 0.85)
        self.assertTrue(res["match"])

    def test_name_mismatch(self):
        # PRD Test 6: "RAHUL KUMAR" vs "AMIT KUMAR"
        res = match_names("RAHUL KUMAR", "AMIT KUMAR")
        self.assertIsNotNone(res)
        self.assertLess(res["similarity"], 0.85)
        self.assertFalse(res["match"])

    def test_dob_matching_year_only(self):
        # PRD Test 5: OCR year "2002" vs QR full date "12/04/2002"
        res = match_dob("2002", "12/04/2002")
        self.assertIsNotNone(res)
        self.assertTrue(res["match"])
        self.assertEqual(res["comparison"], "year")

    def test_dob_matching_full_date(self):
        res = match_dob("12/04/2002", "12/04/2002")
        self.assertIsNotNone(res)
        self.assertTrue(res["match"])
        self.assertEqual(res["comparison"], "full")

    def test_dob_mismatch(self):
        res = match_dob("12/04/2001", "12/04/2002")
        self.assertIsNotNone(res)
        self.assertFalse(res["match"])

    def test_gender_matching(self):
        self.assertTrue(match_gender("MALE", "MALE")["match"])
        self.assertTrue(match_gender("पुरुष", "MALE")["match"])
        self.assertFalse(match_gender("FEMALE", "MALE")["match"])

    def test_full_cross_validation(self):
        ocr_fields = {
            "name": {"value": "RAHUI KUMAR", "confidence": 0.92},
            "dob": {"year": 2002, "month": None, "day": None, "precision": "year", "confidence": 0.90},
            "gender": {"value": "MALE", "confidence": 0.95},
        }
        qr_fields = {
            "name": "RAHUL KUMAR",
            "dob": "12/04/2002",
            "gender": "MALE",
        }
        cross_val = cross_validate_ocr_and_qr(ocr_fields, qr_fields)
        self.assertIsNotNone(cross_val)
        self.assertTrue(cross_val["name"]["match"])
        self.assertGreaterEqual(cross_val["name"]["similarity"], 0.85)
        self.assertTrue(cross_val["dob"]["match"])
        self.assertEqual(cross_val["dob"]["comparison"], "year")
        self.assertTrue(cross_val["gender"]["match"])


if __name__ == "__main__":
    unittest.main()
