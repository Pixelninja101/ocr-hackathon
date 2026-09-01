"""
Tests for normalization module: Name, DOB precision, Gender handling.
"""

import unittest
from document_processor.ocr.normalization import (
    format_dob_for_display,
    normalize_dob,
    normalize_gender,
    normalize_name,
)


class TestNormalization(unittest.TestCase):

    def test_name_normalization(self):
        self.assertEqual(normalize_name(" Rahul  Kumar "), "rahul kumar")
        self.assertEqual(normalize_name("RAHUL KUMAR"), "rahul kumar")
        self.assertEqual(normalize_name("Mr. Rahul Kumar-Sharma"), "mr rahul kumar sharma")
        self.assertEqual(normalize_name(""), "")
        self.assertEqual(normalize_name(None), "")

    def test_gender_normalization_english(self):
        self.assertEqual(normalize_gender("Male"), "MALE")
        self.assertEqual(normalize_gender("MALE"), "MALE")
        self.assertEqual(normalize_gender("M"), "MALE")
        self.assertEqual(normalize_gender("Female"), "FEMALE")
        self.assertEqual(normalize_gender("FEMALE"), "FEMALE")
        self.assertEqual(normalize_gender("F"), "FEMALE")
        self.assertEqual(normalize_gender("Transgender"), "TRANSGENDER")

    def test_gender_normalization_hindi(self):
        self.assertEqual(normalize_gender("पुरुष"), "MALE")
        self.assertEqual(normalize_gender("पु०"), "MALE")
        self.assertEqual(normalize_gender("महिला"), "FEMALE")
        self.assertEqual(normalize_gender("स्त्री"), "FEMALE")
        self.assertEqual(normalize_gender("तृतीय लिंग"), "TRANSGENDER")

    def test_dob_normalization_full_date(self):
        res = normalize_dob("12/04/2002")
        self.assertIsNotNone(res)
        self.assertEqual(res["year"], 2002)
        self.assertEqual(res["month"], 4)
        self.assertEqual(res["day"], 12)
        self.assertEqual(res["precision"], "full")
        self.assertEqual(format_dob_for_display(res), "12/04/2002")

        res_dash = normalize_dob("05-11-1998")
        self.assertIsNotNone(res_dash)
        self.assertEqual(res_dash["year"], 1998)
        self.assertEqual(res_dash["month"], 11)
        self.assertEqual(res_dash["day"], 5)
        self.assertEqual(res_dash["precision"], "full")

    def test_dob_normalization_year_only(self):
        res = normalize_dob("2002")
        self.assertIsNotNone(res)
        self.assertEqual(res["year"], 2002)
        self.assertIsNone(res["month"])
        self.assertIsNone(res["day"])
        self.assertEqual(res["precision"], "year")
        self.assertEqual(format_dob_for_display(res), "2002")


if __name__ == "__main__":
    unittest.main()
