"""
Comprehensive unit tests for the Aadhaar Document Detection Layer.
Covers all Prompt 4 test requirements:
1. Strong Aadhaar evidence
2. Hindi evidence
3. Generic government document (false-positive prevention)
4. Random 12-digit number alone
5. Formatting / casing / whitespace normalization
6. QR supporting evidence
7. Insufficient / random text
8. Competing identity documents (PAN, Voter ID, Driving Licence)
9. Privacy masking in evidence
"""

import unittest

from document_processor.document_detector import (
    DocumentDetectionResult,
    detect_aadhaar,
    detect_aadhaar_document,
    detect_document_type,
    normalize_detection_text,
)


class TestAadhaarDocumentDetection(unittest.TestCase):

    # 1. Strong Aadhaar evidence
    def test_1_strong_aadhaar_evidence(self):
        sample = """
        Government of India
        Unique Identification Authority of India
        Aadhaar
        1234 5678 9012
        DOB: 12/04/2002
        Male
        """
        res = detect_document_type(ocr_text=sample)
        self.assertEqual(res.document_type, "aadhaar")
        self.assertEqual(res.status, "PASS")
        self.assertGreaterEqual(res.confidence, 0.70)
        self.assertIn("aadhaar_english_text", res.signals_detected)
        self.assertIn("aadhaar_number_pattern", res.signals_detected)

    # 2. Hindi evidence
    def test_2_hindi_aadhaar_evidence(self):
        sample = """
        भारत सरकार
        भारतीय विशिष्ट पहचान प्राधिकरण
        आधार
        मेरा आधार मेरी पहचान
        जन्म तिथि: 12/04/2002
        पुरुष
        """
        res = detect_document_type(ocr_text=sample)
        self.assertEqual(res.document_type, "aadhaar")
        self.assertEqual(res.status, "PASS")
        self.assertGreaterEqual(res.confidence, 0.60)
        self.assertIn("aadhaar_hindi_text", res.signals_detected)

    # 3. Generic government document (Must NOT cause confident Aadhaar PASS)
    def test_3_generic_government_document(self):
        sample = """
        Government of India
        भारत सरकार
        Name: John Doe
        Address: 123 Main Street, New Delhi
        Date: 01/01/2026
        """
        res = detect_document_type(ocr_text=sample)
        # Should be UNKNOWN or NOT_AADHAAR, definitely NOT PASS
        self.assertNotEqual(res.status, "PASS")
        self.assertIn(res.document_type, ["unknown", "not_aadhaar"])
        self.assertLess(res.confidence, 0.50)

    # 4. Random 12-digit number alone (Must NOT be sufficient)
    def test_4_random_12_digit_number_alone(self):
        sample = "123456789012"
        res = detect_document_type(ocr_text=sample)
        self.assertNotEqual(res.status, "PASS")
        self.assertIn(res.document_type, ["unknown", "not_aadhaar"])
        self.assertLess(res.confidence, 0.40)

        # Spaced 12-digit number without any Aadhaar keyword
        sample_spaced = "Phone bill transaction ID: 9876 5432 1098"
        res_spaced = detect_document_type(ocr_text=sample_spaced)
        self.assertNotEqual(res_spaced.status, "PASS")

    # 5. Formatting differences and normalization
    def test_5_formatting_normalization(self):
        self.assertEqual(normalize_detection_text("  AADHAAR  "), "AADHAAR")
        self.assertEqual(normalize_detection_text("Aadhaar\n\nCard"), "Aadhaar Card")
        self.assertEqual(normalize_detection_text("  आधार   "), "आधार")

        # Mixed casing detection
        res_upper = detect_document_type(ocr_text="AADHAAR CARD 1234 5678 9012")
        res_lower = detect_document_type(ocr_text="aadhaar card 1234 5678 9012")
        res_mixed = detect_document_type(ocr_text="  AadHaaR   Card 1234 5678 9012 ")

        self.assertEqual(res_upper.document_type, "aadhaar")
        self.assertEqual(res_lower.document_type, "aadhaar")
        self.assertEqual(res_mixed.document_type, "aadhaar")

    # 6. QR supporting evidence
    def test_6_qr_supporting_evidence(self):
        text_without_qr = "DOB: 12/04/2002\nMale\nUnique Identification"
        res_no_qr = detect_document_type(ocr_text=text_without_qr, qr_detected=False)

        res_with_qr = detect_document_type(ocr_text=text_without_qr, qr_detected=True)
        # QR presence should increase the confidence score
        self.assertGreater(res_with_qr.confidence, res_no_qr.confidence)
        self.assertIn("qr_present", res_with_qr.signals_detected)

        # Aadhaar QR payload should provide an even stronger confidence boost
        res_aadhaar_qr = detect_document_type(ocr_text=text_without_qr, qr_is_aadhaar=True)
        self.assertGreater(res_aadhaar_qr.confidence, res_with_qr.confidence)
        self.assertIn("aadhaar_qr_payload", res_aadhaar_qr.signals_detected)

    # 7. Insufficient / random text
    def test_7_insufficient_evidence(self):
        weak_sample = "Coffee Shop Receipt\nTotal: $12.50\nThank you for visiting"
        res = detect_document_type(ocr_text=weak_sample)
        self.assertEqual(res.document_type, "unknown")
        self.assertEqual(res.status, "UNKNOWN")
        self.assertLess(res.confidence, 0.20)

        # Empty string
        res_empty = detect_document_type(ocr_text="")
        self.assertEqual(res_empty.document_type, "unknown")
        self.assertEqual(res_empty.status, "UNKNOWN")
        self.assertEqual(res_empty.confidence, 0.0)

    # 8. Competing identity documents (PAN Card, Driving Licence, Voter ID, Passport)
    def test_8_competing_identity_documents(self):
        # PAN Card
        pan_text = """
        INCOME TAX DEPARTMENT
        GOVT. OF INDIA
        Permanent Account Number
        ABCDE1234F
        Name: RAHUL KUMAR
        Father's Name: RAJESH KUMAR
        DOB: 12/04/2002
        """
        res_pan = detect_document_type(ocr_text=pan_text)
        self.assertEqual(res_pan.document_type, "not_aadhaar")
        self.assertEqual(res_pan.status, "FAIL")
        self.assertIn("competing_doc:pan_card", res_pan.signals_detected)

        # Driving Licence
        dl_text = "Transport Department\nDriving Licence\nDL-1420110012345\nValid Till: 2035"
        res_dl = detect_document_type(ocr_text=dl_text)
        self.assertEqual(res_dl.document_type, "not_aadhaar")
        self.assertEqual(res_dl.status, "FAIL")

        # Voter ID
        voter_text = "Election Commission of India\nElector Photo Identity Card\nEPIC NO: ABC1234567"
        res_voter = detect_document_type(ocr_text=voter_text)
        self.assertEqual(res_voter.document_type, "not_aadhaar")
        self.assertEqual(res_voter.status, "FAIL")

    # 9. Privacy & Masking Verification
    def test_9_privacy_and_masking(self):
        sample = "Aadhaar Card\nNumber: 9876 5432 1098\nUnique Identification Authority"
        res = detect_document_type(ocr_text=sample)

        # Check evidence items for proper masking
        for ev in res.evidence:
            if ev.get("signal") == "aadhaar_number_pattern" and ev.get("detected"):
                masked = ev.get("masked_pattern", "")
                self.assertNotIn("9876", masked)
                self.assertIn("1098", masked)
                self.assertTrue(masked.startswith("XXXX"))

    # 10. Convenience helpers and dictionary serialization
    def test_10_convenience_aliases_and_serialization(self):
        sample = "Aadhaar 1234 5678 9012"
        res_obj = detect_aadhaar(ocr_text=sample)
        self.assertIsInstance(res_obj, DocumentDetectionResult)
        self.assertEqual(res_obj.status, "PASS")

        dict_res = detect_aadhaar_document(sample)
        self.assertIsInstance(dict_res, dict)
        self.assertEqual(dict_res["type"], "aadhaar")
        self.assertEqual(dict_res["document_type"], "aadhaar")
        self.assertEqual(dict_res["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
