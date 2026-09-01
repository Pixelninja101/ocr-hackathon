"""
Unit tests for Aadhaar OCR Field Extraction Layer.
Covers all requirements from Prompt 6:
1. Full DOB (YYYY-MM-DD + Year of Birth)
2. Year-only DOB (DOB is None, Year of Birth is present)
3. Gender (English and Devanagari normalization)
4. Aadhaar number candidate normalization and privacy masking
5. Invalid/unrelated 12-digit numbers (uncertainty handling)
6. English name extraction using contextual layout
7. Hindi name extraction (Unicode Devanagari)
8. Multi-line address block extraction
9. Bilingual duplicate fields consolidation
10. Missing fields (explicit NOT_FOUND status)
11. Low-confidence extraction (UNCERTAIN status)
12. Empty OCR input safety
"""

import unittest

from document_processor.ocr.engine import OCRResult
from document_processor.ocr.fields import (
    AadhaarFieldsResult,
    ExtractedField,
    extract_aadhaar_fields,
    extract_aadhaar_number_from_text,
    extract_address_block,
    extract_all_fields_from_ocr,
    extract_dob_from_text,
    extract_gender_from_text,
    extract_name_from_lines,
)


class TestOCRFields(unittest.TestCase):

    # 1. Full DOB Extraction
    def test_1_full_dob_extraction(self):
        sample = "Government of India\nDate of Birth: 12/04/2002\nGender: Male"
        res = extract_aadhaar_fields(sample)
        self.assertEqual(res.fields["dob"].value, "2002-04-12")
        self.assertEqual(res.fields["dob"].status, "FOUND")
        self.assertEqual(res.fields["year_of_birth"].value, "2002")
        self.assertEqual(res.fields["dob"].metadata["precision"], "full")

        # Hindi date test
        sample_hi = "भारतीय विशिष्ट पहचान प्राधिकरण\nजन्म तिथि: 15/08/1995\nपुरुष"
        res_hi = extract_aadhaar_fields(sample_hi)
        self.assertEqual(res_hi.fields["dob"].value, "1995-08-15")
        self.assertEqual(res_hi.fields["year_of_birth"].value, "1995")

    # 2. Year-Only DOB Extraction (Must NOT invent day/month)
    def test_2_year_only_dob(self):
        sample = "Government of India\nYear of Birth: 2002\nGender: Female"
        res = extract_aadhaar_fields(sample)
        self.assertIsNone(res.fields["dob"].value)
        self.assertEqual(res.fields["dob"].metadata.get("precision"), "year")
        self.assertEqual(res.fields["year_of_birth"].value, "2002")
        self.assertEqual(res.fields["year_of_birth"].status, "FOUND")

        # Hindi year test
        sample_hi = "जन्म वर्ष: 1988\nमहिला"
        res_hi = extract_aadhaar_fields(sample_hi)
        self.assertIsNone(res_hi.fields["dob"].value)
        self.assertEqual(res_hi.fields["year_of_birth"].value, "1988")

    # 3. Gender Extraction & Normalization
    def test_3_gender_normalization(self):
        # English Male & Female
        res_m = extract_aadhaar_fields("Name: RAJESH\nGender: MALE")
        self.assertEqual(res_m.fields["gender"].value, "MALE")

        res_f = extract_aadhaar_fields("Name: PRIYA\nSex: FEMALE")
        self.assertEqual(res_f.fields["gender"].value, "FEMALE")

        # Hindi Male & Female
        res_hi_m = extract_aadhaar_fields("लिंग: पुरुष")
        self.assertEqual(res_hi_m.fields["gender"].value, "MALE")

        res_hi_f = extract_aadhaar_fields("लिंग: महिला")
        self.assertEqual(res_hi_f.fields["gender"].value, "FEMALE")

    # 4. Aadhaar Number Formatting and Privacy Masking
    def test_4_aadhaar_number_formatting_and_masking(self):
        sample = "Aadhaar Card\n1234 5678 9012\nVID: 9876543210987654"
        res = extract_aadhaar_fields(sample)
        uid_field = res.fields["aadhaar_number"]
        self.assertEqual(uid_field.value, "123456789012")
        self.assertEqual(uid_field.status, "FOUND")
        self.assertEqual(uid_field.metadata["masked_value"], "XXXX XXXX 9012")

        # Masked input
        sample_masked = "Aadhaar No: XXXX XXXX 4321"
        res_masked = extract_aadhaar_fields(sample_masked)
        self.assertEqual(res_masked.fields["aadhaar_number"].metadata["masked_value"], "XXXX XXXX 4321")

    # 5. Invalid / Unrelated 12-digit number (uncertainty handling)
    def test_5_invalid_12_digit_number(self):
        # Continuous number without any Aadhaar keyword
        sample = "Order Invoice #987654321098\nTotal: 450.00"
        res = extract_aadhaar_fields(sample, overall_confidence_override=0.50)
        uid_field = res.fields["aadhaar_number"]
        self.assertEqual(uid_field.status, "UNCERTAIN")
        self.assertLess(uid_field.confidence, 0.60)

    # 6. Name Extraction (English)
    def test_6_english_name_extraction(self):
        sample = """
        Government of India
        Unique Identification Authority of India
        RAHUL KUMAR
        DOB: 12/04/2002
        MALE
        1234 5678 9012
        """
        res = extract_aadhaar_fields(sample)
        self.assertEqual(res.fields["name"].value, "RAHUL KUMAR")
        self.assertEqual(res.fields["name"].status, "FOUND")
        self.assertGreaterEqual(res.fields["name"].confidence, 0.70)

    # 7. Hindi Name Extraction
    def test_7_hindi_name_extraction(self):
        sample = """
        भारत सरकार
        भारतीय विशिष्ट पहचान प्राधिकरण
        राहुल कुमार
        जन्म तिथि: 12/04/2002
        पुरुष
        1234 5678 9012
        """
        res = extract_aadhaar_fields(sample)
        self.assertEqual(res.fields["name"].value, "राहुल कुमार")
        self.assertEqual(res.fields["name"].metadata.get("hindi"), "राहुल कुमार")
        self.assertEqual(res.fields["name"].status, "FOUND")

    # 8. Address Block Extraction
    def test_8_address_block_extraction(self):
        sample = """
        Address:
        House No 123, Block B
        Green Park Main, South Delhi
        New Delhi - 110016
        1234 5678 9012
        www.uidai.gov.in
        """
        res = extract_aadhaar_fields(sample)
        self.assertIsNotNone(res.fields["address"].value)
        self.assertIn("Green Park Main", res.fields["address"].value)
        self.assertIn("110016", res.fields["address"].value)
        # Should stop before UIDAI website or number
        self.assertNotIn("www.uidai.gov.in", res.fields["address"].value)

    # 9. Bilingual Fields Handling
    def test_9_bilingual_fields_consolidation(self):
        sample = """
        NAME: RAHUL KUMAR
        नाम: राहुल कुमार
        DOB: 12/04/2002
        जन्म तिथि: 12/04/2002
        GENDER: MALE
        लिंग: पुरुष
        1234 5678 9012
        """
        res = extract_aadhaar_fields(sample)
        self.assertEqual(res.fields["name"].value, "RAHUL KUMAR")
        self.assertEqual(res.fields["name"].metadata.get("english"), "RAHUL KUMAR")
        self.assertEqual(res.fields["name"].metadata.get("hindi"), "राहुल कुमार")
        self.assertEqual(res.fields["dob"].value, "2002-04-12")
        self.assertEqual(res.fields["gender"].value, "MALE")

    # 10. Missing Fields Return NOT_FOUND
    def test_10_missing_fields_not_found(self):
        sample = "DOB: 12/04/2002\n1234 5678 9012"
        res = extract_aadhaar_fields(sample)
        self.assertIsNone(res.fields["name"].value)
        self.assertEqual(res.fields["name"].status, "NOT_FOUND")
        self.assertIsNone(res.fields["gender"].value)
        self.assertEqual(res.fields["gender"].status, "NOT_FOUND")
        self.assertIsNone(res.fields["address"].value)
        self.assertEqual(res.fields["address"].status, "NOT_FOUND")

    # 11. Low-Confidence / Uncertain Extraction
    def test_11_low_confidence_uncertainty(self):
        sample = "Candidate Name Maybe\nDOB: 12/04/2002"
        res = extract_aadhaar_fields(sample, overall_confidence_override=0.40)
        self.assertEqual(res.fields["name"].status, "UNCERTAIN")
        self.assertLess(res.fields["name"].confidence, 0.60)

    # 12. Empty OCR Result Safety
    def test_12_empty_ocr_safety(self):
        res_empty = extract_aadhaar_fields("")
        self.assertTrue(res_empty.success)
        for f in res_empty.fields.values():
            self.assertIsNone(f.value)
            self.assertEqual(f.status, "NOT_FOUND")
        self.assertGreater(len(res_empty.warnings), 0)

    # 13. Backwards Compatibility with extract_all_fields_from_ocr
    def test_13_legacy_schema_compatibility(self):
        raw_ocr = """
        Government of India
        Unique Identification Authority of India
        Rahul Kumar
        DOB: 12/04/2002
        MALE
        9876 5432 1098
        """
        extracted = extract_all_fields_from_ocr(raw_ocr, overall_ocr_confidence=0.94)
        self.assertEqual(extracted["name"]["value"], "RAHUL KUMAR")
        self.assertEqual(extracted["dob"]["year"], 2002)
        self.assertEqual(extracted["dob"]["precision"], "full")
        self.assertEqual(extracted["gender"]["value"], "MALE")
        self.assertEqual(extracted["aadhaar_number"]["value"], "9876 5432 1098")

    # 14. Valid explicitly labelled DOB
    def test_14_valid_explicitly_labelled_dob(self):
        samples = [
            ("DOB: 12/04/2002", "2002-04-12", "2002"),
            ("Date of Birth: 15/08/1995", "1995-08-15", "1995"),
            ("जन्म तिथि: 01/01/2000", "2000-01-01", "2000"),
            ("जन्म तारीख: 25/12/1998", "1998-12-25", "1998"),
        ]
        for text, exp_dob, exp_yob in samples:
            res = extract_aadhaar_fields(text)
            self.assertEqual(res.fields["dob"].value, exp_dob)
            self.assertEqual(res.fields["dob"].status, "FOUND")
            self.assertEqual(res.fields["year_of_birth"].value, exp_yob)

    # 15. Valid explicitly labelled YOB
    def test_15_valid_explicitly_labelled_yob(self):
        samples = [
            ("YOB: 1990", "1990"),
            ("Year of Birth: 1985", "1985"),
            ("जन्म वर्ष: 1978", "1978"),
            ("जन्म का वर्ष: 1992", "1992"),
        ]
        for text, exp_yob in samples:
            res = extract_aadhaar_fields(text)
            self.assertIsNone(res.fields["dob"].value)
            self.assertEqual(res.fields["year_of_birth"].value, exp_yob)
            self.assertEqual(res.fields["year_of_birth"].status, "FOUND")

    # 16. "Details as on" date rejected
    def test_16_details_as_on_date_rejected(self):
        sample = """
        भारतीय विशिष्ट पहचान प्राधिकरण
        Unique Identification Authority of India
        पता:
        C/O: Sumit Agrawal, ward no 11 shah bazar, Burhanpur
        Details as on: 30/12/2025
        """
        res = extract_aadhaar_fields(sample)
        self.assertIsNone(res.fields["dob"].value)
        self.assertEqual(res.fields["dob"].status, "NOT_FOUND")
        self.assertIsNone(res.fields["year_of_birth"].value)
        self.assertEqual(res.fields["year_of_birth"].status, "NOT_FOUND")

    # 17. "Generated on" date rejected
    def test_17_generated_on_date_rejected(self):
        sample = """
        Government of India
        Unique Identification Authority of India
        Card generated on: 15/05/2024
        """
        res = extract_aadhaar_fields(sample)
        self.assertIsNone(res.fields["dob"].value)
        self.assertEqual(res.fields["dob"].status, "NOT_FOUND")
        self.assertIsNone(res.fields["year_of_birth"].value)
        self.assertEqual(res.fields["year_of_birth"].status, "NOT_FOUND")

    # 18. "Issue Date" rejected
    def test_18_issue_date_rejected(self):
        sample_en = "Unique Identification Authority of India\nIssue Date: 12/08/2024"
        res_en = extract_aadhaar_fields(sample_en)
        self.assertIsNone(res_en.fields["dob"].value)
        self.assertEqual(res_en.fields["dob"].status, "NOT_FOUND")

        sample_doi = "Unique Identification Authority of India\nDate of Issue: 01/06/2023"
        res_doi = extract_aadhaar_fields(sample_doi)
        self.assertIsNone(res_doi.fields["dob"].value)
        self.assertEqual(res_doi.fields["dob"].status, "NOT_FOUND")

        sample_hi = "भारतीय विशिष्ट पहचान प्राधिकरण\nजारी करने की तिथि: 12/08/2024"
        res_hi = extract_aadhaar_fields(sample_hi)
        self.assertIsNone(res_hi.fields["dob"].value)
        self.assertEqual(res_hi.fields["dob"].status, "NOT_FOUND")

    # 19. "Date of Download" and "Printed on" rejected
    def test_19_date_of_download_and_printed_on_rejected(self):
        sample_dl = "Unique Identification Authority of India\nDate of Download: 05/02/2024"
        res_dl = extract_aadhaar_fields(sample_dl)
        self.assertIsNone(res_dl.fields["dob"].value)
        self.assertEqual(res_dl.fields["dob"].status, "NOT_FOUND")

        sample_pr = "Unique Identification Authority of India\nPrinted on: 22/03/2024"
        res_pr = extract_aadhaar_fields(sample_pr)
        self.assertIsNone(res_pr.fields["dob"].value)
        self.assertEqual(res_pr.fields["dob"].status, "NOT_FOUND")

    # 20. Text containing only an unrelated date returns NOT_FOUND
    def test_20_unrelated_date_only_returns_not_found(self):
        sample = "Invoice #98765 generated for client on 14/07/2021 by ABC Corp."
        res = extract_aadhaar_fields(sample)
        self.assertIsNone(res.fields["dob"].value)
        self.assertEqual(res.fields["dob"].status, "NOT_FOUND")
        self.assertIsNone(res.fields["year_of_birth"].value)
        self.assertEqual(res.fields["year_of_birth"].status, "NOT_FOUND")

    # 21. Multiline Aadhaar address
    def test_21_multiline_aadhaar_address(self):
        sample = """
        Address:
        Flat 402, Shanti Heights
        Near City Mall, Sector 15
        Rohini, North West Delhi
        Delhi - 110085
        1234 5678 9012
        """
        res = extract_aadhaar_fields(sample)
        addr = res.fields["address"].value
        self.assertIsNotNone(addr)
        self.assertIn("Flat 402", addr)
        self.assertIn("Shanti Heights", addr)
        self.assertIn("Rohini", addr)
        self.assertIn("110085", addr)

    # 22. Address containing D/O
    def test_22_address_containing_do_prefix(self):
        sample = """
        D/O: Ramesh Sharma,
        Plot No 45, Gandhi Nagar,
        Jaipur, Rajasthan - 302015
        1234 5678 9012
        """
        res = extract_aadhaar_fields(sample)
        addr = res.fields["address"].value
        self.assertIsNotNone(addr)
        self.assertIn("D/O: Ramesh Sharma", addr)
        self.assertIn("Gandhi Nagar", addr)
        self.assertIn("302015", addr)

    # 23. Address containing C/O
    def test_23_address_containing_co_prefix(self):
        sample = """
        C/O: Suresh Kumar,
        H.No 12-A, Model Town,
        Ludhiana, Punjab - 141002
        1234 5678 9012
        """
        res = extract_aadhaar_fields(sample)
        addr = res.fields["address"].value
        self.assertIsNotNone(addr)
        self.assertIn("C/O: Suresh Kumar", addr)
        self.assertIn("Model Town", addr)
        self.assertIn("141002", addr)

    # 24. Address containing VTC, PO, Sub District, District, State, PIN Code
    def test_24_address_containing_vtc_po_subdistrict_district_state_pincode(self):
        sample = """
        D/O: Shailendra Singh Tomar,
        mig. - 293,
        darpan colony,
        thatipur,
        gwalior,
        VTC: Gwalior,
        PO: R.k Puri Gwalior,
        Sub District: Gird,
        District: Gwalior,
        State: Madhya Pradesh,
        PIN Code: 474011
        =o 5190595
        आपका आधार क्रमांक / Your Aadhaar No. :
        9876 5432 1098
        """
        res = extract_aadhaar_fields(sample)
        addr = res.fields["address"].value
        self.assertIsNotNone(addr)
        self.assertIn("D/O: Shailendra Singh Tomar", addr)
        self.assertIn("mig. - 293", addr)
        self.assertIn("darpan colony", addr)
        self.assertIn("thatipur", addr)
        self.assertIn("VTC: Gwalior", addr)
        self.assertIn("PO: R.k Puri Gwalior", addr)
        self.assertIn("Sub District: Gird", addr)
        self.assertIn("District: Gwalior", addr)
        self.assertIn("State: Madhya Pradesh", addr)
        self.assertIn("PIN Code: 474011", addr)
        self.assertNotIn("Your Aadhaar No", addr)

    # 25. Address termination when unrelated footer begins
    def test_25_address_termination_at_unrelated_footer(self):
        sample = """
        Address:
        House 10, MG Road, Pune, Maharashtra - 411001
        Aadhaar is proof of identity, not of citizenship.
        www.uidai.gov.in
        """
        res = extract_aadhaar_fields(sample)
        addr = res.fields["address"].value
        self.assertIsNotNone(addr)
        self.assertIn("House 10, MG Road, Pune", addr)
        self.assertNotIn("Aadhaar is proof of identity", addr)
        self.assertNotIn("www.uidai.gov.in", addr)

    # 26. Hindi + English bilingual address
    def test_26_hindi_english_bilingual_address(self):
        sample = """
        पता:
        द्वारा: सुमित अग्रवाल, वॉर्ड नं 11, शाह बाज़ार, बुरहानपुर
        C/O: Sumit Agrawal, ward no 11, shah bazar, Burhanpur
        Madhya Pradesh - 450331
        Details as on: 30/12/2025
        """
        res = extract_aadhaar_fields(sample)
        addr = res.fields["address"].value
        self.assertIsNotNone(addr)
        self.assertIn("सुमित अग्रवाल", addr)
        self.assertIn("Sumit Agrawal", addr)
        self.assertIn("450331", addr)
        self.assertNotIn("Details as on", addr)

    # 27. Ensure guardian name is not returned as Aadhaar holder name
    def test_27_guardian_name_not_returned_as_holder_name(self):
        sample = """
        भारत सरकार
        Government of India
        To
        अर्न॑वी तोमर
        Amavi Tomar
        D/O: Shailendra Singh Tomar,
        mig. - 293, darpan colony,
        Gwalior, Madhya Pradesh - 474011
        9876 5432 1098
        """
        res = extract_aadhaar_fields(sample)
        name_val = res.fields["name"].value
        self.assertEqual(name_val, "AMAVI TOMAR")
        self.assertNotEqual(name_val, "SHAILENDRA SINGH TOMAR")
        self.assertNotIn("D/O", name_val)

    # 28. Front-side guardian line is not falsely classified as postal address
    def test_28_front_side_guardian_line_not_treated_as_address(self):
        sample = """
        Government of India
        Unique Identification Authority of India
        S/O: RAJESH KUMAR
        RAHUL KUMAR
        DOB: 12/04/2002
        MALE
        1234 5678 9012
        """
        res = extract_aadhaar_fields(sample)
        self.assertEqual(res.fields["name"].value, "RAHUL KUMAR")
        self.assertEqual(res.fields["dob"].value, "2002-04-12")
        self.assertEqual(res.fields["gender"].value, "MALE")
        self.assertEqual(res.fields["aadhaar_number"].metadata.get("masked_value"), "XXXX XXXX 9012")
        self.assertIsNone(res.fields["address"].value)
        self.assertEqual(res.fields["address"].status, "NOT_FOUND")

    # 29. Front-side bilingual DOB and Gender delimiters
    def test_29_front_side_bilingual_dob_and_gender(self):
        sample = """
        भारत सरकार / Government of India
        भारतीय विशिष्ट पहचान प्राधिकरण / UIDAI
        राहुल कुमार
        TEST PERSON
        जन्म तिथि / DOB: 12/04/2002
        पुरुष / MALE
        9876 5432 1098
        """
        res = extract_aadhaar_fields(sample)
        self.assertEqual(res.fields["name"].value, "TEST PERSON")
        self.assertEqual(res.fields["dob"].value, "2002-04-12")
        self.assertEqual(res.fields["dob"].status, "FOUND")
        self.assertEqual(res.fields["gender"].value, "MALE")
        self.assertEqual(res.fields["gender"].status, "FOUND")
        self.assertEqual(res.fields["aadhaar_number"].metadata.get("masked_value"), "XXXX XXXX 1098")


if __name__ == "__main__":
    unittest.main()



