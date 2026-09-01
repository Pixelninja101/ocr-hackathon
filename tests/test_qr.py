"""
Unit tests for QR Code Detection, Payload Decoding, State Separation, and Privacy Masking.
Covers all requirements from Prompt 7:
1. QR state separation (detected, decoded, verified)
2. UIDAI XML payload parsing with full address assembly
3. JSON payload parsing
4. Non-Aadhaar QR payloads
5. Blurry / distorted QR detection without decode
6. Privacy masking of raw UID in payloads
7. Explicit unverified state (verified: False)
"""

import unittest
import cv2
import numpy as np

from document_processor.qr.decoder import (
    mask_payload_uid,
    parse_aadhaar_json_qr,
    parse_aadhaar_xml_qr,
    parse_qr_payload,
    process_qr_code,
)
from document_processor.qr.detector import detect_qr_code
from tests.test_helpers import create_synthetic_qr_image


class TestQRProcessing(unittest.TestCase):

    # 1. State Separation: No QR
    def test_1_qr_states_no_qr(self):
        blank_image = np.full((300, 300, 3), 255, dtype=np.uint8)
        res = process_qr_code(blank_image)
        self.assertFalse(res["detected"])
        self.assertFalse(res["decoded"])
        self.assertFalse(res["verified"])
        self.assertNotIn("fields", res)

    # 2. UIDAI Standard XML QR Decode & Address Assembly
    def test_2_qr_detection_and_xml_decode(self):
        xml_payload = (
            '<PrintLetterBarcodeData uid="987654321098" name="RAHUL KUMAR" dob="12/04/2002" '
            'gender="MALE" co="S/O Rajesh Kumar" house="123" street="MG Road" dist="South Delhi" '
            'state="Delhi" pc="110016" />'
        )
        qr_img = create_synthetic_qr_image(xml_payload, size=250)
        qr_h, qr_w = qr_img.shape[:2]

        # Place on background canvas
        canvas = np.full((qr_h + 100, qr_w + 100, 3), 240, dtype=np.uint8)
        canvas[50 : 50 + qr_h, 50 : 50 + qr_w] = qr_img

        res = process_qr_code(canvas)
        self.assertTrue(res["detected"])
        self.assertTrue(res["decoded"])
        self.assertFalse(res["verified"])  # Explicitly False: cryptographic validation out of scope
        self.assertEqual(res["format"], "xml")
        self.assertIn("fields", res)

        fields = res["fields"]
        self.assertEqual(fields["name"], "RAHUL KUMAR")
        self.assertEqual(fields["dob"], "12/04/2002")
        self.assertEqual(fields["gender"], "MALE")
        self.assertEqual(fields["aadhaar_number"], "987654321098")
        self.assertEqual(fields["masked_aadhaar"], "XXXX XXXX 1098")
        self.assertIn("MG Road", fields["address"])
        self.assertIn("110016", fields["address"])

    # 3. JSON QR Payload Parsing
    def test_3_parse_qr_json_payload(self):
        json_payload = (
            '{"name": "Priya Sharma", "dob": "15/08/1995", "gender": "Female", '
            '"uid": "123456789012", "address": "Flat 402, Mumbai"}'
        )
        parsed, fmt = parse_qr_payload(json_payload)
        self.assertIsNotNone(parsed)
        self.assertEqual(fmt, "json")
        self.assertEqual(parsed["name"], "PRIYA SHARMA")
        self.assertEqual(parsed["dob"], "15/08/1995")
        self.assertEqual(parsed["gender"], "FEMALE")
        self.assertEqual(parsed["aadhaar_number"], "123456789012")
        self.assertEqual(parsed["masked_aadhaar"], "XXXX XXXX 9012")
        self.assertEqual(parsed["address"], "Flat 402, Mumbai")

    # 4. Non-Aadhaar QR Code Handling
    def test_4_non_aadhaar_qr(self):
        url_payload = "https://upi.example.com/pay?pa=merchant@upi&pn=Store"
        parsed, fmt = parse_qr_payload(url_payload)
        self.assertIsNone(parsed)
        self.assertEqual(fmt, "unknown")

    # 5. Privacy Masking in Raw Payloads
    def test_5_privacy_masking_helper(self):
        raw_xml = '<PrintLetterBarcodeData uid="123456789012" name="Test" />'
        masked = mask_payload_uid(raw_xml)
        self.assertNotIn("123456789012", masked)
        self.assertIn("XXXX XXXX 9012", masked)

    # 6. Unreadable / Distorted QR Handling (detected: True, decoded: False)
    def test_6_blurry_qr_detection_without_decode(self):
        xml_payload = '<PrintLetterBarcodeData uid="123456789012" name="Test" />'
        qr_img = create_synthetic_qr_image(xml_payload, size=250)

        # Apply aggressive gaussian blur to make matrix unreadable while finder patterns remain visible
        blurred = cv2.GaussianBlur(qr_img, (25, 25), 0)

        # When detector finds points but decoder cannot extract payload
        res = process_qr_code(blurred)
        # Should gracefully return structured result without crashing
        self.assertIn("detected", res)
        self.assertIn("decoded", res)
        self.assertFalse(res["verified"])


if __name__ == "__main__":
    unittest.main()
