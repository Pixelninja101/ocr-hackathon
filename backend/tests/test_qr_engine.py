"""
Unit and integration tests for the QR Code Detection & Decoding Engine.
"""

import io
import cv2
import numpy as np
import pymupdf
import pytest
import qrcode
from PIL import Image

from backend.qr.decoder import QRCodeDecoderWrapper
from backend.qr.detector import QRCodeDetectorWrapper
from backend.qr.processor import QRProcessor
from backend.utils.image_handler import bytes_to_cv2_image, cv2_to_base64_data_url
from backend.utils.pdf_handler import render_pdf_page_to_cv2_image
from backend.utils.validation import validate_uploaded_file


def generate_qr_image(data: str, box_size: int = 10, border: int = 4) -> np.ndarray:
    """Helper to generate an OpenCV BGR image containing a valid QR code."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    rgb_arr = np.array(pil_img)
    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)


def generate_multi_qr_image(payload1: str, payload2: str) -> np.ndarray:
    """Helper to generate an image with two separate QR codes."""
    qr1 = generate_qr_image(payload1, box_size=8, border=3)
    qr2 = generate_qr_image(payload2, box_size=8, border=3)

    h1, w1 = qr1.shape[:2]
    h2, w2 = qr2.shape[:2]

    canvas_h = max(h1, h2) + 100
    canvas_w = w1 + w2 + 150
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Place QR1 on the left
    canvas[50 : 50 + h1, 50 : 50 + w1] = qr1
    # Place QR2 on the right
    canvas[50 : 50 + h2, 100 + w1 : 100 + w1 + w2] = qr2

    return canvas


def generate_pdf_with_qr(payload: str) -> bytes:
    """Helper to generate an in-memory PDF containing a QR code."""
    qr_img = generate_qr_image(payload, box_size=10, border=4)
    _, encoded_jpg = cv2.imencode(".jpg", qr_img)
    jpg_bytes = encoded_jpg.tobytes()

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 standard

    # Insert image into PDF page
    rect = pymupdf.Rect(100, 100, 300, 300)
    page.insert_image(rect, stream=jpg_bytes)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestQRValidation:
    """Tests for file validation and security checks."""

    def test_empty_file(self):
        valid, ftype, err = validate_uploaded_file("test.png", b"")
        assert not valid
        assert "empty" in err.lower()

    def test_invalid_extension(self):
        valid, ftype, err = validate_uploaded_file("test.exe", b"MZ\x90\x00SomeExecutable")
        assert not valid
        assert "unsupported file extension" in err.lower()

    def test_valid_image(self):
        qr_img = generate_qr_image("TEST_VALIDATION")
        _, buf = cv2.imencode(".png", qr_img)
        valid, ftype, err = validate_uploaded_file("doc.png", buf.tobytes())
        assert valid
        assert ftype == "image"
        assert err is None

    def test_valid_pdf(self):
        pdf_bytes = generate_pdf_with_qr("PDF_VALIDATION")
        valid, ftype, err = validate_uploaded_file("document.pdf", pdf_bytes)
        assert valid
        assert ftype == "pdf"
        assert err is None


class TestQREngine:
    """Tests for detection, decoding, and processing logic."""

    def test_single_qr_detection_and_decoding(self):
        payload = "https://uidai.gov.in/sample-aadhaar-qr-test"
        img = generate_qr_image(payload)

        processor = QRProcessor()
        result = processor.process_image(img, filename="sample_qr.png", file_type="image")

        assert result["success"] is True
        assert result["qr_detected"] is True
        assert result["qr_count"] == 1
        assert len(result["codes"]) == 1
        assert result["codes"][0]["decoded"] is True
        assert result["codes"][0]["data"] == payload
        assert len(result["codes"][0]["bbox"]) == 4
        assert len(result["warnings"]) == 0
        assert len(result["errors"]) == 0

    def test_multi_qr_detection_and_decoding(self):
        payload1 = "QR_CODE_ALPHA_01"
        payload2 = "QR_CODE_BETA_02"
        img = generate_multi_qr_image(payload1, payload2)

        processor = QRProcessor()
        result = processor.process_image(img, filename="multi_qr.png", file_type="image")

        assert result["success"] is True
        assert result["qr_detected"] is True
        assert result["qr_count"] >= 2
        decoded_payloads = [c["data"] for c in result["codes"] if c["decoded"]]
        assert payload1 in decoded_payloads
        assert payload2 in decoded_payloads

    def test_no_qr_detected(self):
        # Blank white canvas
        blank = np.ones((500, 500, 3), dtype=np.uint8) * 255

        processor = QRProcessor()
        result = processor.process_image(blank, filename="blank.jpg", file_type="image")

        assert result["success"] is True
        assert result["qr_detected"] is False
        assert result["qr_count"] == 0
        assert len(result["codes"]) == 0
        assert "No QR code detected" in result["warnings"]

    def test_pdf_rendering_and_qr_extraction(self):
        payload = "SECURE_AADHAAR_PDF_QR_DATA"
        pdf_bytes = generate_pdf_with_qr(payload)

        # Render PDF page at ~300 DPI
        cv2_img, total_pages, curr_page = render_pdf_page_to_cv2_image(pdf_bytes, page_number=0, dpi=300)
        assert total_pages == 1
        assert curr_page == 0
        assert cv2_img.shape[0] > 1000  # 300 DPI rendered image is high resolution

        processor = QRProcessor()
        result = processor.process_image(
            cv2_img,
            filename="document.pdf",
            file_type="pdf",
            page_metadata={"page_count": total_pages, "current_page": curr_page + 1},
        )

        assert result["success"] is True
        assert result["qr_detected"] is True
        assert result["qr_count"] == 1
        assert result["codes"][0]["decoded"] is True
        assert result["codes"][0]["data"] == payload

    def test_low_resolution_dense_aadhaar_qr(self):
        # Simulate low-res 280x280 dense QR code (like Aadhaar cropped screenshot)
        dense_payload = "492837492837492837492837" * 20
        qr = qrcode.QRCode(version=14, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=3, border=1)
        qr.add_data(dense_payload)
        qr.make(fit=True)
        pil_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2_img = cv2.resize(cv2_img, (280, 280), interpolation=cv2.INTER_AREA)

        processor = QRProcessor()
        result = processor.process_image(cv2_img, filename="dense_aadhaar_test.png", file_type="image")

        assert result["success"] is True
        assert result["qr_detected"] is True
        assert result["qr_count"] == 1
        assert result["codes"][0]["decoded"] is True
        assert result["codes"][0]["data"] == dense_payload

    def test_binary_byte_qr_payload(self):
        raw_bytes = bytes([0x1F, 0x8B, 0x08, 0x00]) + b"SECURE_BYTE_STREAM" + bytes([0x00, 0xFF, 0xFE])
        qr = qrcode.QRCode(version=6, box_size=6, border=3)
        qr.add_data(raw_bytes)
        qr.make(fit=True)
        pil_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        processor = QRProcessor()
        result = processor.process_image(cv2_img, filename="binary_qr.png", file_type="image")

        assert result["success"] is True
        assert result["qr_detected"] is True
        assert result["codes"][0]["decoded"] is True
        assert "SECURE_BYTE_STREAM" in result["codes"][0]["data"]

