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

    def test_inaccurate_quad_fallback_decoding(self):
        payload = "AADHAAR_INACCURATE_QUAD_RECOVERY_TEST"
        qr_img = generate_qr_image(payload, box_size=8, border=4)
        qh, qw = qr_img.shape[:2]

        # Place QR inside a larger canvas
        canvas = np.ones((qh + 200, qw + 200, 3), dtype=np.uint8) * 255
        x_off, y_off = 100, 100
        canvas[y_off : y_off + qh, x_off : x_off + qw] = qr_img

        # Ground truth quad: corners of QR in canvas
        exact_quad = np.array([
            [x_off, y_off],
            [x_off + qw, y_off],
            [x_off + qw, y_off + qh],
            [x_off, y_off + qh],
        ], dtype=np.float32)

        # Intentionally perturb/contract quadrilateral vertices (simulating imprecise detection)
        center = np.mean(exact_quad, axis=0)
        imprecise_quad = center + (exact_quad - center) * 0.85  # 15% contracted

        decoder = QRCodeDecoderWrapper()
        ok, data, method, attempts = decoder.decode_quad(canvas, imprecise_quad)

        assert ok is True
        assert data == payload
        assert attempts >= 1

    def test_low_contrast_detected_qr_fallback(self):
        payload = "LOW_CONTRAST_FALLBACK_TEST"
        qr_img = generate_qr_image(payload, box_size=6, border=3)
        qh, qw = qr_img.shape[:2]

        # Reduce contrast
        low_contrast_qr = (qr_img.astype(np.float32) * 0.35 + 120).astype(np.uint8)

        canvas = np.ones((qh + 150, qw + 150, 3), dtype=np.uint8) * 230
        canvas[75 : 75 + qh, 75 : 75 + qw] = low_contrast_qr

        quad = np.array([
            [75, 75],
            [75 + qw, 75],
            [75 + qw, 75 + qh],
            [75, 75 + qh],
        ], dtype=np.float32)

        decoder = QRCodeDecoderWrapper()
        ok, data, method, attempts = decoder.decode_quad(canvas, quad)

        assert ok is True
        assert data == payload

    def test_dim_underexposed_qr_decoding(self):
        """Test decoding of dim / underexposed QR code."""
        payload = "DIM_UNDEREXPOSED_QR_TEST_DATA"
        qr_img = generate_qr_image(payload, box_size=8, border=3)
        # Strongly dim the image (simulate low-light camera capture)
        dim_qr = (qr_img.astype(np.float32) * 0.3).astype(np.uint8)

        processor = QRProcessor()
        result = processor.process_image(dim_qr, filename="dim_test.png", file_type="image")

        assert result["success"] is True
        assert result["qr_detected"] is True
        assert result["codes"][0]["decoded"] is True
        assert result["codes"][0]["data"] == payload

    def test_blurry_qr_decoding(self):
        """Test decoding of out-of-focus / motion-blurred QR code."""
        payload = "BLURRY_MOTION_QR_TEST_DATA"
        qr_img = generate_qr_image(payload, box_size=10, border=4)
        # Apply Gaussian blur
        blurred_qr = cv2.GaussianBlur(qr_img, (5, 5), sigmaX=1.5)

        processor = QRProcessor()
        result = processor.process_image(blurred_qr, filename="blurry_test.png", file_type="image")

        assert result["success"] is True
        assert result["qr_detected"] is True
        assert result["codes"][0]["decoded"] is True
        assert result["codes"][0]["data"] == payload

    def test_noisy_qr_decoding(self):
        """Test decoding of sensor-noise corrupted QR code."""
        payload = "NOISY_SENSOR_QR_TEST_DATA"
        qr_img = generate_qr_image(payload, box_size=8, border=4)
        # Add Gaussian noise
        noise = np.random.normal(0, 25, qr_img.shape).astype(np.float32)
        noisy_qr = np.clip(qr_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        processor = QRProcessor()
        result = processor.process_image(noisy_qr, filename="noisy_test.png", file_type="image")

        assert result["success"] is True
        assert result["qr_detected"] is True
        assert result["codes"][0]["decoded"] is True
        assert result["codes"][0]["data"] == payload

    def test_arbitrary_rotated_qr_in_document(self):
        """Test decoding of QR code rotated at an arbitrary angle (e.g. 37 degrees) in a document."""
        payload = "ARBITRARY_ROTATION_37DEG_TEST"
        qr_img = generate_qr_image(payload, box_size=8, border=4)
        qh, qw = qr_img.shape[:2]

        # Embed inside a larger document canvas to avoid clipping corners when rotated
        canvas_dim = int(max(qh, qw) * 2.2)
        canvas = np.ones((canvas_dim, canvas_dim, 3), dtype=np.uint8) * 255
        oy = (canvas_dim - qh) // 2
        ox = (canvas_dim - qw) // 2
        canvas[oy : oy + qh, ox : ox + qw] = qr_img

        # Rotate canvas by 37 degrees
        center = (canvas_dim / 2.0, canvas_dim / 2.0)
        M = cv2.getRotationMatrix2D(center, 37.0, 1.0)
        rotated_doc = cv2.warpAffine(
            canvas, M, (canvas_dim, canvas_dim), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
        )

        processor = QRProcessor()
        result = processor.process_image(rotated_doc, filename="rotated_37deg.png", file_type="image")

        assert result["success"] is True
        assert result["qr_detected"] is True
        assert result["codes"][0]["decoded"] is True
        assert result["codes"][0]["data"] == payload

    def test_skewed_tilted_perspective_qr(self):
        """Test decoding of perspective-distorted / tilted QR code."""
        payload = "PERSPECTIVE_TILTED_SKEWED_QR"
        qr_img = generate_qr_image(payload, box_size=8, border=4)
        qh, qw = qr_img.shape[:2]

        src_pts = np.array([[0, 0], [qw, 0], [qw, qh], [0, qh]], dtype=np.float32)
        # Skew perspective (one side narrower and tilted)
        dst_pts = np.array([[20, 15], [qw - 40, 35], [qw - 10, qh - 20], [30, qh - 10]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_qr = cv2.warpPerspective(
            qr_img, H, (qw + 50, qh + 50), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
        )

        processor = QRProcessor()
        result = processor.process_image(warped_qr, filename="tilted_qr.png", file_type="image")

        assert result["success"] is True
        assert result["qr_detected"] is True
        assert result["codes"][0]["decoded"] is True
        assert result["codes"][0]["data"] == payload


class TestAadhaarPayloadParser:
    """Tests for authentic UIDAI Aadhaar Secure QR parsing."""

    def test_aadhaar_secure_qr_v2_parsing(self):
        import zlib
        from backend.qr.parser import parse_qr_payload

        fields = [
            "V2", "3", "123456789012345678", "Sunita Sharma", "12-05-1992",
            "F", "W/O: Rajesh Sharma", "Bengaluru", "Near Metro Station",
            "45/A", "Indiranagar", "560038", "Indiranagar PO", "Karnataka",
            "100 Feet Road", "Bengaluru East", "Bengaluru", "9876"
        ]
        decompressed = b"\xff".join([f.encode("latin-1") for f in fields]) + b"\xff" + b"image_bytes_dummy" + (b"\x00" * 320)
        compressed = zlib.compress(decompressed)
        val = int.from_bytes(compressed, "big")
        val_str = str(val)

        parsed = parse_qr_payload(val_str)
        assert parsed is not None
        assert parsed["type"] == "aadhaar_secure"
        assert parsed["version"] == "V2"
        assert parsed["raw_attributes"]["name"] == "Sunita Sharma"
        assert parsed["raw_attributes"]["dob"] == "12-05-1992"
        assert parsed["raw_attributes"]["gender"] == "F"
        assert parsed["raw_attributes"]["pincode"] == "560038"
        assert parsed["raw_attributes"]["state"] == "Karnataka"
        assert parsed["raw_attributes"]["last_4_digits_mobile_no"] == "9876"

    def test_aadhaar_secure_qr_v1_parsing(self):
        import zlib
        from backend.qr.parser import parse_qr_payload

        # V1 without header
        fields = [
            "2", "987654321012345678", "Vikram Patel", "20-03-1985",
            "M", "S/O: Mohan Patel", "Ahmedabad", "Opposite Garden",
            "12", "Navrangpura", "380009", "Navrangpura PO", "Gujarat",
            "CG Road", "Ahmedabad City", "Ahmedabad"
        ]
        decompressed = b"\xff".join([f.encode("latin-1") for f in fields]) + b"\xff" + b"image_bytes_dummy" + (b"\x00" * 288)
        compressed = zlib.compress(decompressed)
        val = int.from_bytes(compressed, "big")
        val_str = str(val)

        parsed = parse_qr_payload(val_str)
        assert parsed is not None
        assert parsed["type"] == "aadhaar_secure"
        assert parsed["version"] == "V1"
        assert parsed["raw_attributes"]["referenceid"] == "987654321012345678"
        assert parsed["raw_attributes"]["name"] == "Vikram Patel"
        assert parsed["raw_attributes"]["dob"] == "20-03-1985"
        assert parsed["raw_attributes"]["gender"] == "M"
        assert parsed["raw_attributes"]["district"] == "Ahmedabad"
        assert parsed["raw_attributes"]["state"] == "Gujarat"

        # V1 with V1 header
        fields_v1_hdr = [
            "V1", "3", "123456789012345678", "Rahul Sharma", "14-02-1990",
            "M", "S/O: Om Sharma", "Jaipur", "Civil Lines",
            "12", "Near GPO", "302001", "GPO", "Rajasthan",
            "Station Rd", "Jaipur", "Jaipur"
        ]
        decomp_v1_hdr = b"\xff".join([f.encode("latin-1") for f in fields_v1_hdr]) + b"\xff" + b"photo" + (b"\x00" * 320)
        parsed_v1_hdr = parse_qr_payload(str(int.from_bytes(zlib.compress(decomp_v1_hdr), "big")))
        assert parsed_v1_hdr is not None
        assert parsed_v1_hdr["raw_attributes"]["referenceid"] == "123456789012345678"
        assert parsed_v1_hdr["raw_attributes"]["name"] == "Rahul Sharma"
        assert parsed_v1_hdr["raw_attributes"]["dob"] == "14-02-1990"
        assert parsed_v1_hdr["raw_attributes"]["gender"] == "M"
        assert parsed_v1_hdr["raw_attributes"]["state"] == "Rajasthan"

    def test_aadhaar_legacy_xml_parsing(self):
        from backend.qr.parser import parse_qr_payload

        xml_data = '<PrintLetterBarcodeData uid="123456789012" name="Anita Roy" gender="F" yob="1994" co="D/O: Subhash Roy" house="10" street="Park Street" loc="Park Circus" vtc="Kolkata" dist="Kolkata" state="West Bengal" pc="700016" />'
        parsed = parse_qr_payload(xml_data)
        assert parsed is not None
        assert parsed["type"] == "aadhaar_old"
        assert parsed["raw_attributes"]["name"] == "Anita Roy"
        assert parsed["raw_attributes"]["gender"] == "F"
        assert parsed["raw_attributes"]["yob"] == "1994"
        assert parsed["raw_attributes"]["pc"] == "700016"
        assert parsed["raw_attributes"]["state"] == "West Bengal"

