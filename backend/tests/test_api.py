"""
API integration tests for FastAPI endpoints using TestClient.
"""

import cv2
import numpy as np
import pymupdf
import pytest
import qrcode
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    """Create test client fixture."""
    return TestClient(app)


def make_test_qr_png_bytes(data: str = "TEST_API_PAYLOAD") -> bytes:
    """Generate PNG bytes of a QR code."""
    qr = qrcode.QRCode(version=1, box_size=8, border=3)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = cv2.imencode(".png", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))[1]
    return buf.tobytes()


def make_test_pdf_bytes(data: str = "TEST_API_PDF_PAYLOAD") -> bytes:
    """Generate PDF bytes containing a QR code."""
    png_bytes = make_test_qr_png_bytes(data)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(50, 50, 250, 250), stream=png_bytes)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestAPIEndpoints:
    """FastAPI endpoint test cases."""

    def test_health_check(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        json_data = res.json()
        assert json_data["status"] == "ok"
        assert "QRCodeDetector" in json_data["engine"]

    def test_scan_image_success(self, client):
        png_bytes = make_test_qr_png_bytes("SAMPLE_AADHAAR_QR_CODE")
        files = {"file": ("test_doc.png", png_bytes, "image/png")}
        res = client.post("/api/qr/scan", files=files)

        assert res.status_code == 200
        json_data = res.json()
        assert json_data["success"] is True
        assert json_data["qr_detected"] is True
        assert json_data["qr_count"] == 1
        assert len(json_data["codes"]) == 1
        assert json_data["codes"][0]["decoded"] is True
        assert json_data["codes"][0]["data"] == "SAMPLE_AADHAAR_QR_CODE"
        assert json_data["metadata"]["preview_image"] is None  # Image does not return unnecessary base64 preview

    def test_scan_pdf_success(self, client):
        pdf_bytes = make_test_pdf_bytes("SAMPLE_PDF_QR_DATA")
        files = {"file": ("document.pdf", pdf_bytes, "application/pdf")}
        res = client.post("/api/qr/scan", files=files, data={"page": 0})

        assert res.status_code == 200
        json_data = res.json()
        assert json_data["success"] is True
        assert json_data["qr_detected"] is True
        assert json_data["qr_count"] == 1
        assert json_data["codes"][0]["decoded"] is True
        assert json_data["codes"][0]["data"] == "SAMPLE_PDF_QR_DATA"
        # PDF returns base64 preview for canvas visualization
        assert json_data["metadata"]["preview_image"] is not None
        assert json_data["metadata"]["preview_image"].startswith("data:image/jpeg;base64,")

    def test_scan_blank_image(self, client):
        blank = np.ones((300, 300, 3), dtype=np.uint8) * 255
        buf = cv2.imencode(".png", blank)[1].tobytes()
        files = {"file": ("blank.png", buf, "image/png")}
        res = client.post("/api/qr/scan", files=files)

        assert res.status_code == 200
        json_data = res.json()
        assert json_data["success"] is True
        assert json_data["qr_detected"] is False
        assert json_data["qr_count"] == 0
        assert len(json_data["codes"]) == 0
        assert "No QR code detected" in json_data["warnings"]

    def test_scan_invalid_empty_file(self, client):
        files = {"file": ("empty.png", b"", "image/png")}
        res = client.post("/api/qr/scan", files=files)

        assert res.status_code == 400
        json_data = res.json()
        assert json_data["success"] is False
        assert len(json_data["errors"]) > 0
