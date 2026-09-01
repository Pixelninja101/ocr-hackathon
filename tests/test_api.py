"""
Comprehensive Automated Tests for FastAPI Verification API (tests/test_api.py).
Tests all endpoints, versioning, request IDs, file validations, error handling, PII privacy, and resilience.
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import re
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from risk_engine.config import config
from tests.fixtures.verification_cases import DEMO_CASES


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for FastAPI REST API endpoints and validation layers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

        # Load synthetic test image generator from OCR package
        spec = importlib.util.spec_from_file_location(
            "ocr_test_helpers", r"C:\icons\ocr\tests\test_helpers.py"
        )
        assert spec is not None and spec.loader is not None
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        cls.ocr_helpers = helpers

        # Generate a valid synthetic Aadhaar image for API integration tests
        cls.synthetic_valid_image = cls.ocr_helpers.create_synthetic_aadhaar_bytes(
            name="RAJESH SHARMA",
            dob="15/07/1990",
            gender="MALE",
            aadhaar_num="2345 6789 0124",  # Valid Verhoeff check digit
            include_qr=True,
        )

    # -------------------------------------------------------------------------
    # Test A: GET /health
    # -------------------------------------------------------------------------

    def test_a_health_endpoint(self) -> None:
        """Test A: GET /health returns HTTP 200 with {'status': 'ok'} and X-Request-ID header."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertIn("X-Request-ID", response.headers)

    # -------------------------------------------------------------------------
    # Test B: GET /
    # -------------------------------------------------------------------------

    def test_b_root_endpoint(self) -> None:
        """Test B: GET / returns HTTP 200 with service information banner and version."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "running")
        self.assertEqual(data.get("name"), "Document Verification Engine API")
        self.assertIn("version", data)

    # -------------------------------------------------------------------------
    # Test C: Missing upload
    # -------------------------------------------------------------------------

    def test_c_missing_upload_returns_422(self) -> None:
        """Test C: POST /v1/verify with missing file parameter returns controlled 422 error."""
        response = self.client.post("/v1/verify", data={})
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "INVALID_REQUEST_PARAMETERS")

    # -------------------------------------------------------------------------
    # Test D: Empty upload (0 bytes)
    # -------------------------------------------------------------------------

    def test_d_empty_upload_returns_400_empty_upload(self) -> None:
        """Test D: POST /v1/verify with a 0-byte file returns HTTP 400 with code EMPTY_UPLOAD."""
        empty_file = io.BytesIO(b"")
        files = {"file": ("empty.png", empty_file, "image/png")}

        response = self.client.post("/v1/verify", files=files)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertEqual(data.get("error", {}).get("code"), "EMPTY_UPLOAD")
        self.assertIn("empty", data.get("error", {}).get("message", "").lower())

    # -------------------------------------------------------------------------
    # Test E: Synthetic valid document on /v1/verify
    # -------------------------------------------------------------------------

    def test_e_synthetic_valid_document_success(self) -> None:
        """Test E: POST /v1/verify with valid synthetic image returns HTTP 200 and standard response."""
        file_obj = io.BytesIO(self.synthetic_valid_image)
        files = {"file": ("aadhaar_valid.png", file_obj, "image/png")}

        response = self.client.post("/v1/verify", files=files)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("document", data)
        self.assertIn("verification", data)
        self.assertIn("checks", data)
        self.assertIn("findings", data)
        self.assertIn("warnings", data)

        ver = data["verification"]
        self.assertIn("risk_score", ver)
        self.assertIn("risk_level", ver)
        self.assertIn("decision", ver)
        self.assertIn("override_applied", ver)
        self.assertIn("override_reasons", ver)

    # -------------------------------------------------------------------------
    # Test F: Backwards-compatible alias /verify
    # -------------------------------------------------------------------------

    def test_f_backwards_compatible_verify_route(self) -> None:
        """Test F: POST /verify operates as a backwards-compatible alias for /v1/verify."""
        file_obj = io.BytesIO(self.synthetic_valid_image)
        files = {"file": ("aadhaar_valid.png", file_obj, "image/png")}

        response = self.client.post("/verify", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data["document"]["type"], "aadhaar")

    # -------------------------------------------------------------------------
    # Test G: Verification-level failure returns HTTP 200 (not 4xx/5xx)
    # -------------------------------------------------------------------------

    def test_g_verification_issue_returns_http_200(self) -> None:
        """Test G: Corrupted/non-Aadhaar image returns HTTP 200 with structured verification result."""
        corrupt_file = io.BytesIO(b"RANDOM_NON_IMAGE_CORRUPT_BYTES_DATA_12345")
        files = {"file": ("corrupt.png", corrupt_file, "image/png")}

        response = self.client.post("/v1/verify", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)
        self.assertEqual(data["verification"]["decision"], "REVIEW")

    # -------------------------------------------------------------------------
    # Test H: Unexpected service exception returns HTTP 500 without stack trace
    # -------------------------------------------------------------------------

    def test_h_unexpected_service_exception_handled_safely(self) -> None:
        """Test H: Internal service exception returns controlled HTTP 500 with no stack traces or PII."""
        file_obj = io.BytesIO(b"TEST_FILE_CONTENT")
        files = {"file": ("test.png", file_obj, "image/png")}

        with patch("api.main.verify_document", side_effect=RuntimeError("Secret database crash in C:/icons/internal")):
            response = self.client.post("/v1/verify", files=files)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertEqual(data.get("error", {}).get("code"), "INTERNAL_ERROR")
        error_msg = data.get("error", {}).get("message", "")
        self.assertNotIn("Traceback", error_msg)
        self.assertNotIn("C:/icons", error_msg)
        self.assertNotIn("RuntimeError", error_msg)

    # -------------------------------------------------------------------------
    # Test I: Privacy - Zero Unmasked 12-Digit Numbers in Response
    # -------------------------------------------------------------------------

    def test_i_privacy_recursive_response_inspection(self) -> None:
        """Test I: Complete JSON response must never contain continuous 12-digit unmasked Aadhaar numbers."""
        file_obj = io.BytesIO(self.synthetic_valid_image)
        files = {"file": ("aadhaar.png", file_obj, "image/png")}

        response = self.client.post("/v1/verify", files=files)
        self.assertEqual(response.status_code, 200)

        serialized = json.dumps(response.json())
        self.assertFalse(bool(re.search(r"\b\d{12}\b", serialized)))

    # -------------------------------------------------------------------------
    # Test J: Large upload exceeding MAX_UPLOAD_SIZE returns HTTP 413
    # -------------------------------------------------------------------------

    def test_j_oversized_upload_returns_413(self) -> None:
        """Test J: File exceeding configured upload limit returns HTTP 413 with FILE_TOO_LARGE."""
        max_limit = config.MAX_UPLOAD_SIZE_BYTES
        oversized_bytes = b"0" * (max_limit + 10)
        file_obj = io.BytesIO(oversized_bytes)
        files = {"file": ("large.png", file_obj, "image/png")}

        response = self.client.post("/v1/verify", files=files)
        self.assertEqual(response.status_code, 413)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertEqual(data.get("error", {}).get("code"), "FILE_TOO_LARGE")

    # -------------------------------------------------------------------------
    # Test K: MIME Spoofing
    # -------------------------------------------------------------------------

    def test_k_mime_spoofing_handled_gracefully(self) -> None:
        """Test K: Valid image bytes sent with spoofed text/plain MIME type are processed safely."""
        file_obj = io.BytesIO(self.synthetic_valid_image)
        files = {"file": ("document.txt", file_obj, "text/plain")}

        response = self.client.post("/v1/verify", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("document", data)

    # -------------------------------------------------------------------------
    # Test L: Request ID Uniqueness and Propagation
    # -------------------------------------------------------------------------

    def test_l_request_id_uniqueness_and_propagation(self) -> None:
        """Test L: X-Request-ID header is generated uniquely across requests and preserves valid incoming IDs."""
        # 1. Generated unique IDs
        r1 = self.client.get("/health")
        r2 = self.client.get("/health")
        id1 = r1.headers.get("X-Request-ID")
        id2 = r2.headers.get("X-Request-ID")
        self.assertTrue(bool(id1))
        self.assertTrue(bool(id2))
        self.assertNotEqual(id1, id2)

        # 2. Preserves valid custom request ID
        custom_id = "test-req-correlation-12345"
        r3 = self.client.get("/health", headers={"X-Request-ID": custom_id})
        self.assertEqual(r3.headers.get("X-Request-ID"), custom_id)

    # -------------------------------------------------------------------------
    # Test M: OpenAPI Documentation Metadata
    # -------------------------------------------------------------------------

    def test_m_openapi_schema_metadata(self) -> None:
        """Test M: OpenAPI schema contains title, version, tags, and endpoint documentation."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        self.assertIn("info", schema)
        self.assertEqual(schema["info"]["title"], "Document Verification & Risk Assessment Engine API")
        self.assertEqual(schema["info"]["version"], "1.0.0")
        self.assertIn("/v1/verify", schema["paths"])
        self.assertIn("/verify", schema["paths"])
        self.assertIn("/health", schema["paths"])

    # -------------------------------------------------------------------------
    # Test N: Demo Fixtures Validation via API
    # -------------------------------------------------------------------------

    def test_n_demo_fixtures_produce_expected_results(self) -> None:
        """Test N: All synthetic demo cases evaluate to their expected risk levels and decisions."""
        from integration.verification_service import verify_document

        for case in DEMO_CASES:
            with self.subTest(case=case["name"]):
                res = verify_document(case["payload"])
                self.assertEqual(res["verification"]["risk_level"], case["expected_level"])
                self.assertEqual(res["verification"]["decision"], case["expected_decision"])
                self.assertEqual(res["verification"]["override_applied"], case["expected_override"])


if __name__ == "__main__":
    unittest.main()
