"""
Security, Rate Limiting, Correlation ID, and Operational Hardening Tests for API.
"""

import io
import re
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.main import app
from api.rate_limiter import rate_limiter
from api.config import api_config
from integration.verification_service import verify_document
from tests.fixtures.verification_cases import CASE_CLEAN


class TestAPISecurityAndOperations(unittest.TestCase):
    """Test suite for API security hardening, headers, rate limiting, and privacy."""

    def setUp(self):
        self.client = TestClient(app)
        rate_limiter.reset()

    def tearDown(self):
        rate_limiter.reset()

    # -------------------------------------------------------------------------
    # 1. Request / Correlation ID
    # -------------------------------------------------------------------------

    def test_01_supplied_request_id_preserved(self):
        """Incoming valid X-Request-ID should be preserved in response headers."""
        custom_id = "req-custom-trace-12345"
        resp = self.client.get("/health", headers={"X-Request-ID": custom_id})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("X-Request-ID"), custom_id)

    def test_02_missing_request_id_generated(self):
        """If X-Request-ID is omitted, a unique opaque ID should be generated."""
        resp1 = self.client.get("/health")
        resp2 = self.client.get("/health")
        id1 = resp1.headers.get("X-Request-ID")
        id2 = resp2.headers.get("X-Request-ID")
        self.assertIsNotNone(id1)
        self.assertIsNotNone(id2)
        self.assertNotEqual(id1, id2)
        self.assertTrue(len(id1) >= 16)

    def test_03_request_id_present_on_errors(self):
        """X-Request-ID must be present on error responses (400, 422, 429, 500)."""
        # 422 validation error
        resp_422 = self.client.post("/v1/verify", data={})
        self.assertEqual(resp_422.status_code, 422)
        self.assertIn("X-Request-ID", resp_422.headers)

        # 400 empty upload
        resp_400 = self.client.post(
            "/v1/verify",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")}
        )
        self.assertEqual(resp_400.status_code, 400)
        self.assertIn("X-Request-ID", resp_400.headers)

    # -------------------------------------------------------------------------
    # 2. Security Headers
    # -------------------------------------------------------------------------

    def test_04_security_headers_present_on_all_endpoints(self):
        """Defensive security headers must be injected on all HTTP responses."""
        endpoints = ["/", "/health", "/ready"]
        for ep in endpoints:
            with self.subTest(endpoint=ep):
                resp = self.client.get(ep)
                self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
                self.assertEqual(resp.headers.get("X-Frame-Options"), "DENY")
                self.assertEqual(resp.headers.get("Referrer-Policy"), "no-referrer")
                self.assertEqual(resp.headers.get("Cache-Control"), "no-store")

    # -------------------------------------------------------------------------
    # 3. Rate Limiting
    # -------------------------------------------------------------------------

    def test_05_rate_limiting_under_limit_allowed(self):
        """Requests under configured limit succeed without 429."""
        for _ in range(5):
            resp = self.client.get("/health")
            self.assertEqual(resp.status_code, 200)

    def test_06_rate_limiting_exceeded_returns_429(self):
        """Exceeding configured rate limit returns HTTP 429 with Retry-After header."""
        with patch.object(type(api_config), "RATE_LIMIT_REQUESTS", new=3):
            # 3 requests allowed
            for i in range(3):
                resp = self.client.get("/health", headers={"X-Forwarded-For": "198.51.100.1"})
                self.assertEqual(resp.status_code, 200)

            # 4th request rejected with 429
            resp_429 = self.client.get("/health", headers={"X-Forwarded-For": "198.51.100.1"})
            self.assertEqual(resp_429.status_code, 429)
            self.assertIn("Retry-After", resp_429.headers)
            body = resp_429.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], "RATE_LIMIT_EXCEEDED")
            self.assertIn("Rate limit exceeded", body["error"]["message"])

    def test_07_rate_limiting_independent_clients(self):
        """Different client IPs do not consume each other's rate limit quota."""
        with patch.object(type(api_config), "RATE_LIMIT_REQUESTS", new=2):
            # Exhaust client A limit
            self.client.get("/health", headers={"X-Forwarded-For": "192.0.2.1"})
            self.client.get("/health", headers={"X-Forwarded-For": "192.0.2.1"})
            resp_a_blocked = self.client.get("/health", headers={"X-Forwarded-For": "192.0.2.1"})
            self.assertEqual(resp_a_blocked.status_code, 429)

            # Client B is unaffected
            resp_b_ok = self.client.get("/health", headers={"X-Forwarded-For": "192.0.2.2"})
            self.assertEqual(resp_b_ok.status_code, 200)

    def test_08_rate_limiter_reset(self):
        """Calling reset() clears tracked counts and allows requests again."""
        with patch.object(type(api_config), "RATE_LIMIT_REQUESTS", new=1):
            resp1 = self.client.get("/health")
            self.assertEqual(resp1.status_code, 200)

            resp2 = self.client.get("/health")
            self.assertEqual(resp2.status_code, 429)

            rate_limiter.reset()

            resp3 = self.client.get("/health")
            self.assertEqual(resp3.status_code, 200)

    # -------------------------------------------------------------------------
    # 4. Health & Readiness
    # -------------------------------------------------------------------------

    def test_09_health_endpoint_liveness(self):
        """GET /health returns lightweight status ok."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_10_ready_endpoint_readiness(self):
        """GET /ready returns readiness confirmation and environment."""
        resp = self.client.get("/ready")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body.get("status"), "ready")
        self.assertIn("environment", body)

    # -------------------------------------------------------------------------
    # 5. Error Contract & Internal Protection
    # -------------------------------------------------------------------------

    def test_11_internal_error_does_not_leak_stack_trace_or_paths(self):
        """When an unhandled exception occurs, API returns sanitized 500 INTERNAL_ERROR."""
        with patch("api.main.verify_document", side_effect=RuntimeError("Secret internal database /var/data/leak")):
            resp = self.client.post(
                "/v1/verify",
                files={"file": ("sample.png", io.BytesIO(b"dummy image bytes"), "image/png")}
            )
            self.assertEqual(resp.status_code, 500)
            body = resp.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], "INTERNAL_ERROR")
            self.assertEqual(body["error"]["message"], "Document verification could not be completed.")
            # Zero leakage of internal path or exception name
            self.assertNotIn("RuntimeError", resp.text)
            self.assertNotIn("/var/data/leak", resp.text)
            self.assertNotIn("Traceback", resp.text)

    # -------------------------------------------------------------------------
    # 6. Upload Hardening & Oversized Rejection
    # -------------------------------------------------------------------------

    def test_12_oversized_upload_rejected_with_413(self):
        """Files exceeding MAX_UPLOAD_SIZE_BYTES are rejected with 413 FILE_TOO_LARGE."""
        with patch.object(type(api_config), "MAX_UPLOAD_SIZE_BYTES", new=500):
            oversized_data = b"X" * 600
            resp = self.client.post(
                "/v1/verify",
                files={"file": ("large.png", io.BytesIO(oversized_data), "image/png")}
            )
            self.assertEqual(resp.status_code, 413)
            body = resp.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], "FILE_TOO_LARGE")

    # -------------------------------------------------------------------------
    # 7. Privacy Regressions
    # -------------------------------------------------------------------------

    def test_13_privacy_regression_check_no_pii_emitted(self):
        """Ensures that verified responses never leak raw 12-digit Aadhaar numbers or unmasked names."""
        clean_result = verify_document(CASE_CLEAN["payload"])
        with patch("api.main.verify_document", return_value=clean_result):
            resp = self.client.post(
                "/v1/verify",
                files={"file": ("aadhaar.png", io.BytesIO(b"fake-bytes"), "image/png")}
            )
            self.assertEqual(resp.status_code, 200)
            resp_str = resp.text
            # No unmasked continuous 12-digit sequence
            self.assertIsNone(re.search(r"\b\d{12}\b", resp_str))
            # No raw OCR text keys
            self.assertNotIn("raw_ocr_text", resp_str)
            self.assertNotIn("qr_payload", resp_str)


if __name__ == "__main__":
    unittest.main()
