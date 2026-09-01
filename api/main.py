"""
Production-Hardened FastAPI REST API for Document Verification & Risk Assessment Engine.

Endpoints:
- GET  /            : Service status banner.
- GET  /health      : Lightweight liveness probe.
- GET  /ready       : Readiness probe verifying component initialization.
- POST /v1/verify   : Primary document verification endpoint.
- POST /verify      : Backwards-compatible alias for /v1/verify.

Security & Reliability Features:
- Security HTTP headers (X-Content-Type-Options, X-Frame-Options, Referrer-Policy, Cache-Control).
- In-process sliding-window rate limiting with 429 Retry-After response.
- Unique request correlation ID (X-Request-ID) propagation and operational metadata logging.
- Defensive upload size limits and empty-payload rejection.
- Zero raw PII, unmasked Aadhaar numbers, OCR text, or QR payloads in responses or logs.
- Standardized, deterministic JSON response contracts.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import api_config
from api.middleware import SecurityAndObservabilityMiddleware
from integration.verification_service import verify_document
from risk_engine.models import sanitize_pii_string

# Configure root logger based on api_config
logging.basicConfig(
    level=getattr(logging, api_config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.main")

API_DESCRIPTION = """
**Aadhaar Document Verification & Heuristic Risk Assessment Engine API**

### Verification Semantics & Guarantees:
- **Heuristic Risk Model**: `risk_score` (0–100) indicates document quality and data consistency. It is **NOT** a probability of fraud.
- **PASS Decision**: Indicates the document passed available automated checks with low observed risk. It does **not** establish legal identity or definitive authenticity.
- **REVIEW Decision**: Indicates one or more risk signals (e.g. image degradation, field mismatch, unreadable QR) require human review. It does **not** imply fraud.
- **Verhoeff Checksum**: Establishes mathematical consistency of 12-digit numbers; does not query external UIDAI databases.
- **Privacy by Design**: Zero raw PII, unmasked Aadhaar numbers, or OCR text are returned in responses or logs.
- **API Security**: Security headers enforced, in-process rate limiting (HTTP 429), and request correlation (X-Request-ID).
"""

app = FastAPI(
    title="Document Verification & Risk Assessment Engine API",
    description=API_DESCRIPTION,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# 1. Add Security, Observability, and Rate Limiting Middleware
app.add_middleware(SecurityAndObservabilityMiddleware)

# 2. Configure CORS (Explicit whitelist only; no wildcard fallback)
allowed_origins = api_config.ALLOWED_ORIGINS
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# -----------------------------------------------------------------------------
# Defensive Privacy Response Sanitizer
# -----------------------------------------------------------------------------

def _deep_sanitize_response(data: Any) -> Any:
    """
    Recursively scans and sanitizes response data to guarantee zero raw PII
    or continuous unmasked 12-digit Aadhaar numbers in string fields.
    """
    if isinstance(data, dict):
        return {k: _deep_sanitize_response(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_deep_sanitize_response(elem) for elem in data]
    if isinstance(data, str):
        sanitized = sanitize_pii_string(data)
        # Redact any remaining accidental 12 continuous digits
        return re.sub(r"\b\d{12}\b", "XXXX XXXX XXXX", sanitized)
    return data


# -----------------------------------------------------------------------------
# Global Exception Handlers
# -----------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handles FastAPI/Pydantic request validation errors cleanly without exposing internals."""
    req_id = getattr(request.state, "request_id", "unknown")
    logger.warning("[%s] Request validation failed on %s", req_id, request.url.path)
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "INVALID_REQUEST_PARAMETERS",
                "message": "The request was missing required fields or contained invalid parameters.",
            },
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handles explicit HTTP exceptions cleanly."""
    req_id = getattr(request.state, "request_id", "unknown")
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        content = {
            "success": False,
            "error": {
                "code": "REQUEST_ERROR",
                "message": str(exc.detail),
            },
        }
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches all unhandled server exceptions without leaking stack traces or internal paths."""
    req_id = getattr(request.state, "request_id", "unknown")
    logger.error("[%s] Unhandled internal server error on %s", req_id, request.url.path, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Document verification could not be completed.",
            },
        },
    )


# -----------------------------------------------------------------------------
# Core Verification Business Logic
# -----------------------------------------------------------------------------

async def _process_verification_upload(file: UploadFile, req_id: str) -> JSONResponse:
    """Internal core handler for processing uploaded document files."""
    try:
        contents = await file.read()
    except Exception as read_err:
        logger.warning("[%s] Failed to read uploaded file: %s", req_id, read_err)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "FILE_READ_ERROR",
                    "message": "Could not read uploaded file content.",
                },
            },
        )

    # 1. Reject empty upload
    if not contents or len(contents) == 0:
        logger.warning("[%s] Empty file uploaded", req_id)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "EMPTY_UPLOAD",
                    "message": "Uploaded file is empty.",
                },
            },
        )

    # 2. Reject oversized upload
    max_size = api_config.MAX_UPLOAD_SIZE_BYTES
    if len(contents) > max_size:
        logger.warning("[%s] File upload exceeds maximum allowed size: %d bytes", req_id, len(contents))
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "error": {
                    "code": "FILE_TOO_LARGE",
                    "message": f"File size exceeds the maximum limit of {max_size} bytes.",
                },
            },
        )

    # 3. Pass bytes to integration verification service
    try:
        raw_result = verify_document(contents)
    except Exception as svc_err:
        logger.error("[%s] Service-level exception during verification: %s", req_id, svc_err, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Document verification could not be completed.",
                },
            },
        )

    # 4. Defensive final privacy sweep
    sanitized_result = _deep_sanitize_response(raw_result)

    return JSONResponse(
        status_code=200,
        content=sanitized_result,
    )


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/", tags=["Status"])
async def root() -> Dict[str, str]:
    """Root endpoint indicating service health and version."""
    return {
        "name": "Document Verification Engine API",
        "version": "1.0.0",
        "status": "running",
        "environment": api_config.ENVIRONMENT,
    }


@app.get("/health", tags=["Status"])
async def health_check() -> Dict[str, str]:
    """Lightweight liveness probe (does not run OCR)."""
    return {
        "status": "ok",
    }


@app.get("/ready", tags=["Status"])
async def readiness_check() -> JSONResponse:
    """
    Readiness probe verifying that internal components and configuration are ready to serve traffic.
    Does not execute an actual OCR pipeline pass.
    """
    try:
        # Verify configuration validity
        api_config.validate()
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "environment": api_config.ENVIRONMENT,
            },
        )
    except Exception as exc:
        logger.error("Readiness check failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unready",
                "error": "Configuration validation failed.",
            },
        )


@app.post(
    "/v1/verify",
    tags=["Verification"],
    summary="Verify document (v1)",
    description="Processes and verifies an uploaded Aadhaar document image or PDF.",
)
async def verify_document_v1(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """Primary v1 endpoint for document verification."""
    req_id = getattr(request.state, "request_id", "unknown")
    return await _process_verification_upload(file, req_id)


@app.post(
    "/verify",
    tags=["Verification"],
    summary="Verify document (legacy alias)",
    description="Backwards-compatible alias pointing to /v1/verify.",
)
async def verify_document_legacy(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """Backwards-compatible alias endpoint pointing to v1 verification."""
    req_id = getattr(request.state, "request_id", "unknown")
    return await _process_verification_upload(file, req_id)
