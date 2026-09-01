"""
Integration package exposing high-level verification service for backend consumption.
"""

from integration.verification_service import (
    ChecksSummary,
    ChecksumCheck,
    CrossValidationCheck,
    DocumentInfo,
    OCRCheck,
    QRCheck,
    VerificationResponse,
    VerificationSummary,
    format_verification_response,
    verify_document,
)

__all__ = [
    "verify_document",
    "format_verification_response",
    "VerificationResponse",
    "VerificationSummary",
    "DocumentInfo",
    "ChecksSummary",
    "OCRCheck",
    "QRCheck",
    "ChecksumCheck",
    "CrossValidationCheck",
]
