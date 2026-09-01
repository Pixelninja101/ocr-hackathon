"""
Backend-facing Verification Service for Aadhaar Document Processing & Risk Assessment.
Exposes a single high-level function: verify_document(file_input).

Security & Privacy Guarantees:
- Zero unmasked Aadhaar numbers in response objects.
- Zero raw OCR text or QR payloads exposed.
- Deterministic, auditable JSON response contract.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from document_processor import process_document
from risk_engine import assess_document
from risk_engine.models import sanitize_pii_string
from risk_engine.scorer import HIGH_RISK_OVERRIDE_RULES

logger = logging.getLogger("integration.verification_service")


@dataclass
class DocumentInfo:
    """Document classification and confidence summary."""
    type: str = "unknown"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": str(self.type),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class VerificationSummary:
    """Risk scoring, risk tier, decision, and explicit override transparency."""
    risk_score: Optional[int]
    risk_level: str
    decision: str
    override_applied: bool = False
    override_reasons: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": int(self.risk_score) if self.risk_score is not None else None,
            "risk_level": str(self.risk_level),
            "decision": str(self.decision),
            "override_applied": bool(self.override_applied),
            "override_reasons": [str(r) for r in self.override_reasons],
            "summary": sanitize_pii_string(str(self.summary)) or "",
        }


@dataclass
class OCRCheck:
    """Visual OCR quality and presence summary."""
    available: bool = False
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": bool(self.available),
            "confidence": round(float(self.confidence), 4) if self.confidence is not None else None,
        }


@dataclass
class QRCheck:
    """QR barcode detection, decoding, and verification summary."""
    detected: bool = False
    decoded: bool = False
    verified: bool = False
    status: str = "QR_NOT_DETECTED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": bool(self.detected),
            "decoded": bool(self.decoded),
            "verified": bool(self.verified),
            "status": str(self.status),
        }


@dataclass
class ChecksumCheck:
    """Verhoeff checksum validation check."""
    available: bool = False
    valid: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": bool(self.available),
            "valid": bool(self.valid) if self.valid is not None else None,
        }


@dataclass
class CrossValidationCheck:
    """Visual OCR vs QR cross-validation consistency summary."""
    available: bool = False
    name_match: Optional[bool] = None
    dob_match: Optional[bool] = None
    gender_match: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": bool(self.available),
            "name_match": bool(self.name_match) if self.name_match is not None else None,
            "dob_match": bool(self.dob_match) if self.dob_match is not None else None,
            "gender_match": bool(self.gender_match) if self.gender_match is not None else None,
        }


@dataclass
class ChecksSummary:
    """Consolidated verification checks for the document."""
    ocr: OCRCheck = field(default_factory=OCRCheck)
    qr: QRCheck = field(default_factory=QRCheck)
    checksum: ChecksumCheck = field(default_factory=ChecksumCheck)
    cross_validation: CrossValidationCheck = field(default_factory=CrossValidationCheck)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ocr": self.ocr.to_dict(),
            "qr": self.qr.to_dict(),
            "checksum": self.checksum.to_dict(),
            "cross_validation": self.cross_validation.to_dict(),
        }


@dataclass
class VerificationResponse:
    """Standardized backend-facing document verification response."""
    success: bool
    document: DocumentInfo
    verification: VerificationSummary
    checks: ChecksSummary
    findings: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "success": bool(self.success),
            "document": self.document.to_dict(),
            "verification": self.verification.to_dict(),
            "checks": self.checks.to_dict(),
            "findings": [dict(f) for f in self.findings],
            "warnings": [sanitize_pii_string(str(w)) for w in self.warnings if w is not None],
        }
        if not self.success or self.error is not None:
            data["error"] = (
                {
                    k: sanitize_pii_string(v) if isinstance(v, str) else v
                    for k, v in self.error.items()
                }
                if self.error is not None
                else {
                    "code": "PROCESSING_FAILED",
                    "message": "Document processing was unsuccessful.",
                }
            )
        return data


def format_verification_response(assessment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms the internal risk engine assessment dictionary into the standardized
    backend verification contract.
    """
    success = bool(assessment.get("success", False))
    signals = assessment.get("signals", {})
    risk_info = assessment.get("risk", {})
    findings = assessment.get("findings", [])
    flags = assessment.get("flags", [])
    warnings = assessment.get("warnings", [])
    error = assessment.get("error")

    # Document details
    doc_sig = signals.get("document", {})
    doc_info = DocumentInfo(
        type=str(doc_sig.get("type", "unknown")),
        confidence=float(doc_sig.get("confidence", 0.0)),
    )

    # Overrides calculation
    triggered_overrides = [flag for flag in flags if flag in HIGH_RISK_OVERRIDE_RULES]
    override_applied = bool(triggered_overrides)

    # Verification summary
    ver_summary = VerificationSummary(
        risk_score=risk_info.get("score"),
        risk_level=str(risk_info.get("level", "UNKNOWN")),
        decision=str(risk_info.get("decision", "REVIEW")),
        override_applied=override_applied,
        override_reasons=triggered_overrides,
        summary=str(risk_info.get("summary", "")),
    )

    # OCR check
    ocr_sig = signals.get("ocr", {})
    ocr_available = bool(
        ocr_sig.get("confidence") is not None
        or any(isinstance(f, dict) and f.get("available") for f in ocr_sig.get("fields", {}).values())
    )
    ocr_check = OCRCheck(
        available=ocr_available,
        confidence=ocr_sig.get("confidence"),
    )

    # QR check
    qr_sig = signals.get("qr", {})
    qr_check = QRCheck(
        detected=bool(qr_sig.get("detected", False)),
        decoded=bool(qr_sig.get("decoded", False)),
        verified=bool(qr_sig.get("verified", False)),
        status=str(qr_sig.get("verification_status", "QR_NOT_DETECTED")),
    )

    # Checksum check
    chk_sig = signals.get("aadhaar_checksum", {})
    chk_check = ChecksumCheck(
        available=bool(chk_sig.get("available", False)),
        valid=chk_sig.get("valid"),
    )

    # Cross-validation check
    cv_sig = signals.get("cross_validation", {})
    cv_check = CrossValidationCheck(
        available=bool(cv_sig.get("available", False)),
        name_match=cv_sig.get("name", {}).get("match") if isinstance(cv_sig.get("name"), dict) else None,
        dob_match=cv_sig.get("dob", {}).get("match") if isinstance(cv_sig.get("dob"), dict) else None,
        gender_match=cv_sig.get("gender", {}).get("match") if isinstance(cv_sig.get("gender"), dict) else None,
    )

    checks_summary = ChecksSummary(
        ocr=ocr_check,
        qr=qr_check,
        checksum=chk_check,
        cross_validation=cv_check,
    )

    response = VerificationResponse(
        success=success,
        document=doc_info,
        verification=ver_summary,
        checks=checks_summary,
        findings=findings,
        warnings=warnings,
        error=error,
    )

    return response.to_dict()


def verify_document(
    file_input: Union[str, Path, bytes, io.BytesIO, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Main backend-facing API function for end-to-end document verification.

    Parameters:
        file_input: File path (str | Path), raw document bytes, BytesIO stream,
                    or an already-processed result dictionary.

    Returns:
        dict: Standardized, strictly JSON-serializable verification response.
    """
    try:
        if isinstance(file_input, dict):
            # If already an assessment result
            if "risk" in file_input and "signals" in file_input:
                return format_verification_response(file_input)
            # If an OCR processing result
            assessment = assess_document(file_input)
            return format_verification_response(assessment)

        # 1. Run OCR document processing
        processing_result = process_document(file_input)

        # 2. Run Risk Assessment Engine
        assessment = assess_document(processing_result)

        # 3. Format and return standard backend response
        return format_verification_response(assessment)

    except Exception as exc:
        logger.exception("Unexpected error in verify_document: %s", exc)
        err_response = VerificationResponse(
            success=False,
            document=DocumentInfo(type="unknown", confidence=0.0),
            verification=VerificationSummary(
                risk_score=None,
                risk_level="UNKNOWN",
                decision="REVIEW",
                override_applied=False,
                override_reasons=[],
                summary="An unexpected error occurred during document verification.",
            ),
            checks=ChecksSummary(),
            findings=[],
            warnings=[],
            error={
                "code": "INTERNAL_VERIFICATION_ERROR",
                "message": "Document verification could not be completed.",
            },
        )
        return err_response.to_dict()
