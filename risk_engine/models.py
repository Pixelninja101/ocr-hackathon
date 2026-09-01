"""
Data models, enums, and response containers for Risk Engine.
Ensures strict JSON serializability and PII safety.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RiskDecision(str, Enum):
    """Overall document verification / risk decision."""
    PASS = "PASS"
    REVIEW = "REVIEW"
    # Legacy aliases
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"
    REJECTED = "REJECTED"


class RiskLevel(str, Enum):
    """Categorical risk tiers."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class RuleSeverity(str, Enum):
    """Severity tier for individual risk findings."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class QRVerificationStatus(str, Enum):
    """Normalized QR detection and verification states."""
    QR_NOT_DETECTED = "QR_NOT_DETECTED"
    QR_DETECTED_NOT_DECODED = "QR_DETECTED_NOT_DECODED"
    QR_DECODED_VERIFICATION_UNAVAILABLE = "QR_DECODED_VERIFICATION_UNAVAILABLE"
    QR_VERIFIED = "QR_VERIFIED"
    QR_VERIFICATION_FAILED = "QR_VERIFICATION_FAILED"


def mask_sensitive_number(val: Optional[str]) -> str:
    """
    Masks sensitive 12-digit Aadhaar / ID numbers in data structures and logs.
    Example: '987654321098' -> 'XXXX XXXX 1098'
    """
    if not val:
        return ""
    clean = "".join(ch for ch in str(val) if ch.isdigit() or ch.upper() == "X")
    if len(clean) >= 12:
        return f"XXXX XXXX {clean[-4:]}"
    elif len(clean) >= 4:
        return f"XXXX {clean[-4:]}"
    return "XXXX"


def sanitize_pii_string(text: Optional[str]) -> Optional[str]:
    """
    Sanitizes arbitrary text strings to ensure no unmasked 12-digit Aadhaar numbers leak.
    Supports continuous 12 digits (987654321098), spaced (9876 5432 1098), or hyphenated.
    """
    if not text:
        return text
    # Pattern matching 12 digits with optional spaces or hyphens between 4-digit blocks
    pattern = r"\b(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})\b"
    return re.sub(pattern, r"XXXX XXXX \3", str(text))


@dataclass
class DocumentSignal:
    type: str = "unknown"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": str(self.type),
            "confidence": float(self.confidence),
        }


@dataclass
class OCRFieldSignal:
    available: bool = False
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": bool(self.available),
            "confidence": float(self.confidence) if self.confidence is not None else None,
        }


@dataclass
class OCRDOBFieldSignal:
    available: bool = False
    confidence: Optional[float] = None
    precision: Optional[str] = None  # "full" | "year" | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": bool(self.available),
            "confidence": float(self.confidence) if self.confidence is not None else None,
            "precision": str(self.precision) if self.precision is not None else None,
        }


@dataclass
class OCRFieldsSignal:
    name: OCRFieldSignal = field(default_factory=OCRFieldSignal)
    dob: OCRDOBFieldSignal = field(default_factory=OCRDOBFieldSignal)
    gender: OCRFieldSignal = field(default_factory=OCRFieldSignal)
    aadhaar_number: OCRFieldSignal = field(default_factory=OCRFieldSignal)
    address: OCRFieldSignal = field(default_factory=OCRFieldSignal)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name.to_dict(),
            "dob": self.dob.to_dict(),
            "gender": self.gender.to_dict(),
            "aadhaar_number": self.aadhaar_number.to_dict(),
            "address": self.address.to_dict(),
        }


@dataclass
class OCRSignal:
    confidence: Optional[float] = None
    language: Optional[str] = None
    fields: OCRFieldsSignal = field(default_factory=OCRFieldsSignal)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": float(self.confidence) if self.confidence is not None else None,
            "language": str(self.language) if self.language is not None else None,
            "fields": self.fields.to_dict(),
        }


@dataclass
class QRSignal:
    detected: bool = False
    decoded: bool = False
    verified: bool = False
    verification_status: str = QRVerificationStatus.QR_NOT_DETECTED.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": bool(self.detected),
            "decoded": bool(self.decoded),
            "verified": bool(self.verified),
            "verification_status": str(self.verification_status),
        }


@dataclass
class CrossValidationNameSignal:
    available: bool = False
    similarity: Optional[float] = None
    match: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": bool(self.available),
            "similarity": float(self.similarity) if self.similarity is not None else None,
            "match": bool(self.match) if self.match is not None else None,
        }


@dataclass
class CrossValidationDOBSignal:
    available: bool = False
    match: Optional[bool] = None
    comparison: Optional[str] = None  # "full" | "year" | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": bool(self.available),
            "match": bool(self.match) if self.match is not None else None,
            "comparison": str(self.comparison) if self.comparison is not None else None,
        }


@dataclass
class CrossValidationGenderSignal:
    available: bool = False
    match: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": bool(self.available),
            "match": bool(self.match) if self.match is not None else None,
        }


@dataclass
class CrossValidationSignal:
    available: bool = False
    name: CrossValidationNameSignal = field(default_factory=CrossValidationNameSignal)
    dob: CrossValidationDOBSignal = field(default_factory=CrossValidationDOBSignal)
    gender: CrossValidationGenderSignal = field(default_factory=CrossValidationGenderSignal)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": bool(self.available),
            "name": self.name.to_dict(),
            "dob": self.dob.to_dict(),
            "gender": self.gender.to_dict(),
        }


@dataclass
class AadhaarChecksumSignal:
    """
    Aadhaar number Verhoeff checksum validation signal.
    Validates mathematical checksum structure without exposing or storing raw digits.
    """
    available: bool = False
    valid: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": bool(self.available),
            "valid": bool(self.valid) if self.valid is not None else None,
        }


@dataclass
class NormalizedSignals:
    """
    Standard normalized signal representation extracted from OCR processing results.
    """
    document: DocumentSignal = field(default_factory=DocumentSignal)
    ocr: OCRSignal = field(default_factory=OCRSignal)
    qr: QRSignal = field(default_factory=QRSignal)
    cross_validation: CrossValidationSignal = field(default_factory=CrossValidationSignal)
    aadhaar_checksum: AadhaarChecksumSignal = field(default_factory=AadhaarChecksumSignal)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "ocr": self.ocr.to_dict(),
            "qr": self.qr.to_dict(),
            "cross_validation": self.cross_validation.to_dict(),
            "aadhaar_checksum": self.aadhaar_checksum.to_dict(),
            "warnings": [sanitize_pii_string(str(w)) for w in self.warnings],
        }


@dataclass
class RiskFinding:
    """
    Structured explainable finding for an individual triggered risk rule.
    """
    rule_id: str
    severity: str
    points: int
    triggered: bool = True
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": str(self.rule_id),
            "severity": str(self.severity),
            "points": int(self.points),
            "triggered": bool(self.triggered),
            "reason": sanitize_pii_string(str(self.reason)),
            "evidence": {
                k: sanitize_pii_string(v) if isinstance(v, str) else v
                for k, v in self.evidence.items()
            },
        }


@dataclass
class DocumentRiskSummary:
    """
    Structured summary of the final document risk scoring and decision.
    """
    score: Optional[int]
    level: str  # "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
    decision: str  # "PASS" | "REVIEW"
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": int(self.score) if self.score is not None else None,
            "level": str(self.level),
            "decision": str(self.decision),
            "summary": sanitize_pii_string(str(self.summary)),
        }


@dataclass
class RiskAssessmentResult:
    """
    Standard JSON-serializable assessment result container.
    """
    success: bool
    risk: DocumentRiskSummary
    findings: List[Dict[str, Any]] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts assessment to a JSON-serializable dictionary with PII protection.
        """
        data: Dict[str, Any] = {
            "success": bool(self.success),
            "risk": self.risk.to_dict(),
            "findings": [dict(f) for f in self.findings],
            "flags": [str(f) for f in self.flags],
            "signals": dict(self.signals),
            "warnings": [sanitize_pii_string(str(w)) for w in self.warnings],
            "error": (
                {
                    k: sanitize_pii_string(v) if isinstance(v, str) else v
                    for k, v in self.error.items()
                }
                if self.error is not None
                else None
            ),
        }
        if self.status is not None:
            data["status"] = str(self.status)
        return data
