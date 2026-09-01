"""
Explainable Risk Rules Module for Risk Engine.
Evaluates normalized document verification signals and produces structured risk findings.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from risk_engine.models import (
    NormalizedSignals,
    RiskFinding,
    RuleSeverity,
)

logger = logging.getLogger("risk_engine.rules")


def rule_document_not_identified(signals: Dict[str, Any]) -> Optional[RiskFinding]:
    """
    RULE 1: DOCUMENT_NOT_IDENTIFIED
    Triggers when the document type is not confidently identified as Aadhaar.
    """
    doc_sig = signals.get("document") if isinstance(signals.get("document"), dict) else {}
    doc_type = doc_sig.get("type", "unknown")
    doc_conf = float(doc_sig.get("confidence", 0.0)) if doc_sig.get("confidence") is not None else 0.0

    if doc_type != "aadhaar":
        return RiskFinding(
            rule_id="DOCUMENT_NOT_IDENTIFIED",
            severity=RuleSeverity.MEDIUM.value,
            points=30,
            triggered=True,
            reason="The uploaded document could not be confidently identified as an Aadhaar document.",
            evidence={
                "document_type": doc_type,
                "confidence": doc_conf,
            },
        )
    return None


def rule_low_document_confidence(signals: Dict[str, Any]) -> Optional[RiskFinding]:
    """
    RULE 2: LOW_DOCUMENT_CONFIDENCE
    Triggers when document is identified as Aadhaar but with marginal confidence (< 0.80).
    """
    doc_sig = signals.get("document") if isinstance(signals.get("document"), dict) else {}
    doc_type = doc_sig.get("type", "unknown")
    doc_conf = float(doc_sig.get("confidence", 0.0)) if doc_sig.get("confidence") is not None else 0.0

    if doc_type == "aadhaar" and doc_conf < 0.80:
        return RiskFinding(
            rule_id="LOW_DOCUMENT_CONFIDENCE",
            severity=RuleSeverity.MEDIUM.value,
            points=15,
            triggered=True,
            reason="Document classification confidence is below the high-confidence threshold (0.80).",
            evidence={
                "confidence": doc_conf,
                "threshold": 0.80,
            },
        )
    return None


def rule_low_ocr_confidence(signals: Dict[str, Any]) -> Optional[RiskFinding]:
    """
    RULE 3: LOW_OCR_CONFIDENCE
    Triggers when overall OCR confidence is low (< 0.70), indicating readability issues.
    """
    ocr_sig = signals.get("ocr") if isinstance(signals.get("ocr"), dict) else {}
    raw_conf = ocr_sig.get("confidence")

    if raw_conf is not None:
        try:
            ocr_conf = float(raw_conf)
            if ocr_conf < 0.70:
                return RiskFinding(
                    rule_id="LOW_OCR_CONFIDENCE",
                    severity=RuleSeverity.MEDIUM.value,
                    points=15,
                    triggered=True,
                    reason="Overall OCR text extraction confidence is low, indicating potential image quality or readability issues.",
                    evidence={
                        "ocr_confidence": ocr_conf,
                        "threshold": 0.70,
                    },
                )
        except (ValueError, TypeError):
            pass
    return None


def rule_missing_critical_fields(signals: Dict[str, Any]) -> Optional[RiskFinding]:
    """
    RULE 4: MISSING_CRITICAL_FIELDS
    Triggers when critical identity fields (name, dob, gender, aadhaar_number) are missing.
    Address is non-critical and excluded from this rule.
    """
    ocr_sig = signals.get("ocr") if isinstance(signals.get("ocr"), dict) else {}
    fields_sig = ocr_sig.get("fields") if isinstance(ocr_sig.get("fields"), dict) else {}

    critical_fields = ["name", "dob", "gender", "aadhaar_number"]
    missing_fields: List[str] = []

    for field_name in critical_fields:
        f_data = fields_sig.get(field_name) if isinstance(fields_sig.get(field_name), dict) else {}
        if not f_data.get("available", False):
            missing_fields.append(field_name)

    count = len(missing_fields)
    if count == 0:
        return None

    if count == 1:
        severity = RuleSeverity.LOW.value
        points = 5
    elif count == 2:
        severity = RuleSeverity.MEDIUM.value
        points = 10
    elif count == 3:
        severity = RuleSeverity.HIGH.value
        points = 20
    else:  # count == 4
        severity = RuleSeverity.HIGH.value
        points = 25

    return RiskFinding(
        rule_id="MISSING_CRITICAL_FIELDS",
        severity=severity,
        points=points,
        triggered=True,
        reason=f"{count} critical identity field(s) could not be extracted from document visual OCR.",
        evidence={
            "missing_fields": missing_fields,
            "missing_count": count,
        },
    )


def rule_qr_not_detected(signals: Dict[str, Any]) -> Optional[RiskFinding]:
    """
    RULE 5: QR_NOT_DETECTED
    Triggers when no QR code was detected in the document.
    Mutually exclusive with QR_DETECTED_NOT_DECODED, QR_VERIFICATION_UNAVAILABLE, and QR_VERIFICATION_FAILED.
    """
    qr_sig = signals.get("qr") if isinstance(signals.get("qr"), dict) else {}
    detected = bool(qr_sig.get("detected", False))

    if not detected:
        return RiskFinding(
            rule_id="QR_NOT_DETECTED",
            severity=RuleSeverity.MEDIUM.value,
            points=15,
            triggered=True,
            reason="No QR code was detected on the document.",
            evidence={
                "qr_detected": False,
            },
        )
    return None


def rule_qr_detected_not_decoded(signals: Dict[str, Any]) -> Optional[RiskFinding]:
    """
    RULE 6: QR_DETECTED_NOT_DECODED
    Triggers when a QR code was detected but could not be decoded (e.g. blur, damage).
    Takes precedence over QR_NOT_DETECTED.
    """
    qr_sig = signals.get("qr") if isinstance(signals.get("qr"), dict) else {}
    detected = bool(qr_sig.get("detected", False))
    decoded = bool(qr_sig.get("decoded", False))

    if detected and not decoded:
        return RiskFinding(
            rule_id="QR_DETECTED_NOT_DECODED",
            severity=RuleSeverity.MEDIUM.value,
            points=20,
            triggered=True,
            reason="A QR code was detected on the document, but could not be successfully decoded.",
            evidence={
                "qr_detected": True,
                "qr_decoded": False,
            },
        )
    return None


def rule_qr_verification_unavailable(signals: Dict[str, Any]) -> Optional[RiskFinding]:
    """
    RULE 7: QR_VERIFICATION_UNAVAILABLE
    Triggers when QR is decoded but cryptographic signature verification is unavailable.
    Does NOT indicate a verification failure.
    """
    qr_sig = signals.get("qr") if isinstance(signals.get("qr"), dict) else {}
    v_status = qr_sig.get("verification_status")

    if v_status == "QR_DECODED_VERIFICATION_UNAVAILABLE":
        return RiskFinding(
            rule_id="QR_VERIFICATION_UNAVAILABLE",
            severity=RuleSeverity.LOW.value,
            points=5,
            triggered=True,
            reason="QR code was decoded successfully, but cryptographic authenticity verification is unavailable.",
            evidence={
                "qr_decoded": True,
                "verified": False,
            },
        )
    return None


def rule_qr_verification_failed(signals: Dict[str, Any]) -> Optional[RiskFinding]:
    """
    RULE 8: QR_VERIFICATION_FAILED
    Triggers ONLY when the normalized signal explicitly indicates a cryptographic signature verification failure.
    """
    qr_sig = signals.get("qr") if isinstance(signals.get("qr"), dict) else {}
    v_status = qr_sig.get("verification_status")

    if v_status == "QR_VERIFICATION_FAILED":
        return RiskFinding(
            rule_id="QR_VERIFICATION_FAILED",
            severity=RuleSeverity.HIGH.value,
            points=35,
            triggered=True,
            reason="Cryptographic signature verification for the QR code payload failed.",
            evidence={
                "verification_status": "QR_VERIFICATION_FAILED",
            },
        )
    return None


def rule_name_mismatch(signals: Dict[str, Any]) -> Optional[RiskFinding]:
    """
    RULE 9: NAME_MISMATCH
    Triggers when cross-validation is available and OCR name explicitly does not match QR name.
    Does not trigger if comparison is unavailable.
    """
    cv_sig = signals.get("cross_validation") if isinstance(signals.get("cross_validation"), dict) else {}
    if not cv_sig.get("available", False):
        return None

    name_cv = cv_sig.get("name") if isinstance(cv_sig.get("name"), dict) else {}
    if not name_cv.get("available", False):
        return None

    if name_cv.get("match") is False:
        similarity = name_cv.get("similarity")
        return RiskFinding(
            rule_id="NAME_MISMATCH",
            severity=RuleSeverity.HIGH.value,
            points=30,
            triggered=True,
            reason="Name extracted from document visual OCR does not match the name encoded in the QR code.",
            evidence={
                "similarity": similarity,
            },
        )
    return None


def rule_dob_mismatch(signals: Dict[str, Any]) -> Optional[RiskFinding]:
    """
    RULE 10: DOB_MISMATCH
    Triggers when cross-validation is available and OCR DOB explicitly does not match QR DOB.
    Does not trigger if comparison is unavailable.
    """
    cv_sig = signals.get("cross_validation") if isinstance(signals.get("cross_validation"), dict) else {}
    if not cv_sig.get("available", False):
        return None

    dob_cv = cv_sig.get("dob") if isinstance(cv_sig.get("dob"), dict) else {}
    if not dob_cv.get("available", False):
        return None

    if dob_cv.get("match") is False:
        comparison = dob_cv.get("comparison")
        return RiskFinding(
            rule_id="DOB_MISMATCH",
            severity=RuleSeverity.HIGH.value,
            points=30,
            triggered=True,
            reason="Date/Year of birth extracted from document visual OCR does not match the QR code.",
            evidence={
                "comparison": comparison,
            },
        )
    return None


def rule_gender_mismatch(signals: Dict[str, Any]) -> Optional[RiskFinding]:
    """
    RULE 11: GENDER_MISMATCH
    Triggers when cross-validation is available and OCR gender explicitly does not match QR gender.
    Does not trigger if comparison is unavailable.
    """
    cv_sig = signals.get("cross_validation") if isinstance(signals.get("cross_validation"), dict) else {}
    if not cv_sig.get("available", False):
        return None

    gender_cv = cv_sig.get("gender") if isinstance(cv_sig.get("gender"), dict) else {}
    if not gender_cv.get("available", False):
        return None

    if gender_cv.get("match") is False:
        return RiskFinding(
            rule_id="GENDER_MISMATCH",
            severity=RuleSeverity.MEDIUM.value,
            points=20,
            triggered=True,
            reason="Gender extracted from document visual OCR does not match the QR code.",
            evidence={
                "gender_match": False,
            },
        )
    return None


def rule_low_field_ocr_confidence(signals: Dict[str, Any]) -> Optional[RiskFinding]:
    """
    RULE 12: LOW_FIELD_OCR_CONFIDENCE
    Inspects available OCR fields and aggregates those with recognition confidence < 0.60.
    Produces a single finding with points capped at 15.
    """
    ocr_sig = signals.get("ocr") if isinstance(signals.get("ocr"), dict) else {}
    fields_sig = ocr_sig.get("fields") if isinstance(ocr_sig.get("fields"), dict) else {}

    all_fields = ["name", "dob", "gender", "aadhaar_number", "address"]
    low_confidence_fields: List[str] = []

    for field_name in all_fields:
        f_data = fields_sig.get(field_name) if isinstance(fields_sig.get(field_name), dict) else {}
        if f_data.get("available", False):
            raw_conf = f_data.get("confidence")
            if raw_conf is not None:
                try:
                    conf = float(raw_conf)
                    if conf < 0.60:
                        low_confidence_fields.append(field_name)
                except (ValueError, TypeError):
                    pass

    if not low_confidence_fields:
        return None

    points = min(len(low_confidence_fields) * 5, 15)

    return RiskFinding(
        rule_id="LOW_FIELD_OCR_CONFIDENCE",
        severity=RuleSeverity.LOW.value,
        points=points,
        triggered=True,
        reason=f"{len(low_confidence_fields)} extracted OCR field(s) have low recognition confidence (< 0.60).",
        evidence={
            "low_confidence_fields": low_confidence_fields,
            "threshold": 0.60,
        },
    )


def rule_aadhaar_checksum_invalid(signals: Dict[str, Any]) -> Optional[RiskFinding]:
    """
    RULE 13: AADHAAR_CHECKSUM_INVALID
    Triggers when an unmasked 12-digit Aadhaar number is available and fails Verhoeff checksum validation.
    Does NOT trigger if the Aadhaar number is missing, masked, or unavailable.
    """
    chk_sig = signals.get("aadhaar_checksum") if isinstance(signals.get("aadhaar_checksum"), dict) else {}

    # Must be explicitly available and invalid
    if chk_sig.get("available") and chk_sig.get("valid") is False:
        return RiskFinding(
            rule_id="AADHAAR_CHECKSUM_INVALID",
            severity=RuleSeverity.MEDIUM.value,
            points=15,
            triggered=True,
            reason="The extracted Aadhaar number failed checksum validation.",
            evidence={
                "checksum_valid": False,
            },
        )
    return None


# Ordered registry of all rule evaluators
RULES: List[Callable[[Dict[str, Any]], Optional[RiskFinding]]] = [
    rule_document_not_identified,
    rule_low_document_confidence,
    rule_low_ocr_confidence,
    rule_missing_critical_fields,
    rule_qr_not_detected,
    rule_qr_detected_not_decoded,
    rule_qr_verification_unavailable,
    rule_qr_verification_failed,
    rule_name_mismatch,
    rule_dob_mismatch,
    rule_gender_mismatch,
    rule_low_field_ocr_confidence,
    rule_aadhaar_checksum_invalid,
]


def evaluate_rules(
    signals: Union[NormalizedSignals, Dict[str, Any], None]
) -> Dict[str, Any]:
    """
    Evaluates all risk rules against extracted document signals.

    Parameters:
        signals (NormalizedSignals | dict | None): Normalized signals from signals.extract_signals()

    Returns:
        dict: {
            "findings": list of finding dicts,
            "total_points": integer sum of triggered rule points
        }
    """
    if signals is None:
        signals_dict: Dict[str, Any] = {}
    elif isinstance(signals, NormalizedSignals):
        signals_dict = signals.to_dict()
    elif isinstance(signals, dict):
        signals_dict = copy.deepcopy(signals)
    else:
        signals_dict = {}

    findings: List[Dict[str, Any]] = []
    total_points = 0

    for rule_fn in RULES:
        finding = rule_fn(signals_dict)
        if finding is not None:
            findings.append(finding.to_dict())
            total_points += finding.points

    logger.info(
        "Evaluated %d rules: %d triggered, total_points=%d",
        len(RULES),
        len(findings),
        total_points,
    )

    return {
        "findings": findings,
        "total_points": total_points,
    }
