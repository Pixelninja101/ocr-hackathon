"""
Signal extraction and normalization module for Risk Engine.
Converts OCR/QR processing outputs into a structured, PII-safe, normalized signal model.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

from risk_engine.models import (
    AadhaarChecksumSignal,
    CrossValidationDOBSignal,
    CrossValidationGenderSignal,
    CrossValidationNameSignal,
    CrossValidationSignal,
    DocumentSignal,
    NormalizedSignals,
    OCRDOBFieldSignal,
    OCRFieldSignal,
    OCRFieldsSignal,
    OCRSignal,
    QRSignal,
    QRVerificationStatus,
    sanitize_pii_string,
)
from risk_engine.verhoeff import validate_aadhaar_checksum

logger = logging.getLogger("risk_engine.signals")


def _extract_document_signal(doc_raw: Any) -> DocumentSignal:
    """Extracts document-level classification and confidence."""
    if not isinstance(doc_raw, dict):
        return DocumentSignal(type="unknown", confidence=0.0)

    doc_type = doc_raw.get("type", "unknown")
    if doc_type not in ("aadhaar", "not_aadhaar", "unknown"):
        doc_type = "unknown"

    raw_conf = doc_raw.get("confidence")
    try:
        doc_conf = float(raw_conf) if raw_conf is not None else 0.0
    except (ValueError, TypeError):
        doc_conf = 0.0

    return DocumentSignal(
        type=str(doc_type),
        confidence=round(max(0.0, min(doc_conf, 1.0)), 4),
    )


def _extract_ocr_field_signal(field_raw: Any) -> OCRFieldSignal:
    """Extracts metadata for a standard OCR field without copying raw PII values."""
    if not isinstance(field_raw, dict) or not field_raw.get("value"):
        return OCRFieldSignal(available=False, confidence=None)

    raw_conf = field_raw.get("confidence")
    conf: Optional[float] = None
    if raw_conf is not None:
        try:
            conf = round(float(raw_conf), 4)
        except (ValueError, TypeError):
            conf = None

    return OCRFieldSignal(
        available=True,
        confidence=conf,
    )


def _extract_ocr_dob_signal(dob_raw: Any) -> OCRDOBFieldSignal:
    """Extracts metadata for DOB OCR field including precision."""
    if not isinstance(dob_raw, dict):
        return OCRDOBFieldSignal(available=False, confidence=None, precision=None)

    has_value = bool(dob_raw.get("year") or dob_raw.get("value"))
    if not has_value:
        return OCRDOBFieldSignal(available=False, confidence=None, precision=None)

    raw_conf = dob_raw.get("confidence")
    conf: Optional[float] = None
    if raw_conf is not None:
        try:
            conf = round(float(raw_conf), 4)
        except (ValueError, TypeError):
            conf = None

    raw_prec = dob_raw.get("precision")
    if raw_prec in ("full", "year"):
        precision: Optional[str] = str(raw_prec)
    elif dob_raw.get("day") is not None and dob_raw.get("month") is not None:
        precision = "full"
    else:
        precision = "year"

    return OCRDOBFieldSignal(
        available=True,
        confidence=conf,
        precision=precision,
    )


def _extract_ocr_signal(ocr_raw: Any) -> OCRSignal:
    """Extracts OCR aggregate confidence, language, and individual field signals."""
    if not isinstance(ocr_raw, dict):
        return OCRSignal()

    raw_conf = ocr_raw.get("confidence")
    conf: Optional[float] = None
    if raw_conf is not None:
        try:
            conf = round(float(raw_conf), 4)
        except (ValueError, TypeError):
            conf = None

    lang = str(ocr_raw["language"]) if ocr_raw.get("language") is not None else None

    fields_raw = ocr_raw.get("fields") if isinstance(ocr_raw.get("fields"), dict) else {}

    name_sig = _extract_ocr_field_signal(fields_raw.get("name"))
    dob_sig = _extract_ocr_dob_signal(fields_raw.get("dob"))
    gender_sig = _extract_ocr_field_signal(fields_raw.get("gender"))
    aadhaar_sig = _extract_ocr_field_signal(fields_raw.get("aadhaar_number"))
    address_sig = _extract_ocr_field_signal(fields_raw.get("address"))

    fields_signal = OCRFieldsSignal(
        name=name_sig,
        dob=dob_sig,
        gender=gender_sig,
        aadhaar_number=aadhaar_sig,
        address=address_sig,
    )

    return OCRSignal(
        confidence=conf,
        language=lang,
        fields=fields_signal,
    )


def _extract_qr_signal(qr_raw: Any) -> QRSignal:
    """
    Extracts QR code signals adhering to PRD cryptographic scope rules:
    qr.verified == False does NOT imply failed verification; it indicates cryptographic verification is unavailable.
    """
    if not isinstance(qr_raw, dict):
        return QRSignal(
            detected=False,
            decoded=False,
            verified=False,
            verification_status=QRVerificationStatus.QR_NOT_DETECTED.value,
        )

    detected = bool(qr_raw.get("detected", False))
    decoded = bool(qr_raw.get("decoded", False))
    verified = bool(qr_raw.get("verified", False))

    if not detected:
        status = QRVerificationStatus.QR_NOT_DETECTED.value
    elif not decoded:
        status = QRVerificationStatus.QR_DETECTED_NOT_DECODED.value
    elif verified:
        status = QRVerificationStatus.QR_VERIFIED.value
    else:
        # Check if upstream explicitly recorded a signature verification failure
        raw_status = qr_raw.get("verification_status")
        raw_err = qr_raw.get("error")
        if raw_status in ("FAILED", "INVALID", "SIGNATURE_INVALID") or raw_err == "SIGNATURE_MISMATCH":
            status = QRVerificationStatus.QR_VERIFICATION_FAILED.value
        else:
            status = QRVerificationStatus.QR_DECODED_VERIFICATION_UNAVAILABLE.value

    return QRSignal(
        detected=detected,
        decoded=decoded,
        verified=verified,
        verification_status=status,
    )


def _extract_cross_validation_signal(cv_raw: Any) -> CrossValidationSignal:
    """
    Extracts OCR ↔ QR cross-validation comparisons.
    Preserves explicit match=False (mismatch) vs match=None (unavailable).
    """
    if not isinstance(cv_raw, dict) or not cv_raw:
        return CrossValidationSignal(available=False)

    # 1. Name cross-validation
    name_raw = cv_raw.get("name") if isinstance(cv_raw.get("name"), dict) else None
    if name_raw is not None:
        raw_sim = name_raw.get("similarity")
        sim: Optional[float] = None
        if raw_sim is not None:
            try:
                sim = round(float(raw_sim), 4)
            except (ValueError, TypeError):
                sim = None

        if isinstance(name_raw.get("match"), bool):
            match_val: Optional[bool] = bool(name_raw["match"])
        elif sim is not None:
            # Derive match if similarity is provided without explicit match flag
            match_val = bool(sim >= 0.85)
        else:
            match_val = None

        name_sig = CrossValidationNameSignal(
            available=bool(sim is not None or match_val is not None),
            similarity=sim,
            match=match_val,
        )
    else:
        name_sig = CrossValidationNameSignal(available=False, similarity=None, match=None)

    # 2. DOB cross-validation
    dob_raw = cv_raw.get("dob") if isinstance(cv_raw.get("dob"), dict) else None
    if dob_raw is not None:
        dob_match = bool(dob_raw["match"]) if isinstance(dob_raw.get("match"), bool) else None
        dob_comp = str(dob_raw["comparison"]) if dob_raw.get("comparison") in ("full", "year") else None
        dob_sig = CrossValidationDOBSignal(
            available=bool(dob_match is not None or dob_comp is not None),
            match=dob_match,
            comparison=dob_comp,
        )
    else:
        dob_sig = CrossValidationDOBSignal(available=False, match=None, comparison=None)

    # 3. Gender cross-validation
    gender_raw = cv_raw.get("gender") if isinstance(cv_raw.get("gender"), dict) else None
    if gender_raw is not None:
        gender_match = bool(gender_raw["match"]) if isinstance(gender_raw.get("match"), bool) else None
        gender_sig = CrossValidationGenderSignal(
            available=bool(gender_match is not None),
            match=gender_match,
        )
    else:
        gender_sig = CrossValidationGenderSignal(available=False, match=None)

    overall_available = bool(name_sig.available or dob_sig.available or gender_sig.available)

    return CrossValidationSignal(
        available=overall_available,
        name=name_sig,
        dob=dob_sig,
        gender=gender_sig,
    )


def _extract_aadhaar_checksum_signal(processing_result: Optional[Dict[str, Any]]) -> AadhaarChecksumSignal:
    """
    Extracts Verhoeff checksum validation signal from the visual OCR Aadhaar number field.
    Checks if a full 12-digit number is available (not masked with 'X').
    """
    if not isinstance(processing_result, dict):
        return AadhaarChecksumSignal(available=False, valid=None)

    ocr_dict = processing_result.get("ocr")
    if not isinstance(ocr_dict, dict):
        return AadhaarChecksumSignal(available=False, valid=None)

    fields_dict = ocr_dict.get("fields")
    if not isinstance(fields_dict, dict):
        return AadhaarChecksumSignal(available=False, valid=None)

    aadhaar_field = fields_dict.get("aadhaar_number")
    if not isinstance(aadhaar_field, dict):
        return AadhaarChecksumSignal(available=False, valid=None)

    raw_val = aadhaar_field.get("value")
    if not raw_val or not isinstance(raw_val, (str, int)):
        return AadhaarChecksumSignal(available=False, valid=None)

    val_str = str(raw_val).strip()
    if not val_str or "X" in val_str.upper():
        # Masked number (e.g. XXXX XXXX 1234) -> checksum unavailable
        return AadhaarChecksumSignal(available=False, valid=None)

    clean_digits = "".join(ch for ch in val_str if ch.isdigit())
    if len(clean_digits) == 12:
        is_valid = validate_aadhaar_checksum(clean_digits)
        return AadhaarChecksumSignal(available=True, valid=is_valid)
    elif len(clean_digits) > 0:
        # Non-masked value was provided but does not have 12 digits -> invalid format
        return AadhaarChecksumSignal(available=True, valid=False)

    return AadhaarChecksumSignal(available=False, valid=None)


def extract_normalized_signals(
    processing_result: Optional[Dict[str, Any]]
) -> NormalizedSignals:
    """
    Main extraction function returning structured NormalizedSignals container.
    """
    if processing_result is None or not isinstance(processing_result, dict):
        return NormalizedSignals(
            document=DocumentSignal(type="unknown", confidence=0.0),
            warnings=["Input is None or not a dictionary."],
        )

    # If upstream processing returned success=False
    if not processing_result.get("success", False):
        raw_warnings = processing_result.get("warnings", [])
        warnings_list = [
            sanitize_pii_string(str(w)) for w in raw_warnings if w is not None
        ] if isinstance(raw_warnings, list) else []

        return NormalizedSignals(
            document=DocumentSignal(type="unknown", confidence=0.0),
            warnings=[w for w in warnings_list if w],
        )

    doc_signal = _extract_document_signal(processing_result.get("document"))
    ocr_signal = _extract_ocr_signal(processing_result.get("ocr"))
    qr_signal = _extract_qr_signal(processing_result.get("qr"))
    cv_signal = _extract_cross_validation_signal(processing_result.get("cross_validation"))
    chk_signal = _extract_aadhaar_checksum_signal(processing_result)

    raw_warnings = processing_result.get("warnings", [])
    warnings_list = [
        sanitize_pii_string(str(w)) for w in raw_warnings if w is not None
    ] if isinstance(raw_warnings, list) else []

    return NormalizedSignals(
        document=doc_signal,
        ocr=ocr_signal,
        qr=qr_signal,
        cross_validation=cv_signal,
        aadhaar_checksum=chk_signal,
        warnings=[w for w in warnings_list if w],
    )


def extract_signals(
    processing_result: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Public signal extraction function returning a JSON-serializable dictionary.
    Guarantees input immutability and complete PII safety.
    """
    if processing_result is not None and isinstance(processing_result, dict):
        input_copy = copy.deepcopy(processing_result)
    else:
        input_copy = None

    normalized = extract_normalized_signals(input_copy)
    return normalized.to_dict()


# Convenience alias for backwards compatibility
extract_document_signals = extract_signals
