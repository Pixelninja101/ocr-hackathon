"""
Aadhaar Document Detection / Classification Layer.
Evaluates multi-signal textual and visual evidence to determine document type (AADHAAR, NOT_AADHAAR, UNKNOWN).
Produces explainable confidence scores and structured evidence without making legal authenticity claims.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from document_processor.config import (
    DOCUMENT_CONFIDENCE_THRESHOLD,
    mask_sensitive_number,
)

logger = logging.getLogger(__name__)

# --- Keywords & Signal Definitions ---

# Signal A: English Aadhaar specific terms
STRONG_ENGLISH_AADHAAR_TERMS: list[str] = [
    "aadhaar",
    "aadhar",
    "uidai",
    "unique identification authority of india",
    "unique identification authority",
    "unique identification",
    "mera aadhaar",
    "meri pehchan",
    "help@uidai.gov.in",
    "www.uidai.gov.in",
]

# Signal B: Hindi Aadhaar specific terms (Unicode-safe)
STRONG_HINDI_AADHAAR_TERMS: list[str] = [
    "आधार",
    "भारतीय विशिष्ट पहचान प्राधिकरण",
    "विशिष्ट पहचान प्राधिकरण",
    "विशिष्ट पहचान",
    "मेरा आधार",
    "मेरी पहचान",
    "आम आदमी का अधिकार",
]

# Signal D: Supporting field labels (DOB, Gender, Enrollment, VID)
SUPPORTING_AADHAAR_LABELS: list[str] = [
    "date of birth",
    "year of birth",
    "dob",
    "yob",
    "जन्म तिथि",
    "जन्म तारीख",
    "जन्म वर्ष",
    "जन्म का वर्ष",
    "male",
    "female",
    "transgender",
    "पुरुष",
    "महिला",
    "enrolment no",
    "नामांकन संख्या",
    "vid",
]

# Generic Government terms (Present on many Indian ID cards: PAN, Voter ID, DL, Passport)
# Must NOT be used alone as sufficient evidence for Aadhaar.
GENERIC_GOVT_TERMS: list[str] = [
    "government of india",
    "govt of india",
    "भारत सरकार",
    "government document",
    "republic of india",
]

# Competing Identity Document Disqualifiers (Explicit non-Aadhaar signals)
NON_AADHAAR_DOC_SIGNALS: dict[str, list[str]] = {
    "pan_card": [
        "income tax department",
        "permanent account number",
        "pan card",
        "आयकर विभाग",
        "स्थायी खाता संख्या",
    ],
    "driving_licence": [
        "driving licence",
        "driving license",
        "motor vehicles department",
        "transport department",
        "सारथी",
    ],
    "voter_id": [
        "election commission of india",
        "elector photo identity card",
        "epic no",
        "भारत निर्वाचन आयोग",
        "मतदाता फोटो पहचान पत्र",
    ],
    "passport": [
        "passport",
        "republic of india passport",
        "पासपोर्ट",
    ],
}

# Regex for 12-digit grouped / un-grouped / masked numbers
AADHAAR_NUMBER_REGEX = re.compile(
    r"\b(?:\d{4}\s\d{4}\s\d{4}|\d{12}|[Xx\d]{4}\s[Xx\d]{4}\s\d{4})\b"
)
VID_REGEX = re.compile(
    r"\bVID\s*:\s*(?:\d{4}\s\d{4}\s\d{4}\s\d{4}|\d{16})\b", re.IGNORECASE
)


@dataclass
class DocumentDetectionResult:
    """
    Structured, evidence-based result for document type classification.
    Adheres strictly to the separation of concerns (no authenticity/fraud claims).
    """

    document_type: str  # "aadhaar" | "not_aadhaar" | "unknown"
    confidence: float   # Deterministic score [0.0, 1.0]
    status: str         # "PASS" | "FAIL" | "UNKNOWN"
    evidence: list[dict[str, Any]] = field(default_factory=list)
    signals_detected: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-serializable dictionary representation."""
        return {
            "document_type": self.document_type,
            "type": self.document_type,  # Compatible with top-level PRD schema
            "confidence": self.confidence,
            "status": self.status,
            "evidence": self.evidence,
            "signals_detected": self.signals_detected,
            "metadata": self.metadata,
        }


def normalize_detection_text(text: Optional[str]) -> str:
    """
    Normalizes input text for multi-lingual keyword matching:
    - Applies Unicode NFKC normalization
    - Strips leading/trailing whitespace
    - Collapses repeated whitespace
    - Preserves Hindi Devanagari characters safely
    """
    if not text:
        return ""
    # Unicode NFKC normalization
    norm = unicodedata.normalize("NFKC", str(text))
    # Replace newlines and tabs with spaces
    norm = re.sub(r"[\r\n\t]+", " ", norm)
    # Collapse multiple spaces
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def _find_matching_keywords(text_lower: str, keyword_list: list[str]) -> list[str]:
    """Finds all keywords from keyword_list present in text_lower."""
    matches: list[str] = []
    for kw in keyword_list:
        kw_norm = kw.lower()
        if kw_norm in text_lower:
            matches.append(kw)
    return matches


def detect_document_type(
    image: Optional[np.ndarray] = None,
    ocr_text: Optional[str] = None,
    qr_detected: bool = False,
    qr_is_aadhaar: bool = False,
    confidence_threshold: float = DOCUMENT_CONFIDENCE_THRESHOLD,
) -> DocumentDetectionResult:
    """
    Main document classification entrypoint.
    Combines text evidence, number patterns, layout terms, and QR presence to determine
    if the document has characteristics consistent with an Aadhaar card.

    Returns:
        DocumentDetectionResult with status ("PASS" | "FAIL" | "UNKNOWN")
        and document_type ("aadhaar" | "not_aadhaar" | "unknown").
    """
    raw_text = ocr_text or ""
    normalized_text = normalize_detection_text(raw_text)
    text_lower = normalized_text.lower()

    evidence_items: list[dict[str, Any]] = []
    signals_detected: list[str] = []
    score: float = 0.0

    # 1. Signal A: English Aadhaar Keywords
    matched_en_kw = _find_matching_keywords(text_lower, STRONG_ENGLISH_AADHAAR_TERMS)
    if matched_en_kw:
        en_weight = min(0.50, 0.35 + (0.05 * (len(matched_en_kw) - 1)))
        score += en_weight
        signals_detected.append("aadhaar_english_text")
        evidence_items.append({
            "signal": "aadhaar_english_text",
            "detected": True,
            "confidence_weight": round(en_weight, 2),
            "matched": matched_en_kw[:3],
        })
    else:
        evidence_items.append({
            "signal": "aadhaar_english_text",
            "detected": False,
            "confidence_weight": 0.0,
        })

    # 2. Signal B: Hindi Aadhaar Keywords (Unicode-safe)
    matched_hi_kw = _find_matching_keywords(text_lower, STRONG_HINDI_AADHAAR_TERMS)
    if matched_hi_kw:
        hi_weight = min(0.50, 0.35 + (0.05 * (len(matched_hi_kw) - 1)))
        score += hi_weight
        signals_detected.append("aadhaar_hindi_text")
        evidence_items.append({
            "signal": "aadhaar_hindi_text",
            "detected": True,
            "confidence_weight": round(hi_weight, 2),
            "matched": matched_hi_kw[:3],
        })
    else:
        evidence_items.append({
            "signal": "aadhaar_hindi_text",
            "detected": False,
            "confidence_weight": 0.0,
        })

    # 3. Signal C: 12-digit Aadhaar Number Pattern
    num_match = AADHAAR_NUMBER_REGEX.search(raw_text)
    vid_match = VID_REGEX.search(raw_text)
    has_number_pattern = bool(num_match or vid_match)

    if has_number_pattern:
        # A 12-digit number alone gives +0.30 max (insufficient alone to reach threshold 0.60)
        num_weight = 0.30
        score += num_weight
        signals_detected.append("aadhaar_number_pattern")
        masked_repr = mask_sensitive_number(num_match.group(0) if num_match else "XXXX XXXX XXXX")
        evidence_items.append({
            "signal": "aadhaar_number_pattern",
            "detected": True,
            "confidence_weight": num_weight,
            "masked_pattern": masked_repr,
        })
    else:
        evidence_items.append({
            "signal": "aadhaar_number_pattern",
            "detected": False,
            "confidence_weight": 0.0,
        })

    # 4. Signal D: Supporting Field Labels
    matched_labels = _find_matching_keywords(text_lower, SUPPORTING_AADHAAR_LABELS)
    if matched_labels:
        label_weight = min(0.20, 0.10 + (0.05 * (len(matched_labels) - 1)))
        score += label_weight
        signals_detected.append("supporting_field_labels")
        evidence_items.append({
            "signal": "supporting_field_labels",
            "detected": True,
            "confidence_weight": round(label_weight, 2),
            "count": len(matched_labels),
        })
    else:
        evidence_items.append({
            "signal": "supporting_field_labels",
            "detected": False,
            "confidence_weight": 0.0,
        })

    # 5. Signal E: QR Code Presence & Aadhaar Payload Attributes
    if qr_is_aadhaar:
        qr_weight = 0.35
        score += qr_weight
        signals_detected.append("aadhaar_qr_payload")
        evidence_items.append({
            "signal": "qr_code",
            "detected": True,
            "confidence_weight": qr_weight,
            "is_aadhaar_payload": True,
        })
    elif qr_detected:
        qr_weight = 0.15
        score += qr_weight
        signals_detected.append("qr_present")
        evidence_items.append({
            "signal": "qr_code",
            "detected": True,
            "confidence_weight": qr_weight,
            "is_aadhaar_payload": False,
        })
    else:
        evidence_items.append({
            "signal": "qr_code",
            "detected": False,
            "confidence_weight": 0.0,
        })

    # 6. Negative Signal Analysis: Explicit Non-Aadhaar Document Detection
    competing_doc_found = False
    competing_doc_name = ""
    for doc_name, kw_list in NON_AADHAAR_DOC_SIGNALS.items():
        matched_comp = _find_matching_keywords(text_lower, kw_list)
        if matched_comp:
            competing_doc_found = True
            competing_doc_name = doc_name
            signals_detected.append(f"competing_doc:{doc_name}")
            break

    # 7. Generic Government Keywords Check
    matched_generic_govt = _find_matching_keywords(text_lower, GENERIC_GOVT_TERMS)
    has_strong_aadhaar = bool(matched_en_kw or matched_hi_kw or qr_is_aadhaar)

    if matched_generic_govt:
        evidence_items.append({
            "signal": "generic_govt_text",
            "detected": True,
            "matched": matched_generic_govt[:2],
        })
        # Crucial False-Positive Rule: If ONLY generic govt text is present and NO strong Aadhaar anchors:
        if not has_strong_aadhaar and not has_number_pattern:
            # Must cap score very low so it cannot pass
            score = min(score + 0.10, 0.25)
        else:
            score += 0.05

    # If explicit competing non-Aadhaar document text found without strong Aadhaar anchors:
    if competing_doc_found and not has_strong_aadhaar:
        final_confidence = 0.10
        doc_type = "not_aadhaar"
        status = "FAIL"
    else:
        final_confidence = max(0.0, min(round(score, 2), 0.99))
        if final_confidence >= confidence_threshold:
            doc_type = "aadhaar"
            status = "PASS"
        elif final_confidence >= 0.30:
            doc_type = "unknown"
            status = "UNKNOWN"
        else:
            doc_type = "not_aadhaar" if competing_doc_found else "unknown"
            status = "FAIL" if competing_doc_found else "UNKNOWN"

    metadata = {
        "text_length": len(raw_text),
        "qr_detected": qr_detected,
        "qr_is_aadhaar": qr_is_aadhaar,
        "competing_document_detected": competing_doc_name if competing_doc_found else None,
    }

    # Safe log without sensitive PII
    logger.info(
        "Document detection: type=%s, conf=%.2f, status=%s, signals=%s",
        doc_type,
        final_confidence,
        status,
        signals_detected,
    )

    return DocumentDetectionResult(
        document_type=doc_type,
        confidence=final_confidence,
        status=status,
        evidence=evidence_items,
        signals_detected=signals_detected,
        metadata=metadata,
    )


# Convenience aliases
def detect_aadhaar(
    image: Optional[np.ndarray] = None,
    ocr_text: Optional[str] = None,
    qr_detected: bool = False,
    qr_is_aadhaar: bool = False,
) -> DocumentDetectionResult:
    """Convenience alias for detect_document_type()."""
    return detect_document_type(
        image=image,
        ocr_text=ocr_text,
        qr_detected=qr_detected,
        qr_is_aadhaar=qr_is_aadhaar,
    )


def detect_aadhaar_document(
    raw_ocr_text: str,
    qr_detected: bool = False,
    qr_is_aadhaar: bool = False,
) -> dict[str, Any]:
    """
    Legacy wrapper returning dictionary format for processor pipeline integration.
    """
    res = detect_document_type(
        ocr_text=raw_ocr_text,
        qr_detected=qr_detected,
        qr_is_aadhaar=qr_is_aadhaar,
    )
    return res.to_dict()
