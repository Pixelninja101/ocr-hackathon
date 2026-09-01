"""
Aadhaar Field Extraction and Normalization Layer.
Extracts Name (English & Hindi), Date of Birth, Year of Birth, Gender, Candidate Aadhaar Number, and Address block
from structured OCR results while enforcing data privacy, uncertainty tracking, and strict separation of concerns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from document_processor.config import mask_sensitive_number
from document_processor.ocr.engine import OCRResult, OCRWord
from document_processor.ocr.normalization import (
    format_dob_for_display,
    normalize_dob,
    normalize_gender,
    normalize_name,
)

logger = logging.getLogger(__name__)

# Non-name terms / headers / slogans / field labels to filter out during name candidate scanning
DISALLOWED_NAME_WORDS: set[str] = {
    "government",
    "govt",
    "india",
    "bharat",
    "sarkar",
    "authority",
    "unique",
    "identification",
    "uidai",
    "aadhaar",
    "aadhar",
    "enrolment",
    "enrollment",
    "vid",
    "help",
    "date",
    "birth",
    "year",
    "dob",
    "yob",
    "male",
    "female",
    "transgender",
    "father",
    "mother",
    "husband",
    "wife",
    "son",
    "daughter",
    "address",
    "pincode",
    "pin",
    "card",
    "mera",
    "meri",
    "pehchan",
    "adhikar",
    "aam",
    "aadmi",
    "ka",
    "ki",
    "ke",
    "se",
    "ko",
    "par",
    "to",
    "of",
    "and",
    "in",
    # Hindi keywords
    "भारत",
    "सरकार",
    "प्राधिकरण",
    "पहचान",
    "विशिष्ट",
    "आधार",
    "जन्म",
    "तिथि",
    "तारीख",
    "वर्ष",
    "पुरुष",
    "महिला",
    "स्त्री",
    "पिता",
    "माता",
    "पति",
    "पत्नी",
    "पुत्र",
    "पुत्री",
    "पता",
    "पिन",
    "मेरा",
    "मेरी",
    "अधिकार",
    "आम",
    "आदमी",
    "का",
    "की",
    "के",
    "से",
    "को",
}

DOB_PATTERNS = [
    # Full DOB: DOB: 12/04/2002 or जन्म तिथि: 12/04/2002
    re.compile(
        r"(?:DOB|Date\s*of\s*Birth|जन्म\s*तिथि|जन्म\s*तारीख)\s*[:\-–—]?\s*([0-9]{1,2}[\/\-\.\s][0-9]{1,2}[\/\-\.\s][0-9]{4})",
        re.IGNORECASE,
    ),
    # Year of Birth: 2002 or जन्म वर्ष: 2002
    re.compile(
        r"(?:Year\s*of\s*Birth|जन्म\s*का\s*वर्ष|जन्म\s*वर्ष|YOB)\s*[:\-–—]?\s*([12][90]\d\d)",
        re.IGNORECASE,
    ),
]

GENDER_PATTERNS = [
    re.compile(r"(?:Gender|लिंग|Sex)\s*[:\-–—]?\s*(MALE|FEMALE|TRANSGENDER|पुरुष|महिला|स्त्री|तृतीय\s*लिंग)", re.IGNORECASE),
    re.compile(r"\b(MALE|FEMALE|TRANSGENDER)\b", re.IGNORECASE),
    re.compile(r"(पुरुष|महिला|स्त्री|तृतीय\s*लिंग)"),
]

# Standard 4-digit grouped Aadhaar number (e.g. "1234 5678 9012" or "XXXX XXXX 9012")
AADHAAR_GROUPED_REGEX = re.compile(
    r"\b(?:\d{4}\s\d{4}\s\d{4}|[Xx\d]{4}\s[Xx\d]{4}\s\d{4}|[Xx\d]{4}\-[Xx\d]{4}\-[Xx\d]{4})\b"
)
# Standalone 12-digit number (e.g. "123456789012")
AADHAAR_CONTINUOUS_REGEX = re.compile(r"\b([2-9]\d{11})\b")
GENERIC_12_DIGIT_REGEX = re.compile(r"\b(\d{12})\b")

ADDRESS_START_KEYWORDS = [
    "address",
    "पता",
    "c/o",
    "s/o",
    "w/o",
    "d/o",
    "house no",
    "h.no",
    "flat no",
    "ward no",
    "village",
    "post",
    "dist",
    "district",
    "state",
    "pin",
    "pincode",
]


@dataclass
class ExtractedField:
    """Represents a single structured Aadhaar field with source, confidence, and status."""

    value: Any
    raw_value: Optional[str] = None
    confidence: float = 0.0  # [0.0, 1.0]
    source: str = "ocr"
    status: str = "FOUND"    # "FOUND", "NOT_FOUND", "UNCERTAIN"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-serializable dictionary representation."""
        res: dict[str, Any] = {
            "value": self.value,
            "confidence": round(self.confidence, 2),
            "source": self.source,
            "status": self.status,
        }
        if self.raw_value is not None:
            res["raw_value"] = self.raw_value
        if self.metadata:
            res.update(self.metadata)
        return res


@dataclass
class AadhaarFieldsResult:
    """Structured container for all extracted Aadhaar fields from OCR."""

    success: bool
    fields: dict[str, ExtractedField]
    warnings: list[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "warnings": self.warnings,
        }

    def __getitem__(self, item: str) -> Any:
        return self.to_dict()["fields"].get(item)


def _is_disallowed_name_token(token: str) -> bool:
    """Checks if a single word is a disallowed Aadhaar keyword or generic word."""
    cleaned = token.strip().lower()
    return cleaned in DISALLOWED_NAME_WORDS or len(cleaned) < 2


def is_valid_name_candidate(line: str) -> bool:
    """
    Validates if a text line looks like a valid person's name:
    - Contains 2 to 5 words
    - No digits or suspicious symbols
    - Not a guardian label (C/O, D/O, S/O, W/O, द्वारा)
    - Not a known Aadhaar header, field label, or address token
    """
    line_clean = line.strip()
    if not line_clean or len(line_clean) < 3 or len(line_clean) > 60:
        return False

    # Check for digits or disallowed characters
    if re.search(r"\d", line_clean):
        return False

    # Check for guardian prefixes (D/O, S/O, W/O, C/O, etc.)
    if re.match(
        r"^(?:c\/o|s\/o|w\/o|d\/o|care\s*of|son\s*of|daughter\s*of|wife\s*of|द्वारा|आत्मज|सुपुत्र|सुपुत्री|पुत्र|पुत्री|पत्नी|पिता)\b",
        line_clean,
        re.IGNORECASE,
    ):
        return False

    # Check for common address keywords (street, colony, thatipur, vtc, po, dist, state, pin, etc.)
    if re.search(
        r"\b(?:colony|nagar|road|street|lane|enclave|vihar|puram|bazar|bazaar|chowk|haveli|society|apartment|flat|ward|village|town|city|vtc|po|post|sub\s*district|tehsil|taluk|district|dist|state|pincode|pin|मकान|गली|वार्ड|नगर|कॉलोनी|विहार|बाज़ार|चौक|हवेली|सोसायटी|ग्राम|गाँव|शहर|डाकघर|तहसील|ज़िला|जिला|राज्य)\b",
        line_clean,
        re.IGNORECASE,
    ):
        return False

    # Extract alphabetic words (English or Devanagari)
    words = re.findall(r"[A-Za-z\u0900-\u097F]+", line_clean)
    if not words or len(words) > 5:
        return False

    # If all or most words are disallowed keywords, reject
    disallowed_count = sum(1 for w in words if _is_disallowed_name_token(w))
    if disallowed_count >= len(words) or (len(words) > 1 and disallowed_count > 0):
        return False

    return True


def extract_name_from_lines(
    lines: list[str],
    default_confidence: float = 0.90,
    words_data: Optional[list[OCRWord]] = None,
) -> Optional[dict[str, Any]]:
    """
    Extracts person's name using contextual layout & bilingual lines.
    In Aadhaar:
    - Hindi and English names typically appear adjacent or above DOB/Gender.
    - Explicit label 'Name:' or 'नाम:' may precede the candidate.
    """
    english_name: Optional[str] = None
    hindi_name: Optional[str] = None
    detected_conf = default_confidence

    # 1. Check for explicit 'Name:' or 'नाम:' label
    for line in lines:
        match_en = re.search(r"(?:Name|NAME)\s*[:\-–—]\s*([A-Za-z\s]{3,40})", line)
        if match_en:
            cand = match_en.group(1).strip()
            if is_valid_name_candidate(cand):
                english_name = re.sub(r"\s+", " ", cand).strip().upper()
                detected_conf = min(0.95, default_confidence + 0.05)

        match_hi = re.search(r"(?:नाम|नांव)\s*[:\-–—]\s*([\u0900-\u097F\s]{3,40})", line)
        if match_hi:
            cand = match_hi.group(1).strip()
            if is_valid_name_candidate(cand):
                hindi_name = re.sub(r"\s+", " ", cand).strip()

    # 2. Contextual scan relative to DOB line
    dob_line_idx = -1
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ["dob", "date of birth", "year of birth", "जन्म तिथि", "जन्म वर्ष"]):
            dob_line_idx = i
            break

    if dob_line_idx > 0:
        # Check lines immediately preceding DOB
        for offset in range(1, min(5, dob_line_idx + 1)):
            candidate_line = lines[dob_line_idx - offset].strip()
            # Check English name candidate
            if not english_name and is_valid_name_candidate(candidate_line) and re.search(r"[A-Za-z]", candidate_line):
                english_name = re.sub(r"\s+", " ", candidate_line).strip().upper()
            # Check Hindi name candidate
            elif not hindi_name and is_valid_name_candidate(candidate_line) and re.search(r"[\u0900-\u097F]", candidate_line):
                hindi_name = re.sub(r"\s+", " ", candidate_line).strip()

    # 3. Top-down fallback scan
    if not english_name:
        for line in lines[:10]:
            cand = line.strip()
            if is_valid_name_candidate(cand) and re.search(r"[A-Za-z]", cand):
                english_name = re.sub(r"\s+", " ", cand).strip().upper()
                detected_conf = max(0.65, default_confidence * 0.80)
                break

    if not hindi_name:
        for line in lines[:10]:
            cand = line.strip()
            if is_valid_name_candidate(cand) and re.search(r"[\u0900-\u097F]", cand):
                hindi_name = re.sub(r"\s+", " ", cand).strip()
                break

    if english_name or hindi_name:
        primary_val = english_name if english_name else hindi_name
        return {
            "value": primary_val,
            "english": english_name,
            "hindi": hindi_name,
            "confidence": round(detected_conf, 2),
            "status": "FOUND" if detected_conf >= 0.60 else "UNCERTAIN",
        }

    return None


NON_DOB_PREFIX_KEYWORDS: list[str] = [
    "details as on",
    "details on",
    "as on",
    "generated on",
    "generation date",
    "date of generation",
    "issue date",
    "date of issue",
    "issued on",
    "download date",
    "date of download",
    "downloaded on",
    "print date",
    "printed on",
    "printing date",
    "date of print",
    "valid up to",
    "valid thru",
    "valid through",
    "expiry date",
    "date of expiry",
    "enrolment date",
    "enrollment date",
    "update date",
    "updated on",
    "जारी करने की तिथि",
    "जारी करने का दिनांक",
    "जारी तिथि",
    "डाउनलोड तिथि",
    "मुद्रण तिथि",
    "विवरण दिनांक",
    "नामांकन तिथि",
    "अद्यतन तिथि",
]


def _is_excluded_date_context(prefix_text: str, line_text: str) -> bool:
    """
    Checks if the context around a date indicates it is an issue, print, download,
    enrolment, or generation date rather than a date of birth.
    """
    prefix_lower = prefix_text.lower()
    line_lower = line_text.lower()
    return any(kw in prefix_lower or kw in line_lower for kw in NON_DOB_PREFIX_KEYWORDS)


def extract_dob_from_text(
    full_text: str, default_confidence: float = 0.90
) -> Optional[dict[str, Any]]:
    """
    Extracts and validates Date of Birth or Year of Birth from OCR text.
    Distinguishes full date ('2002-04-12') from year-only ('2002').
    Explicit labels (DOB, Date of Birth, जन्म तिथि, YOB, Year of Birth, जन्म वर्ष) are prioritized.
    Guards against misclassifying generation, download, print, or issue dates as DOB.
    """
    if not full_text:
        return None

    # 1. Explicit DOB / YOB pattern scanning
    for pattern in DOB_PATTERNS:
        match = pattern.search(full_text)
        if match:
            start_pos = match.start()
            line_start = full_text.rfind("\n", 0, start_pos)
            prefix_text = full_text[max(0, line_start + 1):start_pos]
            line_end = full_text.find("\n", match.end())
            line_text = full_text[max(0, line_start + 1):line_end if line_end != -1 else len(full_text)]

            if not _is_excluded_date_context(prefix_text, line_text):
                raw_val = match.group(1)
                norm = normalize_dob(raw_val)
                if norm:
                    is_full = norm["precision"] == "full"
                    return {
                        "value": norm["iso"] if is_full else None,
                        "year_of_birth": str(norm["year"]),
                        "year": norm["year"],
                        "month": norm.get("month"),
                        "day": norm.get("day"),
                        "precision": norm["precision"],
                        "raw_value": raw_val,
                        "confidence": round(default_confidence, 2),
                        "status": "FOUND",
                    }

    # 2. Context-aware fallback:
    # Only triggered if the document has explicit Aadhaar front-side identity cues (Gender, Aadhaar header)
    # AND the candidate date is NOT preceded/labelled by non-DOB exclusion phrases or address lines.
    has_identity_context = bool(
        re.search(r"\b(?:gender|sex|लिंग|male|female|पुरुष|महिला|transgender)\b", full_text, re.IGNORECASE)
        or re.search(r"\b(?:aadhaar|aadhar|आधार|uidai|unique identification|मेरा आधार)\b", full_text, re.IGNORECASE)
    )

    if has_identity_context:
        date_pattern = re.compile(
            r"\b(0?[1-9]|[12][0-9]|3[01])[\/\-\.\s](0?[1-9]|1[012])[\/\-\.\s](19\d\d|20\d\d)\b"
        )
        for date_match in date_pattern.finditer(full_text):
            start_pos = date_match.start()
            line_start = full_text.rfind("\n", 0, start_pos)
            line_end = full_text.find("\n", date_match.end())
            line_text = full_text[max(0, line_start + 1):line_end if line_end != -1 else len(full_text)]
            prefix_text = full_text[max(0, line_start + 1):start_pos]

            # Reject if preceded by non-DOB exclusion phrases
            if _is_excluded_date_context(prefix_text, line_text):
                continue

            # Reject if inside an address line
            if any(kw in line_text.lower() for kw in ["address", "पता", "pincode", "pin ", "ward", "dist", "c/o", "s/o", "w/o", "d/o", "house"]):
                continue

            raw_val = date_match.group(0)
            norm = normalize_dob(raw_val)
            if norm and norm.get("year"):
                return {
                    "value": norm["iso"],
                    "year_of_birth": str(norm["year"]),
                    "year": norm["year"],
                    "month": norm.get("month"),
                    "day": norm.get("day"),
                    "precision": "full",
                    "raw_value": raw_val,
                    "confidence": round(default_confidence * 0.80, 2),
                    "status": "FOUND" if default_confidence * 0.80 >= 0.60 else "UNCERTAIN",
                }

    return None


def extract_gender_from_text(
    full_text: str, default_confidence: float = 0.90
) -> Optional[dict[str, Any]]:
    """
    Extracts and normalizes Gender from English or Hindi OCR text.
    """
    for pattern in GENDER_PATTERNS:
        match = pattern.search(full_text)
        if match:
            raw_val = match.group(1)
            norm = normalize_gender(raw_val)
            if norm:
                return {
                    "value": norm,
                    "raw_value": raw_val,
                    "confidence": round(default_confidence, 2),
                    "status": "FOUND",
                }
    return None


def extract_aadhaar_number_from_text(
    full_text: str, default_confidence: float = 0.95
) -> Optional[dict[str, Any]]:
    """
    Detects candidate 12-digit Aadhaar number with formatting tolerance.
    Distinguishes standard grouped formats from ambiguous continuous numbers.
    Never logs unmasked 12-digit numbers.
    """
    # 1. Look for standard grouped format (e.g. '1234 5678 9012' or 'XXXX XXXX 1234')
    match_grouped = AADHAAR_GROUPED_REGEX.search(full_text)
    if match_grouped:
        raw_val = match_grouped.group(0)
        clean_val = "".join(ch for ch in raw_val if ch.isdigit() or ch.upper() == "X")
        if len(clean_val) == 12:
            masked = mask_sensitive_number(clean_val)
            return {
                "value": clean_val,
                "masked_value": masked,
                "raw_value": raw_val,
                "confidence": round(default_confidence, 2),
                "status": "FOUND",
                "format": "grouped_12_digit",
            }

    # 2. Check for continuous 12-digit candidate
    match_cont = AADHAAR_CONTINUOUS_REGEX.search(full_text)
    if match_cont:
        raw_val = match_cont.group(1)
        # Check context: is there an Aadhaar keyword in proximity?
        has_aadhaar_ctx = bool(re.search(r"(?:aadhaar|aadhar|आधार|uid|uidai|mera|meri)", full_text, re.IGNORECASE))
        conf = default_confidence * 0.90 if has_aadhaar_ctx else 0.50
        masked = mask_sensitive_number(raw_val)
        return {
            "value": raw_val,
            "masked_value": masked,
            "raw_value": raw_val,
            "confidence": round(conf, 2),
            "status": "FOUND" if conf >= 0.60 else "UNCERTAIN",
            "format": "continuous_12_digit",
        }

    return None


ADDRESS_START_PATTERNS = [
    re.compile(r"^(?:address|पता|पत्ता)\s*[:\-–—]?\s*", re.IGNORECASE),
    re.compile(r"^(?:c\/o|s\/o|w\/o|d\/o|care\s*of|son\s*of|daughter\s*of|wife\s*of)\s*[:\-–—]?\s*", re.IGNORECASE),
    re.compile(r"^(?:द्वारा|आत्मज|सुपुत्र|सुपुत्री|पुत्र|पुत्री|पत्नी|पिता)\s*[:\-–—]?\s*"),
    re.compile(r"^(?:house\s*no|h\.no|flat\s*no|plot\s*no|ward\s*no|मकान\s*नं)\s*[:\-–—]?\s*", re.IGNORECASE),
]

ADDRESS_STOP_PATTERNS = [
    # Aadhaar numbers & VIDs
    AADHAAR_GROUPED_REGEX,
    AADHAAR_CONTINUOUS_REGEX,
    re.compile(r"(?:आपका\s*आधार\s*क्रमांक|your\s*aadhaar\s*no|vid\s*[:\-])", re.IGNORECASE),
    # UIDAI footer & metadata
    re.compile(r"(?:www\.uidai|help@uidai|1947|unique\s*identification\s*authority|भारतीय\s*विशिष्ट\s*पहचान)", re.IGNORECASE),
    re.compile(r"(?:government\s*of\s*india|भारत\s*सरकार)", re.IGNORECASE),
    re.compile(r"(?:सूचना\s*\/|information\b|आधार\s*पहचान\s*का\s*प्रमाण|aadhaar\s*is\s*proof)", re.IGNORECASE),
    # Issue / generation / download metadata
    re.compile(r"(?:details\s*as\s*on|generated\s*on|issue\s*date|date\s*of\s*issue|download\s*date|date\s*of\s*download|printed\s*on|विवरण\s*दिनांक|जारी\s*करने\s*की\s*तिथि|डाउनलोड\s*तिथि|मुद्रण\s*तिथि)", re.IGNORECASE),
    # Enrolment / To headers
    re.compile(r"(?:enrolment\s*no|नामांकन\s*क्रम|नामांकन\s*संख्या|\bto\b)", re.IGNORECASE),
    # DOB / YOB / Gender identity field boundaries (with/without bilingual slashes)
    re.compile(r"(?:dob|date\s*of\s*birth|year\s*of\s*birth|yob|जन्म\s*तिथि|जन्म\s*तारीख|जन्म\s*वर्ष|जन्म\s*का\s*वर्ष)\s*[:\-–—/]", re.IGNORECASE),
    re.compile(r"(?:gender|लिंग|sex)\s*[:\-–—/]", re.IGNORECASE),
    re.compile(r"^\s*(?:male|female|transgender|पुरुष|महिला|स्त्री|तृतीय\s*लिंग)\s*(?:\/|\-|\–|\—|\s|$)", re.IGNORECASE),
    # Standard legal disclaimer
    re.compile(r"(?:इसका\s*उपयोग\s*सत्यापन|इस\s*आधार\s*पत्र\s*को|offline\s*xml|valid\s*thru|valid\s*through)", re.IGNORECASE),
]


def extract_address_block(
    full_text: str, default_confidence: float = 0.75
) -> Optional[dict[str, Any]]:
    """
    Extracts multi-line postal address block triggered by 'Address', 'पता', 'C/O', 'D/O', etc.
    Collects consecutive address lines across English and Hindi until boundary or footer text.
    Understands address labels (C/O, S/O, D/O, W/O, VTC, PO, Sub District, District, State, PIN Code).
    """
    if not full_text:
        return None

    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    address_lines: list[str] = []
    collecting = False
    has_seen_pincode = False
    has_explicit_address_label = False

    for line in lines:
        # 1. Check start of address section
        if not collecting:
            is_start = False
            clean_line = line
            for start_pat in ADDRESS_START_PATTERNS:
                if start_pat.search(line):
                    is_start = True
                    # If it's an explicit "Address:" or "पता:" label, remove the prefix label
                    if re.match(r"^(?:address|पता|पत्ता)\s*[:\-–—]?\s*", line, re.IGNORECASE):
                        has_explicit_address_label = True
                        clean_line = re.sub(r"^(?:address|पता|पत्ता)\s*[:\-–—]?\s*", "", line, flags=re.IGNORECASE).strip()
                    break

            if is_start:
                collecting = True
                if clean_line:
                    address_lines.append(clean_line)
                    if re.search(r"\b\d{6}\b", clean_line):
                        has_seen_pincode = True
                continue

        # 2. When collecting lines
        if collecting:
            # Check stop boundary patterns
            if any(pat.search(line) for pat in ADDRESS_STOP_PATTERNS):
                break

            # If we already encountered a 6-digit PIN code and the next line is non-alphabetic noise
            if has_seen_pincode:
                if not re.search(r"[A-Za-z\u0900-\u097F]{3,}", line):
                    break

            # If line is an OCR barcode / symbol noise line (e.g. "=o 5190595", "= =", "1/1")
            if re.match(r"^[\=\-\_\~\|\:\.\s\d]{1,10}$", line) or re.match(r"^=\w+\s+\d+$", line):
                if len(address_lines) >= 3 or has_seen_pincode:
                    break
                continue

            address_lines.append(line)
            if re.search(r"\b\d{6}\b", line):
                has_seen_pincode = True

            # Safety cap for extremely long blocks
            if len(address_lines) >= 16:
                break

    if address_lines:
        combined_address = " ".join(address_lines)
        combined_address = re.sub(r"\s+", " ", combined_address).strip()

        # If no explicit "Address:" or "पता:" label was present (e.g. started by C/O, S/O, D/O):
        # Ensure that the block contains at least one postal/location keyword or PIN code.
        if not has_explicit_address_label:
            has_location_indicator = bool(
                has_seen_pincode
                or re.search(
                    r"\b(?:house|flat|plot|building|floor|block|ward|street|road|lane|nagar|colony|vihar|puram|bazar|bazaar|chowk|haveli|apartments|society|gali|sector|phase|vtc|village|town|city|po|post|sub\s*district|tehsil|taluk|taluka|district|dist|state|pin|pincode|मकान|गली|वार्ड|नगर|कॉलोनी|विहार|बाज़ार|चौक|हवेली|सोसायटी|ग्राम|गाँव|शहर|डाकघर|पोस्ट|तहसील|उप जिला|ज़िला|जिला|राज्य|पिन)\b",
                    combined_address,
                    re.IGNORECASE,
                )
            )
            if not has_location_indicator:
                return None

        if len(combined_address) >= 10:
            return {
                "value": combined_address,
                "confidence": round(default_confidence, 2),
                "status": "FOUND",
            }

    return None


def extract_aadhaar_fields(
    ocr_input: Union[OCRResult, dict[str, Any], str],
    overall_confidence_override: Optional[float] = None,
) -> AadhaarFieldsResult:
    """
    Main entrypoint for Aadhaar OCR field extraction.
    Accepts OCRResult, dictionary, or raw text string and returns structured AadhaarFieldsResult.
    """
    # 1. Normalize OCR Input
    if isinstance(ocr_input, OCRResult):
        raw_text = ocr_input.text
        ocr_conf = ocr_input.confidence
        words = ocr_input.words
    elif isinstance(ocr_input, dict):
        raw_text = str(ocr_input.get("text", ""))
        ocr_conf = float(ocr_input.get("confidence", 0.90))
        words = None
    elif isinstance(ocr_input, str):
        raw_text = ocr_input
        ocr_conf = 0.90
        words = None
    else:
        raw_text = ""
        ocr_conf = 0.0
        words = None

    if overall_confidence_override is not None:
        ocr_conf = overall_confidence_override

    warnings: list[str] = []
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    if not raw_text.strip():
        warnings.append("OCR input text is empty; no fields could be extracted.")
        # Return all NOT_FOUND
        empty_fields = {
            "name": ExtractedField(value=None, confidence=0.0, status="NOT_FOUND"),
            "dob": ExtractedField(value=None, confidence=0.0, status="NOT_FOUND"),
            "year_of_birth": ExtractedField(value=None, confidence=0.0, status="NOT_FOUND"),
            "gender": ExtractedField(value=None, confidence=0.0, status="NOT_FOUND"),
            "aadhaar_number": ExtractedField(value=None, confidence=0.0, status="NOT_FOUND"),
            "address": ExtractedField(value=None, confidence=0.0, status="NOT_FOUND"),
        }
        return AadhaarFieldsResult(success=True, fields=empty_fields, warnings=warnings, raw_text="")

    # 2. Extract Individual Fields
    name_data = extract_name_from_lines(lines, default_confidence=ocr_conf, words_data=words)
    dob_data = extract_dob_from_text(raw_text, default_confidence=ocr_conf)
    gender_data = extract_gender_from_text(raw_text, default_confidence=ocr_conf)
    aadhaar_data = extract_aadhaar_number_from_text(raw_text, default_confidence=ocr_conf)
    address_data = extract_address_block(raw_text, default_confidence=ocr_conf * 0.85)

    fields_dict: dict[str, ExtractedField] = {}

    # Field: Name
    if name_data:
        fields_dict["name"] = ExtractedField(
            value=name_data["value"],
            raw_value=name_data.get("english") or name_data.get("hindi"),
            confidence=name_data["confidence"],
            status=name_data["status"],
            metadata={
                "english": name_data.get("english"),
                "hindi": name_data.get("hindi"),
            },
        )
    else:
        fields_dict["name"] = ExtractedField(value=None, confidence=0.0, status="NOT_FOUND")

    # Field: DOB & Year of Birth
    if dob_data:
        fields_dict["dob"] = ExtractedField(
            value=dob_data["value"],
            raw_value=dob_data.get("raw_value"),
            confidence=dob_data["confidence"],
            status=dob_data["status"],
            metadata={
                "year": dob_data.get("year"),
                "month": dob_data.get("month"),
                "day": dob_data.get("day"),
                "precision": dob_data.get("precision"),
            },
        )
        fields_dict["year_of_birth"] = ExtractedField(
            value=dob_data["year_of_birth"],
            raw_value=dob_data.get("raw_value"),
            confidence=dob_data["confidence"],
            status=dob_data["status"],
        )
    else:
        fields_dict["dob"] = ExtractedField(value=None, confidence=0.0, status="NOT_FOUND")
        fields_dict["year_of_birth"] = ExtractedField(value=None, confidence=0.0, status="NOT_FOUND")

    # Field: Gender
    if gender_data:
        fields_dict["gender"] = ExtractedField(
            value=gender_data["value"],
            raw_value=gender_data.get("raw_value"),
            confidence=gender_data["confidence"],
            status=gender_data["status"],
        )
    else:
        fields_dict["gender"] = ExtractedField(value=None, confidence=0.0, status="NOT_FOUND")

    # Field: Aadhaar Number
    if aadhaar_data:
        fields_dict["aadhaar_number"] = ExtractedField(
            value=aadhaar_data["value"],
            raw_value=aadhaar_data.get("raw_value"),
            confidence=aadhaar_data["confidence"],
            status=aadhaar_data["status"],
            metadata={
                "masked_value": aadhaar_data["masked_value"],
                "format": aadhaar_data.get("format"),
            },
        )
    else:
        fields_dict["aadhaar_number"] = ExtractedField(value=None, confidence=0.0, status="NOT_FOUND")

    # Field: Address
    if address_data:
        fields_dict["address"] = ExtractedField(
            value=address_data["value"],
            confidence=address_data["confidence"],
            status=address_data["status"],
        )
    else:
        fields_dict["address"] = ExtractedField(value=None, confidence=0.0, status="NOT_FOUND")

    # Safe log without sensitive PII
    masked_uid = fields_dict["aadhaar_number"].metadata.get("masked_value") if fields_dict["aadhaar_number"].value else "None"
    logger.info(
        "Aadhaar field extraction completed: name=%s, dob=%s, gender=%s, aadhaar=%s",
        "FOUND" if fields_dict["name"].value else "NOT_FOUND",
        "FOUND" if fields_dict["dob"].value or fields_dict["year_of_birth"].value else "NOT_FOUND",
        fields_dict["gender"].value or "NOT_FOUND",
        masked_uid,
    )

    return AadhaarFieldsResult(
        success=True,
        fields=fields_dict,
        warnings=warnings,
        raw_text=raw_text,
    )


def extract_all_fields_from_ocr(
    raw_text: str,
    overall_ocr_confidence: float = 0.90,
    word_tokens: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Backwards-compatible wrapper matching the legacy dictionary schema.
    Used by existing processor, matcher, and test suites.
    """
    result = extract_aadhaar_fields(raw_text, overall_confidence_override=overall_ocr_confidence)
    fields = result.fields

    extracted: dict[str, Any] = {}

    # Name
    if fields["name"].value:
        extracted["name"] = {
            "value": fields["name"].value,
            "confidence": fields["name"].confidence,
        }
    else:
        extracted["name"] = None

    # DOB
    if fields["dob"].metadata.get("year"):
        extracted["dob"] = {
            "year": fields["dob"].metadata["year"],
            "month": fields["dob"].metadata.get("month"),
            "day": fields["dob"].metadata.get("day"),
            "precision": fields["dob"].metadata.get("precision", "year"),
            "confidence": fields["dob"].confidence,
        }
    else:
        extracted["dob"] = None

    # Gender
    if fields["gender"].value:
        extracted["gender"] = {
            "value": fields["gender"].value,
            "confidence": fields["gender"].confidence,
        }
    else:
        extracted["gender"] = None

    # Aadhaar Number
    if fields["aadhaar_number"].value:
        extracted["aadhaar_number"] = {
            "value": fields["aadhaar_number"].raw_value or fields["aadhaar_number"].value,
            "confidence": fields["aadhaar_number"].confidence,
        }
    else:
        extracted["aadhaar_number"] = None

    # Address
    if fields["address"].value:
        extracted["address"] = {
            "value": fields["address"].value,
            "confidence": fields["address"].confidence,
        }

    return extracted
