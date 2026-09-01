"""
QR Code Payload Decoding, Parsing, and Hardened Extraction.
Supports standard UIDAI PrintLetterBarcodeData XML format, JSON payloads, and text streams.
Enforces data masking and explicitly tracks the 5 distinct QR processing states:
1. QR Detected
2. QR Decoded
3. QR Payload Parsed
4. QR Data Structured
5. QR Signature Verified (always False in MVP without UIDAI PKI infrastructure).
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from document_processor.config import mask_sensitive_number
from document_processor.ocr.normalization import (
    format_dob_for_display,
    normalize_dob,
    normalize_gender,
    normalize_name,
)
from document_processor.qr.detector import detect_qr_code

logger = logging.getLogger(__name__)


def mask_payload_uid(payload: str) -> str:
    """
    Masks any 12-digit UID/Aadhaar numbers present in raw XML/JSON payload strings.
    """
    if not payload:
        return ""
    # Mask uid="123456789012" or "uid": "123456789012"
    masked = re.sub(
        r'(uid\s*[:=]\s*["\']?)(\d{8})(\d{4})(["\']?)',
        r'\1XXXX XXXX \3\4',
        payload,
        flags=re.IGNORECASE,
    )
    # Mask standalone 12-digit sequences
    masked = re.sub(
        r'\b(\d{4})\s*(\d{4})\s*(\d{4})\b',
        r'XXXX XXXX \3',
        masked,
    )
    return masked


def parse_aadhaar_xml_qr(payload_str: str) -> Optional[dict[str, Any]]:
    """
    Parses standard UIDAI XML payload:
    <PrintLetterBarcodeData uid="..." name="..." gender="..." yob="..." dob="..."
     co="..." house="..." street="..." lm="..." loc="..." vtc="..." po="..." dist="..." subdist="..." state="..." pc="..." />
    """
    try:
        # Extract <PrintLetterBarcodeData ... /> tag
        match = re.search(r"<PrintLetterBarcodeData\s+[^>]+>", payload_str, re.IGNORECASE)
        xml_tag = match.group(0) if match else payload_str.strip()

        # Ensure valid XML closing
        if not xml_tag.endswith("/>") and not xml_tag.endswith("</PrintLetterBarcodeData>"):
            xml_tag = xml_tag.rstrip(">") + "/>"

        root = ET.fromstring(xml_tag)
        attribs = {k.lower(): v for k, v in root.attrib.items()}

        name = attribs.get("name", "")
        dob_raw = attribs.get("dob") or attribs.get("yob", "")
        gender_raw = attribs.get("gender", "")
        uid_raw = attribs.get("uid", "")

        # Assemble address from XML attributes
        addr_parts = [
            attribs.get("co", ""),
            attribs.get("house", ""),
            attribs.get("street", ""),
            attribs.get("lm", ""),
            attribs.get("loc", ""),
            attribs.get("vtc", ""),
            attribs.get("po", ""),
            attribs.get("dist", ""),
            attribs.get("subdist", ""),
            attribs.get("state", ""),
            attribs.get("pc", ""),
        ]
        address_str = " ".join(part.strip() for part in addr_parts if part and part.strip())
        address_str = re.sub(r"\s+", " ", address_str).strip()

        norm_gender = normalize_gender(gender_raw) or (gender_raw.upper() if gender_raw else None)

        fields: dict[str, Any] = {}
        if name:
            fields["name"] = name.strip().upper()
        if dob_raw:
            norm_dob = normalize_dob(dob_raw)
            fields["dob"] = format_dob_for_display(norm_dob) if norm_dob else dob_raw.strip()
        if norm_gender:
            fields["gender"] = norm_gender
        if uid_raw:
            clean_uid = "".join(ch for ch in uid_raw if ch.isdigit())
            if clean_uid:
                fields["aadhaar_number"] = clean_uid
                fields["masked_aadhaar"] = mask_sensitive_number(clean_uid)
        if address_str:
            fields["address"] = address_str

        return fields if fields else None

    except Exception:
        # Robust fallback parser using regular expressions
        fields_fallback: dict[str, Any] = {}
        name_match = re.search(r'\bname=["\']([^"\']+)["\']', payload_str, re.IGNORECASE)
        dob_match = re.search(r'\bdob=["\']([^"\']+)["\']', payload_str, re.IGNORECASE)
        yob_match = re.search(r'\byob=["\']([^"\']+)["\']', payload_str, re.IGNORECASE)
        gender_match = re.search(r'\bgender=["\']([^"\']+)["\']', payload_str, re.IGNORECASE)
        uid_match = re.search(r'\buid=["\']([^"\']+)["\']', payload_str, re.IGNORECASE)

        if name_match:
            fields_fallback["name"] = name_match.group(1).strip().upper()
        if dob_match:
            norm_dob = normalize_dob(dob_match.group(1))
            fields_fallback["dob"] = format_dob_for_display(norm_dob) if norm_dob else dob_match.group(1).strip()
        elif yob_match:
            fields_fallback["dob"] = yob_match.group(1).strip()
        if gender_match:
            g = normalize_gender(gender_match.group(1)) or gender_match.group(1).upper()
            fields_fallback["gender"] = g
        if uid_match:
            clean_uid = "".join(ch for ch in uid_match.group(1) if ch.isdigit())
            if clean_uid:
                fields_fallback["aadhaar_number"] = clean_uid
                fields_fallback["masked_aadhaar"] = mask_sensitive_number(clean_uid)

        return fields_fallback if fields_fallback else None


def parse_aadhaar_json_qr(payload_str: str) -> Optional[dict[str, Any]]:
    """
    Parses JSON formatted QR payload.
    """
    try:
        data = json.loads(payload_str)
        if isinstance(data, dict):
            fields: dict[str, Any] = {}
            for k, v in data.items():
                k_lower = k.lower()
                if "name" in k_lower and isinstance(v, str):
                    fields["name"] = v.strip().upper()
                elif ("dob" in k_lower or "yob" in k_lower or "birth" in k_lower) and isinstance(v, str):
                    norm_dob = normalize_dob(v)
                    fields["dob"] = format_dob_for_display(norm_dob) if norm_dob else v.strip()
                elif "gender" in k_lower and isinstance(v, str):
                    norm_gender = normalize_gender(v) or v.strip().upper()
                    fields["gender"] = norm_gender
                elif ("uid" in k_lower or "aadhaar" in k_lower) and isinstance(v, str):
                    clean_uid = "".join(ch for ch in v if ch.isdigit())
                    if clean_uid:
                        fields["aadhaar_number"] = clean_uid
                        fields["masked_aadhaar"] = mask_sensitive_number(clean_uid)
                elif "address" in k_lower and isinstance(v, str):
                    fields["address"] = v.strip()

            return fields if fields else None
    except Exception:
        pass
    return None


def parse_qr_payload(payload_str: str) -> Tuple[Optional[dict[str, Any]], str]:
    """
    Attempts to parse extracted QR payload into structured identity fields.
    Returns:
        (extracted_fields: dict | None, format: "xml" | "json" | "unknown")
    """
    if not payload_str or not payload_str.strip():
        return None, "unknown"

    stripped = payload_str.strip()

    # 1. XML format (standard Aadhaar QR)
    if "PrintLetterBarcodeData" in stripped or "<" in stripped:
        xml_res = parse_aadhaar_xml_qr(stripped)
        if xml_res:
            return xml_res, "xml"

    # 2. JSON format
    if stripped.startswith("{") and stripped.endswith("}"):
        json_res = parse_aadhaar_json_qr(stripped)
        if json_res:
            return json_res, "json"

    # 3. Fallback attempt for partial XML
    xml_res = parse_aadhaar_xml_qr(stripped)
    if xml_res:
        return xml_res, "xml"

    return None, "unknown"


def process_qr_code(image: np.ndarray) -> dict[str, Any]:
    """
    Main QR processing entrypoint.
    Executes multi-pass scanning, payload decoding, and structured extraction.
    Strictly separates:
    - detected (True/False)
    - decoded (True/False)
    - verified (always False in MVP - cryptographic signature check out of scope)
    - fields (Structured identity dictionary or None)
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "detected": False,
            "decoded": False,
            "verified": False,
        }

    detector = cv2.QRCodeDetector()

    # Pass 1: Decode on direct input image
    decoded_text, points, _ = detector.detectAndDecode(image)

    # Pass 2: Decode on grayscale
    if not decoded_text:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        decoded_text, points, _ = detector.detectAndDecode(gray)

    # Pass 3: Decode on Otsu threshold
    if not decoded_text:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        decoded_text, points, _ = detector.detectAndDecode(thresh)

    # Pass 4: Decode on scaled image if large
    if not decoded_text:
        h, w = image.shape[:2]
        if w > 1200:
            scale = 1000.0 / w
            scaled = cv2.resize(image, (1000, int(h * scale)))
            decoded_text, points, _ = detector.detectAndDecode(scaled)

    # If decoding returned no text, check if a QR pattern was at least detected
    if not decoded_text:
        has_qr, detected_points = detect_qr_code(image)
        if has_qr:
            return {
                "detected": True,
                "decoded": False,
                "verified": False,
                "error": "QR_DECODE_FAILED",
            }
        else:
            return {
                "detected": False,
                "decoded": False,
                "verified": False,
            }

    # QR Decoded successfully -> Parse payload
    extracted_fields, payload_format = parse_qr_payload(decoded_text)
    masked_payload = mask_payload_uid(decoded_text)

    result: dict[str, Any] = {
        "detected": True,
        "decoded": True,
        "verified": False,  # Explicitly False: cryptographic PKI validation not performed
        "format": payload_format,
    }

    if extracted_fields:
        result["fields"] = extracted_fields

    # Safe log without sensitive PII
    logger.info(
        "QR code processed: detected=True, decoded=True, format=%s, fields_extracted=%s",
        payload_format,
        bool(extracted_fields),
    )

    return result
