"""
Main Document Processor pipeline orchestrator.
Provides the clean public entry point: process_document(file_input).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from document_processor.config import mask_sensitive_number
from document_processor.document_detector import detect_aadhaar_document
from document_processor.file_handler import (
    FileValidationError,
    cleanup_temp_files,
    load_document,
)
from document_processor.ocr.engine import OCREngineError, is_tesseract_available, run_ocr
from document_processor.ocr.fields import extract_all_fields_from_ocr
from document_processor.preprocessing import preprocess_image
from document_processor.qr.decoder import process_qr_code
from document_processor.verification.matcher import cross_validate_ocr_and_qr

logger = logging.getLogger(__name__)


def process_document(
    file_input: Union[str, Path, bytes, io.BytesIO]
) -> dict[str, Any]:
    """
    Main entry point for document verification.
    Accepts an image or PDF input, runs preprocessing, bilingual OCR, QR code decoding,
    field extraction, and OCR ↔ QR cross-validation.

    Returns a JSON-serializable structured dictionary adhering to PRD Section 27.
    """
    warnings: list[str] = []

    # Step 1: File validation & loading
    load_res = load_document(file_input)
    if not load_res.get("success", False):
        return load_res

    doc_image = load_res["images"][0]

    # Step 2: QR Code Processing (runs on original / high-contrast variants)
    try:
        qr_result = process_qr_code(doc_image)
    except Exception as qr_exc:
        warnings.append(f"QR processing encountered an issue: {str(qr_exc)}")
        qr_result = {
            "detected": False,
            "decoded": False,
            "verified": False,
        }

    qr_detected = qr_result.get("detected", False)
    qr_decoded = qr_result.get("decoded", False)
    qr_fields = qr_result.get("fields")
    qr_is_aadhaar = bool(qr_fields and ("name" in qr_fields or "dob" in qr_fields or "gender" in qr_fields))

    # Step 3: Image Preprocessing for OCR
    try:
        preprocessed = preprocess_image(doc_image)
        ocr_input = preprocessed
    except Exception as prep_exc:
        warnings.append(f"Image preprocessing fallback used: {str(prep_exc)}")
        ocr_input = doc_image

    # Step 4: Bilingual OCR
    ocr_text = ""
    ocr_confidence = 0.0
    ocr_tokens: list[dict[str, Any]] = []
    ocr_lang = "eng+hin"

    try:
        ocr_response = run_ocr(ocr_input, languages="eng+hin")
        ocr_text = ocr_response.get("text", "")
        ocr_confidence = ocr_response.get("confidence", 0.0)
        ocr_tokens = ocr_response.get("tokens", [])
        ocr_lang = ocr_response.get("language", "eng+hin")
        if ocr_response.get("warnings"):
            warnings.extend(ocr_response["warnings"])
    except OCREngineError as ocr_err:
        # If Tesseract engine is unavailable or failed
        if ocr_err.code == "OCR_ENGINE_UNAVAILABLE":
            return {
                "success": False,
                "error": ocr_err.to_dict(),
            }
        else:
            warnings.append(f"OCR failure: {ocr_err.message}")
    except Exception as exc:
        warnings.append(f"Unexpected OCR exception: {str(exc)}")

    # Step 5: Document Type Detection & Confidence Scoring
    doc_detection = detect_aadhaar_document(
        raw_ocr_text=ocr_text,
        qr_detected=qr_detected,
        qr_is_aadhaar=qr_is_aadhaar,
    )

    # Step 6: Field Extraction & Normalization
    extracted_fields = extract_all_fields_from_ocr(
        raw_text=ocr_text,
        overall_ocr_confidence=ocr_confidence,
        word_tokens=ocr_tokens,
    )

    # Format OCR output structure
    ocr_output_fields: dict[str, Any] = {
        "name": extracted_fields.get("name"),
        "dob": extracted_fields.get("dob"),
        "gender": extracted_fields.get("gender"),
    }

    if extracted_fields.get("aadhaar_number"):
        ocr_output_fields["aadhaar_number"] = {
            "value": mask_sensitive_number(extracted_fields["aadhaar_number"]["value"]),
            "confidence": extracted_fields["aadhaar_number"]["confidence"],
        }
    else:
        ocr_output_fields["aadhaar_number"] = None

    ocr_output_fields["address"] = extracted_fields.get("address")

    # Step 7: OCR ↔ QR Cross-Validation
    cross_val_result = cross_validate_ocr_and_qr(
        ocr_fields=ocr_output_fields,
        qr_fields=qr_fields,
    )

    # Safe mask for QR aadhaar number in public output
    public_qr = dict(qr_result)
    if public_qr.get("fields") and "aadhaar_number" in public_qr["fields"]:
        public_qr["fields"] = dict(public_qr["fields"])
        public_qr["fields"]["aadhaar_number"] = mask_sensitive_number(public_qr["fields"]["aadhaar_number"])

    # Assemble full JSON output strictly following PRD Section 27
    result: dict[str, Any] = {
        "success": True,
        "document": {
            "type": doc_detection["type"],
            "confidence": doc_detection["confidence"],
        },
        "ocr": {
            "language": ocr_lang,
            "confidence": ocr_confidence,
            "fields": ocr_output_fields,
        },
        "qr": public_qr,
    }

    if cross_val_result:
        result["cross_validation"] = cross_val_result

    result["warnings"] = warnings

    # Safe log without sensitive PII
    logger.info(
        "Processed document: type=%s (conf=%.2f), ocr_conf=%.2f, qr_detected=%s, qr_decoded=%s",
        doc_detection["type"],
        doc_detection["confidence"],
        ocr_confidence,
        qr_detected,
        qr_decoded,
    )

    return result
