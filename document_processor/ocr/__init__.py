"""
OCR Subpackage for bilingual text recognition, field extraction, and normalization.
"""

from document_processor.ocr.engine import (
    OCREngineError,
    OCRResult,
    OCRWord,
    check_ocr_health,
    get_available_tesseract_languages,
    is_tesseract_available,
    run_ocr,
)
from document_processor.ocr.fields import (
    AadhaarFieldsResult,
    ExtractedField,
    extract_aadhaar_fields,
    extract_aadhaar_number_from_text,
    extract_address_block,
    extract_all_fields_from_ocr,
    extract_dob_from_text,
    extract_gender_from_text,
    extract_name_from_lines,
)
from document_processor.ocr.normalization import (
    format_dob_for_display,
    normalize_dob,
    normalize_gender,
    normalize_name,
)

__all__ = [
    "run_ocr",
    "OCRResult",
    "OCRWord",
    "OCREngineError",
    "is_tesseract_available",
    "get_available_tesseract_languages",
    "check_ocr_health",
    "extract_aadhaar_fields",
    "ExtractedField",
    "AadhaarFieldsResult",
    "extract_all_fields_from_ocr",
    "extract_name_from_lines",
    "extract_dob_from_text",
    "extract_gender_from_text",
    "extract_aadhaar_number_from_text",
    "extract_address_block",
    "normalize_name",
    "normalize_gender",
    "normalize_dob",
    "format_dob_for_display",
]
