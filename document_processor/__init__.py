"""
Document Processing & OCR Module for Identity Verification System.
"""

from document_processor.config import (
    NAME_SIMILARITY_THRESHOLD,
    TESSERACT_CMD,
    find_tesseract_cmd,
)
from document_processor.file_handler import (
    FileValidationError,
    LoadedDocument,
    cleanup_temp_files,
    load_document,
    validate_file,
    validate_file_input,
)
from document_processor.preprocessing import (
    PreprocessedDocument,
    PreprocessingError,
    PreprocessingVariant,
    get_ocr_variants,
    get_qr_variants,
    preprocess_image,
    save_debug_variants,
)
from document_processor.document_detector import (
    DocumentDetectionResult,
    detect_aadhaar,
    detect_aadhaar_document,
    detect_document_type,
)
from document_processor.ocr import (
    AadhaarFieldsResult,
    ExtractedField,
    OCREngineError,
    OCRResult,
    OCRWord,
    check_ocr_health,
    extract_aadhaar_fields,
    get_available_tesseract_languages,
    is_tesseract_available,
    run_ocr,
)
from document_processor.qr import (
    detect_qr_code,
    mask_payload_uid,
    parse_aadhaar_json_qr,
    parse_aadhaar_xml_qr,
    parse_qr_payload,
    process_qr_code,
)
from document_processor.processor import process_document

__all__ = [
    "process_document",
    "load_document",
    "validate_file",
    "validate_file_input",
    "LoadedDocument",
    "FileValidationError",
    "detect_document_type",
    "detect_aadhaar",
    "detect_aadhaar_document",
    "DocumentDetectionResult",
    "preprocess_image",
    "get_ocr_variants",
    "get_qr_variants",
    "PreprocessedDocument",
    "PreprocessingVariant",
    "PreprocessingError",
    "save_debug_variants",
    "run_ocr",
    "OCRResult",
    "OCRWord",
    "OCREngineError",
    "check_ocr_health",
    "is_tesseract_available",
    "get_available_tesseract_languages",
    "extract_aadhaar_fields",
    "ExtractedField",
    "AadhaarFieldsResult",
    "process_qr_code",
    "detect_qr_code",
    "parse_qr_payload",
    "parse_aadhaar_xml_qr",
    "parse_aadhaar_json_qr",
    "mask_payload_uid",
    "cleanup_temp_files",
    "TESSERACT_CMD",
    "NAME_SIMILARITY_THRESHOLD",
    "find_tesseract_cmd",
]
