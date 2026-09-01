"""
Configuration module for Document Processing & OCR.
Manages paths, thresholds, and environment settings.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

# File constraints
MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".pdf")
SUPPORTED_MIME_TYPES: tuple[str, ...] = (
    "image/jpeg",
    "image/png",
    "application/pdf",
)

# PDF rendering DPI (higher gives better OCR accuracy)
PDF_RENDER_DPI: int = int(os.getenv("PDF_RENDER_DPI", "300"))

# OCR Configuration
DEFAULT_OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "eng+hin")
FALLBACK_OCR_LANGUAGES: str = "eng"
DEFAULT_OCR_PSM: int = int(os.getenv("OCR_PSM", "3"))  # Fully automatic page segmentation without OSD
OCR_TIMEOUT_SECONDS: int = int(os.getenv("OCR_TIMEOUT_SECONDS", "15"))


def find_tesseract_cmd() -> Optional[str]:
    """
    Locate tesseract executable from environment variable, PATH, or standard installation paths.
    """
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and os.path.isfile(env_cmd):
        return env_cmd

    # Check if 'tesseract' is available in system PATH
    which_cmd = shutil.which("tesseract")
    if which_cmd:
        return which_cmd

    # Standard Windows install locations
    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        r"C:\tools\tesseract\tesseract.exe",
    ]
    for p in windows_paths:
        if os.path.isfile(p):
            return p

    return None


TESSERACT_CMD: Optional[str] = find_tesseract_cmd()

# Matching Thresholds
NAME_SIMILARITY_THRESHOLD: float = float(os.getenv("NAME_SIMILARITY_THRESHOLD", "0.85"))
DOCUMENT_CONFIDENCE_THRESHOLD: float = float(os.getenv("DOCUMENT_CONFIDENCE_THRESHOLD", "0.60"))

# Image Resolution and Preprocessing Constraints
MIN_IMAGE_WIDTH: int = int(os.getenv("MIN_IMAGE_WIDTH", "1200"))
MAX_IMAGE_WIDTH: int = int(os.getenv("MAX_IMAGE_WIDTH", "2500"))
MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "8000"))
MAX_IMAGE_INPUT_DIMENSION: int = int(os.getenv("MAX_IMAGE_INPUT_DIMENSION", "12000"))
UPSCALE_FACTOR: float = float(os.getenv("UPSCALE_FACTOR", "1.5"))
ENABLE_DESKEW: bool = os.getenv("ENABLE_DESKEW", "True").lower() in ("true", "1", "yes")
MAX_DESKEW_ANGLE: float = float(os.getenv("MAX_DESKEW_ANGLE", "30.0"))
MAX_PREPROCESSING_VARIANTS: int = int(os.getenv("MAX_PREPROCESSING_VARIANTS", "3"))
DEBUG_SAVE_PREPROCESSED: bool = os.getenv("DEBUG_SAVE_PREPROCESSED", "False").lower() in ("true", "1", "yes")


def mask_sensitive_number(val: Optional[str]) -> str:
    """
    Mask sensitive 12-digit Aadhaar / ID numbers in logs (e.g., 'XXXX XXXX 1234').
    """
    if not val:
        return ""
    clean = "".join(ch for ch in str(val) if ch.isdigit() or ch.upper() == "X")
    if len(clean) >= 12:
        return f"XXXX XXXX {clean[-4:]}"
    elif len(clean) >= 4:
        return f"XXXX {clean[-4:]}"
    return "XXXX"
