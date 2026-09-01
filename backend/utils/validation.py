"""
Validation utilities for uploaded document and image files.
"""

from typing import Tuple

# Maximum allowed upload size (15 MB)
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024

# Supported MIME types and extensions
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
ALL_SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS

# Magic byte signatures for secure file type verification
MAGIC_SIGNATURES = {
    "pdf": [b"%PDF"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "jpeg": [b"\xff\xd8\xff"],
    "webp": [b"RIFF"],  # also checks WEBP at offset 8
    "bmp": [b"BM"],
    "tiff_le": [b"II*\x00"],
    "tiff_be": [b"MM\x00*"],
}


def get_file_extension(filename: str) -> str:
    """Extract lowercase file extension."""
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def detect_file_type(header: bytes) -> str | None:
    """
    Identify file format from leading magic bytes.
    Returns 'pdf', 'image', or None if unknown.
    """
    if len(header) < 4:
        return None

    if header.startswith(b"%PDF"):
        return "pdf"

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"

    if header.startswith(b"\xff\xd8\xff"):
        return "image"

    if header.startswith(b"BM"):
        return "image"

    if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
        return "image"

    if len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image"

    return None


def validate_uploaded_file(filename: str | None, file_bytes: bytes) -> Tuple[bool, str, str | None]:
    """
    Validate file presence, size, extension, and magic bytes.

    Returns:
        (is_valid: bool, file_type: str, error_message: str | None)
    """
    if not file_bytes or len(file_bytes) == 0:
        return False, "", "The uploaded file is empty."

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        return (
            False,
            "",
            f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    # Extension check
    ext = get_file_extension(filename or "")
    if ext and ext not in ALL_SUPPORTED_EXTENSIONS:
        return (
            False,
            "",
            f"Unsupported file extension '{ext}'. Supported formats: {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}.",
        )

    # Magic byte verification
    header = file_bytes[:16]
    detected_type = detect_file_type(header)

    if not detected_type:
        # If magic bytes didn't match known signatures, fallback to extension if it's standard image/pdf
        if ext in SUPPORTED_PDF_EXTENSIONS:
            detected_type = "pdf"
        elif ext in SUPPORTED_IMAGE_EXTENSIONS:
            detected_type = "image"
        else:
            return (
                False,
                "",
                "Unrecognized or corrupted file format. Please upload a valid Image or PDF document.",
            )

    return True, detected_type, None
