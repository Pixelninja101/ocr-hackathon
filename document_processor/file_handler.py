"""
File validation, safe document loading, and PDF page rendering layer.
Implements robust format validation, magic-byte inspection, size limits, and structured error handling.
"""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import pymupdf as fitz

from document_processor.config import (
    MAX_FILE_SIZE_BYTES,
    PDF_RENDER_DPI,
    SUPPORTED_EXTENSIONS,
)


# Standard magic bytes signatures
MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    "pdf": [b"%PDF-"],
    "jpeg": [b"\xff\xd8\xff"],
    "png": [b"\x89PNG\r\n\x1a\n"],
}


class FileValidationError(Exception):
    """
    Raised when file validation or document loading fails.
    Carries safe, sanitized error information suitable for backend exposure.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        """Returns safe, structured error dictionary without internal paths or traces."""
        return {
            "success": False,
            "error": {
                "code": self.code,
                "message": self.message,
            },
        }


@dataclass
class LoadedDocument:
    """
    Structured internal representation of a successfully validated and loaded document.
    """
    success: bool = True
    file_type: str = "image"  # "image" | "pdf"
    format: str = "jpeg"      # "jpeg" | "png" | "pdf"
    pages: int = 1            # Number of rendered pages included in `images`
    total_pages: int = 1      # Total pages in source document
    images: list[np.ndarray] = field(default_factory=list)  # OpenCV BGR ndarrays
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_image(self) -> np.ndarray:
        """Returns the primary (first) page image for single-page pipeline processing."""
        if not self.images:
            raise ValueError("Document contains no rendered images.")
        return self.images[0]

    def to_dict(self, include_images: bool = True) -> dict[str, Any]:
        """
        Converts the loaded document to a dictionary.
        Set include_images=False for JSON-serializable logging or debugging.
        """
        res: dict[str, Any] = {
            "success": self.success,
            "file_type": self.file_type,
            "format": self.format,
            "pages": self.pages,
            "total_pages": self.total_pages,
            "metadata": self.metadata,
        }
        if include_images:
            res["images"] = self.images
        return res


def _inspect_magic_bytes(data: bytes) -> Optional[str]:
    """
    Inspects leading bytes to verify actual content type.
    Never trusts file extensions alone.
    """
    if len(data) < 4:
        return None

    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    # Secondary check with PIL header parsing
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = (img.format or "").lower()
            if fmt in ("jpeg", "jpg"):
                return "jpeg"
            if fmt == "png":
                return "png"
    except Exception:
        pass

    return None


def validate_file_input(
    file_input: Union[str, Path, bytes, io.BytesIO],
    max_size_bytes: Optional[int] = None,
) -> Tuple[bytes, str, str, int]:
    """
    Validates file existence, type, size constraints, and magic bytes.

    Returns:
        (file_bytes, detected_format, file_type, file_size_bytes)
        where detected_format is 'jpeg' | 'png' | 'pdf'
        and file_type is 'image' | 'pdf'

    Raises:
        FileValidationError with specific error code on invalid inputs.
    """
    size_limit = max_size_bytes if max_size_bytes is not None else MAX_FILE_SIZE_BYTES
    file_bytes: bytes
    file_size: int

    # 1. Path-based input
    if isinstance(file_input, (str, Path)):
        path = Path(file_input)
        if not path.exists():
            raise FileValidationError(
                "FILE_NOT_FOUND",
                f"The specified file '{path.name}' does not exist.",
            )
        if not path.is_file():
            raise FileValidationError(
                "INVALID_FILE",
                f"The specified path '{path.name}' is not a regular file.",
            )

        try:
            file_size = path.stat().st_size
        except OSError:
            raise FileValidationError(
                "FILE_NOT_FOUND",
                f"Unable to access file '{path.name}'.",
            )

        if file_size == 0:
            raise FileValidationError(
                "EMPTY_FILE",
                f"The file '{path.name}' is empty (0 bytes).",
            )

        if file_size > size_limit:
            raise FileValidationError(
                "FILE_TOO_LARGE",
                f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed size ({size_limit / (1024 * 1024):.1f} MB).",
            )

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise FileValidationError(
                "UNSUPPORTED_FILE_TYPE",
                f"Unsupported file extension '{suffix}'. Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}.",
            )

        try:
            file_bytes = path.read_bytes()
        except Exception:
            raise FileValidationError(
                "IMAGE_LOAD_ERROR",
                f"Failed to read file '{path.name}'.",
            )

    # 2. Raw bytes input
    elif isinstance(file_input, bytes):
        file_bytes = file_input
        file_size = len(file_bytes)
        if file_size == 0:
            raise FileValidationError("EMPTY_FILE", "The uploaded byte content is empty (0 bytes).")
        if file_size > size_limit:
            raise FileValidationError(
                "FILE_TOO_LARGE",
                f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed size ({size_limit / (1024 * 1024):.1f} MB).",
            )

    # 3. In-memory buffer input (BytesIO)
    elif isinstance(file_input, io.BytesIO):
        file_bytes = file_input.getvalue()
        file_size = len(file_bytes)
        if file_size == 0:
            raise FileValidationError("EMPTY_FILE", "The uploaded buffer is empty (0 bytes).")
        if file_size > size_limit:
            raise FileValidationError(
                "FILE_TOO_LARGE",
                f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed size ({size_limit / (1024 * 1024):.1f} MB).",
            )
    else:
        raise FileValidationError(
            "INVALID_INPUT",
            f"Unsupported input type '{type(file_input).__name__}'. Expected str, Path, bytes, or BytesIO.",
        )

    # 4. Content / Magic-byte verification (don't trust extensions)
    detected_format = _inspect_magic_bytes(file_bytes)
    if not detected_format:
        raise FileValidationError(
            "CORRUPTED_OR_INVALID_FILE",
            "The file contents do not match any supported image (JPEG/PNG) or PDF format.",
        )

    file_type = "pdf" if detected_format == "pdf" else "image"
    return file_bytes, detected_format, file_type, file_size


def validate_file(
    file_input: Union[str, Path, bytes, io.BytesIO],
    max_size_bytes: Optional[int] = None,
) -> dict[str, Any]:
    """
    Public validation function.
    Returns a structured dictionary indicating whether the file is valid.

    Example successful result:
        {"success": True, "valid": True, "file_type": "image", "format": "jpeg", "file_size_bytes": 45000}

    Example error result:
        {"success": False, "valid": False, "error": {"code": "FILE_TOO_LARGE", "message": "..."}}
    """
    try:
        _, detected_format, file_type, file_size = validate_file_input(
            file_input, max_size_bytes=max_size_bytes
        )
        return {
            "success": True,
            "valid": True,
            "file_type": file_type,
            "format": detected_format,
            "file_size_bytes": file_size,
        }
    except FileValidationError as val_err:
        return {
            "success": False,
            "valid": False,
            "error": {
                "code": val_err.code,
                "message": val_err.message,
            },
        }
    except Exception as exc:
        return {
            "success": False,
            "valid": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": f"Unexpected validation failure: {str(exc)}",
            },
        }


def _load_image_safely(data: bytes, detected_format: str, file_size: int) -> LoadedDocument:
    """
    Decodes image bytes safely using OpenCV and Pillow fallbacks, verifying decoding succeeded.
    """
    np_arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Secondary fallback to Pillow if imdecode returns None
    if image is None:
        try:
            with Image.open(io.BytesIO(data)) as pil_img:
                rgb_img = pil_img.convert("RGB")
                image = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        except Exception:
            raise FileValidationError(
                "INVALID_IMAGE",
                "The image file is corrupted or cannot be decoded into a valid visual representation.",
            )

    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise FileValidationError(
            "INVALID_IMAGE",
            "Decoded image contains zero dimensions or empty pixel data.",
        )

    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    metadata = {
        "width": int(w),
        "height": int(h),
        "channels": int(channels),
        "file_size_bytes": int(file_size),
        "dpi": None,
    }

    return LoadedDocument(
        success=True,
        file_type="image",
        format=detected_format,
        pages=1,
        total_pages=1,
        images=[image],
        metadata=metadata,
    )


def _load_pdf_safely(
    data: bytes,
    file_size: int,
    max_pages: int = 1,
    dpi: int = PDF_RENDER_DPI,
) -> LoadedDocument:
    """
    Safely opens and renders PDF document pages into OpenCV BGR numpy arrays using PyMuPDF.
    """
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise FileValidationError(
            "INVALID_PDF",
            f"The PDF file is corrupted and could not be opened: {str(exc)}",
        )

    try:
        if doc.is_encrypted:
            raise FileValidationError(
                "PDF_UNSUPPORTED",
                "The PDF document is encrypted or password-protected and cannot be processed.",
            )

        total_pages = doc.page_count
        if total_pages == 0:
            raise FileValidationError(
                "INVALID_PDF",
                "The PDF document contains 0 pages.",
            )

        rendered_images: list[np.ndarray] = []
        pages_to_render = min(total_pages, max(1, max_pages))
        zoom = dpi / 72.0  # Standard PDF point = 72 dpi
        matrix = fitz.Matrix(zoom, zoom)

        for page_idx in range(pages_to_render):
            try:
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_data = pix.samples
                # PyMuPDF samples buffer is RGB -> reshape and convert to OpenCV BGR
                img_np = np.frombuffer(img_data, dtype=np.uint8).reshape((pix.height, pix.width, 3))
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                rendered_images.append(img_bgr)
            except Exception as page_exc:
                raise FileValidationError(
                    "PDF_READ_ERROR",
                    f"Failed to render page {page_idx + 1} of PDF: {str(page_exc)}",
                )

        if not rendered_images:
            raise FileValidationError(
                "PDF_READ_ERROR",
                "No visual pages could be rendered from the PDF.",
            )

        first_img = rendered_images[0]
        h, w = first_img.shape[:2]
        channels = first_img.shape[2] if len(first_img.shape) == 3 else 1

        metadata = {
            "width": int(w),
            "height": int(h),
            "channels": int(channels),
            "file_size_bytes": int(file_size),
            "dpi": int(dpi),
        }

        return LoadedDocument(
            success=True,
            file_type="pdf",
            format="pdf",
            pages=len(rendered_images),
            total_pages=total_pages,
            images=rendered_images,
            metadata=metadata,
        )
    finally:
        try:
            doc.close()
        except Exception:
            pass


def load_document(
    file_input: Union[str, Path, bytes, io.BytesIO],
    max_pages: int = 1,
    dpi: int = PDF_RENDER_DPI,
    max_size_bytes: Optional[int] = None,
) -> dict[str, Any]:
    """
    Main public document loading function.
    Accepts any supported image (JPG, JPEG, PNG) or PDF input, validates it, and safely
    converts it into a standardized internal representation containing OpenCV BGR numpy arrays.

    Returns:
        Structured dictionary conforming to the internal result representation:
        On success:
            {
                "success": True,
                "file_type": "image" | "pdf",
                "format": "jpeg" | "png" | "pdf",
                "pages": 1,
                "total_pages": 1,
                "images": [numpy.ndarray (BGR)],
                "metadata": {
                    "width": int,
                    "height": int,
                    "channels": int,
                    "file_size_bytes": int,
                    "dpi": int | None
                }
            }
        On failure:
            {
                "success": False,
                "error": {
                    "code": "FILE_NOT_FOUND" | "FILE_TOO_LARGE" | "INVALID_IMAGE" | ...,
                    "message": "..."
                }
            }
    """
    try:
        file_bytes, detected_format, file_type, file_size = validate_file_input(
            file_input, max_size_bytes=max_size_bytes
        )

        if file_type == "pdf":
            loaded_doc = _load_pdf_safely(
                file_bytes, file_size=file_size, max_pages=max_pages, dpi=dpi
            )
        else:
            loaded_doc = _load_image_safely(
                file_bytes, detected_format=detected_format, file_size=file_size
            )

        return loaded_doc.to_dict(include_images=True)

    except FileValidationError as f_err:
        return f_err.to_dict()
    except Exception as exc:
        return {
            "success": False,
            "error": {
                "code": "DOCUMENT_LOAD_FAILED",
                "message": f"Unexpected failure loading document: {str(exc)}",
            },
        }


def cleanup_temp_files(*paths: Union[str, Path, None]) -> None:
    """
    Safely deletes temporary files from controlled locations without throwing exceptions.
    """
    for p in paths:
        if p is None:
            continue
        try:
            path_obj = Path(p)
            if path_obj.exists() and path_obj.is_file():
                path_obj.unlink()
        except Exception:
            pass
