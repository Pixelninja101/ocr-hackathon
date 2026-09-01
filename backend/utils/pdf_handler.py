"""
PDF processing module using PyMuPDF (pymupdf).
Renders PDF document pages to OpenCV BGR images at high resolution (~300 DPI).
"""

from typing import Tuple
import cv2
import numpy as np
import pymupdf


def render_pdf_page_to_cv2_image(
    pdf_bytes: bytes, page_number: int = 0, dpi: int = 300
) -> Tuple[np.ndarray, int, int]:
    """
    Render a specific page from an in-memory PDF byte buffer to an OpenCV BGR numpy array.
    Uses ~300 DPI (where zoom = dpi / 72 ≈ 4.1667) for high fidelity QR code detection.

    Args:
        pdf_bytes: Raw bytes of the PDF file.
        page_number: 0-indexed page number to render.
        dpi: Target rendering DPI (default 300 DPI).

    Returns:
        Tuple of:
            - cv2_image: np.ndarray (OpenCV BGR format)
            - total_pages: int
            - current_page: int (0-indexed)
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError("Empty PDF byte stream received.")

    try:
        # Open PDF from in-memory stream without touching disk
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to open PDF document: {exc}") from exc

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise ValueError("The PDF document contains 0 pages.")

    if page_number < 0 or page_number >= total_pages:
        # Gracefully clamp to page 0 if out of bounds
        page_number = 0

    page = doc[page_number]

    # Render page at exact target DPI (300 DPI gives ~4.167x matrix zoom)
    pix = page.get_pixmap(dpi=dpi, alpha=False)

    # Convert pixmap raw buffer to numpy array
    # pix.n represents channels (1 for gray, 3 for RGB)
    np_arr = np.frombuffer(pix.samples, dtype=np.uint8)

    if pix.n == 1:
        gray_img = np_arr.reshape((pix.height, pix.width))
        cv2_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    elif pix.n == 3:
        rgb_img = np_arr.reshape((pix.height, pix.width, 3))
        cv2_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:
        rgba_img = np_arr.reshape((pix.height, pix.width, 4))
        cv2_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGR)
    else:
        doc.close()
        raise ValueError(f"Unsupported pixmap channel count: {pix.n}")

    doc.close()
    return cv2_img, total_pages, page_number
