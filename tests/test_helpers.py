"""
Synthetic test helpers for generating synthetic Aadhaar-like documents, PDFs, and QR codes in-memory.
Ensures zero real identity documents or PII are used in tests.
"""

from __future__ import annotations

import io
from typing import Optional
import cv2
import pymupdf as fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_synthetic_qr_image(payload: str, size: int = 250) -> np.ndarray:
    """
    Generates a QR code image as a numpy BGR array using OpenCV QRCodeEncoder
    with standard 4-module quiet zone and integer scale factor for reliable detection.
    """
    encoder = cv2.QRCodeEncoder.create()
    qr_mat = encoder.encode(payload)

    # Add standard quiet zone (4 modules)
    border = 4
    qr_bordered = cv2.copyMakeBorder(
        qr_mat, border, border, border, border, cv2.BORDER_CONSTANT, value=255
    )

    # Scale with integer multiplier
    target_dim = max(size, qr_bordered.shape[0] * 4)
    scale = max(1, target_dim // qr_bordered.shape[0])
    scaled_w = qr_bordered.shape[1] * scale
    scaled_h = qr_bordered.shape[0] * scale
    qr_scaled = cv2.resize(qr_bordered, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

    if len(qr_scaled.shape) == 2:
        qr_bgr = cv2.cvtColor(qr_scaled, cv2.COLOR_GRAY2BGR)
    else:
        qr_bgr = qr_scaled
    return qr_bgr



def create_synthetic_aadhaar_image(
    name: str = "RAHUL KUMAR",
    dob: str = "12/04/2002",
    gender: str = "MALE",
    aadhaar_num: str = "9876 5432 1098",
    include_qr: bool = True,
    qr_payload: Optional[str] = None,
    image_width: int = 1000,
    image_height: int = 650,
) -> np.ndarray:
    """
    Creates a synthetic Aadhaar-like card image in-memory using Pillow and OpenCV.
    """
    img = Image.new("RGB", (image_width, image_height), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Card border
    draw.rectangle([(10, 10), (image_width - 10, image_height - 10)], outline=(180, 180, 180), width=2)
    # Header bar
    draw.rectangle([(15, 15), (image_width - 15, 80)], fill=(220, 50, 50))

    # Header text
    draw.text((30, 25), "Government of India / भारत सरकार", fill=(255, 255, 255))
    draw.text((30, 50), "Unique Identification Authority of India", fill=(255, 255, 255))

    # Details
    draw.text((40, 120), "मेरा आधार, मेरी पहचान", fill=(100, 100, 100))
    draw.text((40, 160), "Name / नाम:", fill=(80, 80, 80))
    draw.text((40, 185), name, fill=(10, 10, 10))

    draw.text((40, 230), f"DOB / जन्म तिथि: {dob}", fill=(10, 10, 10))
    draw.text((40, 270), f"Gender / लिंग: {gender}", fill=(10, 10, 10))

    draw.text((40, 420), f"Aadhaar Number: {aadhaar_num}", fill=(20, 20, 20))

    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if include_qr:
        if not qr_payload:
            qr_payload = f'<PrintLetterBarcodeData uid="{aadhaar_num.replace(" ", "")}" name="{name}" dob="{dob}" gender="{gender}" />'
        qr_img = create_synthetic_qr_image(qr_payload, size=180)
        qr_h, qr_w = qr_img.shape[:2]
        # Place QR in right corner
        qr_y, qr_x = 150, image_width - qr_w - 30
        img_np[qr_y : qr_y + qr_h, qr_x : qr_x + qr_w] = qr_img

    return img_np


def create_synthetic_aadhaar_bytes(
    fmt: str = "png", **kwargs
) -> bytes:
    """Encodes a synthetic Aadhaar image into PNG or JPEG bytes."""
    img_bgr = create_synthetic_aadhaar_image(**kwargs)
    ext = f".{fmt}"
    success, buffer = cv2.imencode(ext, img_bgr)
    if not success:
        raise RuntimeError("Failed to encode synthetic image")
    return buffer.tobytes()


def create_synthetic_aadhaar_pdf(
    **kwargs
) -> bytes:
    """Generates a synthetic PDF document containing an Aadhaar page."""
    img_bytes = create_synthetic_aadhaar_bytes(fmt="png", **kwargs)
    doc = fitz.open()
    page = doc.new_page(width=600, height=400)
    page.insert_image(page.rect, stream=img_bytes)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
