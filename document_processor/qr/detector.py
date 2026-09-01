"""
QR Code Pattern Detection using OpenCV.
Determines whether a visual QR pattern exists on the document.
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np


def detect_qr_code(image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Scans the image across multiple preprocessing scales/transforms to detect QR code position.
    Returns:
        (detected: bool, points: Optional[np.ndarray])
    """
    detector = cv2.QRCodeDetector()

    # Pass 1: Direct scan
    retval, points = detector.detect(image)
    if retval and points is not None:
        return True, points

    # Pass 2: Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    retval, points = detector.detect(gray)
    if retval and points is not None:
        return True, points

    # Pass 3: Contrast / Threshold scan
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    retval, points = detector.detect(thresh)
    if retval and points is not None:
        return True, points

    # Pass 4: Resized scan (if original is very large or small)
    h, w = gray.shape[:2]
    if w > 1200:
        scaled = cv2.resize(gray, (1000, int(h * (1000 / w))))
        retval, points = detector.detect(scaled)
        if retval and points is not None:
            return True, points

    return False, None
