"""
QR Code Detector component using OpenCV's QRCodeDetectorAruco and QRCodeDetector.
Supports multi-scale search, padding, and robust coordinate mapping.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


class QRCodeDetectorWrapper:
    """
    Detector wrapper supporting both OpenCV QRCodeDetectorAruco and QRCodeDetector.
    Handles high-density, low-resolution, and skewed QR code detection.
    """

    def __init__(self):
        if hasattr(cv2, "QRCodeDetectorAruco"):
            self._primary_detector = cv2.QRCodeDetectorAruco()
        else:
            self._primary_detector = cv2.QRCodeDetector()
        self._legacy_detector = cv2.QRCodeDetector()

    @property
    def primary_detector(self):
        """Access the primary OpenCV detector."""
        return self._primary_detector

    @property
    def legacy_detector(self):
        """Access the legacy OpenCV detector."""
        return self._legacy_detector

    def detect_multi(self, image: np.ndarray) -> Tuple[bool, List[np.ndarray]]:
        """
        Detect multiple QR code bounding polygons in the input image.
        Uses primary ArUco detector with legacy detector fallback.
        """
        if image is None or image.size == 0:
            return False, []

        # 1. Try Primary Detector (ArUco)
        for detector in (self._primary_detector, self._legacy_detector):
            try:
                detected, points = detector.detectMulti(image)
                if detected and points is not None and len(points) > 0:
                    quads = self._extract_quad_list(points)
                    if quads:
                        return True, quads
            except Exception:
                pass

        # 2. Try Single Detection fallback
        single_detected, single_quad = self.detect_single(image)
        if single_detected and single_quad is not None:
            return True, [single_quad]

        return False, []

    def detect_single(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect a single QR code bounding box in the input image.
        """
        if image is None or image.size == 0:
            return False, None

        for detector in (self._primary_detector, self._legacy_detector):
            try:
                detected, points = detector.detect(image)
                if detected and points is not None and len(points) > 0:
                    pts = np.array(points, dtype=np.float32)
                    if pts.ndim == 3 and pts.shape[0] == 1:
                        pts = pts[0]
                    if pts.shape == (4, 2):
                        return True, pts
            except Exception:
                pass

        return False, None

    def _extract_quad_list(self, points: np.ndarray) -> List[np.ndarray]:
        """Normalize OpenCV points shapes into a list of (4, 2) numpy arrays."""
        pts = np.array(points, dtype=np.float32)
        quads = []

        if pts.ndim == 4:
            for i in range(pts.shape[0]):
                q = pts[i, 0]
                if q.shape == (4, 2):
                    quads.append(q)
        elif pts.ndim == 3:
            for i in range(pts.shape[0]):
                q = pts[i]
                if q.shape == (4, 2):
                    quads.append(q)
        elif pts.ndim == 2 and pts.shape == (4, 2):
            quads.append(pts)

        return quads
