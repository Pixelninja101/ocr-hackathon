"""
QR Code Detector component using OpenCV's QRCodeDetectorAruco and QRCodeDetector.
Supports multi-scale search, padding, image enhancement variants, and robust coordinate mapping.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

try:
    import zxingcpp as _zxingcpp
    _ZXING_AVAILABLE = True
except ImportError:
    _ZXING_AVAILABLE = False


class QRCodeDetectorWrapper:
    """
    Detector wrapper supporting both OpenCV QRCodeDetectorAruco and QRCodeDetector.
    Handles high-density, low-resolution, noisy, dim, blurry, and skewed QR code detection.
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
        Uses primary ArUco detector with legacy detector fallback,
        followed by image enhancements and ZXing position extraction fallback.
        """
        if image is None or image.size == 0:
            return False, []

        # 1. Try Primary & Legacy Detectors on original image
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

        # 3. Try detection on enhanced image variants (for dim, noisy, or blurry scans)
        enhanced_views = self._generate_quick_enhancements(image)
        for var_img in enhanced_views:
            for detector in (self._primary_detector, self._legacy_detector):
                try:
                    detected, points = detector.detectMulti(var_img)
                    if detected and points is not None and len(points) > 0:
                        quads = self._extract_quad_list(points)
                        if quads:
                            return True, quads
                except Exception:
                    pass

        # 4. Fallback to ZXing geometry extraction if available
        if _ZXING_AVAILABLE:
            zx_quads = self._extract_zxing_quads(image)
            if zx_quads:
                return True, zx_quads

            # Also try ZXing on enhanced views
            for var_img in enhanced_views:
                zx_quads = self._extract_zxing_quads(var_img)
                if zx_quads:
                    return True, zx_quads

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

    def _generate_quick_enhancements(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate fast enhancement variants for difficult scans."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        views = [gray]

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        views.append(clahe.apply(gray))

        # Unsharp mask (deblur)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
        unsharp = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
        views.append(unsharp)

        # Otsu threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        views.append(otsu)

        return views

    def _extract_zxing_quads(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract bounding quads using ZXing position coordinates."""
        quads = []
        try:
            results = _zxingcpp.read_barcodes(
                image,
                formats=_zxingcpp.barcode_formats_from_str("QRCode|MicroQRCode|RMQRCode"),
                try_rotate=True,
                try_downscale=True,
                try_invert=True,
            )
            for r in results:
                if r.valid and r.position is not None:
                    p = r.position
                    pts = np.array([
                        [float(p.top_left.x), float(p.top_left.y)],
                        [float(p.top_right.x), float(p.top_right.y)],
                        [float(p.bottom_right.x), float(p.bottom_right.y)],
                        [float(p.bottom_left.x), float(p.bottom_left.y)],
                    ], dtype=np.float32)
                    quads.append(pts)
        except Exception:
            pass
        return quads

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
