from typing import List, Optional, Tuple, Union
# pyrefly: ignore [missing-import]
import cv2
import numpy as np

try:
    import zxingcpp as _zxingcpp
    _ZXING_AVAILABLE = True
except ImportError:
    _ZXING_AVAILABLE = False


def format_decoded_payload(raw_data: Union[str, bytes]) -> str:
    
    if raw_data is None:
        return ""

    if isinstance(raw_data, str):
        return raw_data.strip()

    if isinstance(raw_data, (bytes, bytearray)):
        if len(raw_data) == 0:
            return ""
        # 1. Try UTF-8 decoding
        try:
            return raw_data.decode("utf-8")
        except UnicodeDecodeError:
            pass

        # 2. Try Latin-1 / ISO-8859-1 (preserves full 0-255 byte fidelity)
        try:
            return raw_data.decode("latin-1")
        except Exception:
            pass

        # 3. Fallback to hex representation
        return raw_data.hex()

    return str(raw_data)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    
    pts = np.array(pts, dtype=np.float32).reshape((4, 2))
    center = np.mean(pts, axis=0)

    # Compute polar angle of each point relative to center
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sort_idx = np.argsort(angles)
    sorted_pts = pts[sort_idx]

    # Find the vertex closest to the top-left coordinate space
    min_x, min_y = np.min(pts[:, 0]), np.min(pts[:, 1])
    dists_to_tl = np.sum((sorted_pts - np.array([min_x, min_y])) ** 2, axis=1)
    tl_idx = int(np.argmin(dists_to_tl))

    # Shift array so top-left is first
    ordered = np.roll(sorted_pts, -tl_idx, axis=0)
    return ordered


def expand_quad_points(pts: np.ndarray, factor: float = 1.06) -> np.ndarray:
    """
    Expand quad points outward from the centroid by a given scale factor.
    Compensates for tight or slightly inaccurate quad bounding boxes that clip finder patterns or quiet zones.
    """
    pts = np.array(pts, dtype=np.float32).reshape((4, 2))
    center = np.mean(pts, axis=0)
    expanded = center + (pts - center) * factor
    return expanded


def add_padding_to_image(image: np.ndarray, pad: int = 60) -> Tuple[np.ndarray, int]:
    
    if pad <= 0:
        return image, 0
    padded = cv2.copyMakeBorder(
        image, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    return padded, pad


def rectify_quad(
    image: np.ndarray,
    quad_points: np.ndarray,
    target_size: int = 500,
    quiet_zone: int = 30,
) -> np.ndarray:
    
    ordered = order_quad_points(quad_points)

    content_size = max(200, target_size - 2 * quiet_zone)
    total_size = content_size + 2 * quiet_zone
    qz = quiet_zone

    # Destination square: QR content placed with quiet-zone borders
    dst_pts = np.array(
        [
            [qz, qz],
            [qz + content_size, qz],
            [qz + content_size, qz + content_size],
            [qz, qz + content_size],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(ordered, dst_pts)
    warped = cv2.warpPerspective(
        image,
        M,
        (total_size, total_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped


def generate_image_variants(bgr_image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    
    variants: List[Tuple[str, np.ndarray]] = []
    if bgr_image is None or bgr_image.size == 0:
        return variants

    # 1. Original crop (BGR)
    variants.append(("original", bgr_image))

    # Get grayscale base
    if len(bgr_image.shape) == 3 and bgr_image.shape[2] == 3:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr_image.copy()

    # 2. Grayscale
    variants.append(("grayscale", gray))

    # 3. CLAHE (Contrast-Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)
    variants.append(("clahe", clahe_gray))

    # 4. Strong CLAHE (for faint / very low contrast barcodes)
    clahe_strong = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
    clahe_strong_gray = clahe_strong.apply(gray)
    variants.append(("clahe_strong", clahe_strong_gray))

    # 5. Unsharp Mask Deblur (for out-of-focus or motion-blurred scans)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    unsharp = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
    variants.append(("unsharp_deblur", unsharp))

    # 6. Bilateral Filter Denoising (for noisy sensor captures)
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    variants.append(("bilateral_denoise", denoised))

    # 7. Gamma Boost for Dim / Underexposed Images (gamma=0.5)
    table_dim = np.array([((i / 255.0) ** (1.0 / 0.5)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    brightened = cv2.LUT(gray, table_dim)
    variants.append(("brightened_gamma", brightened))

    # 8. Min-Max Normalization
    normalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    variants.append(("contrast_normalized", normalized))

    # 9. Otsu Threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))

    # 10. CLAHE + Otsu Threshold
    _, clahe_otsu = cv2.threshold(clahe_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_otsu", clahe_otsu))

    # 11. Adaptive Gaussian Threshold (good for uneven / gradient lighting)
    adaptive_gauss = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    variants.append(("adaptive_gauss", adaptive_gauss))

    # 12. Adaptive Mean Threshold
    adaptive_mean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 7
    )
    variants.append(("adaptive_mean", adaptive_mean))

    # 13. Sharpened grayscale
    sharpen_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(gray, -1, sharpen_k)
    variants.append(("sharpened", sharpened))

    # 14. Morphological Close (reconnects broken / faint QR modules)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_close)
    variants.append(("morph_close", morph_close))

    # 15. Inverted (for dark-background / light-module QR codes)
    inverted = cv2.bitwise_not(gray)
    variants.append(("inverted", inverted))

    # 16. Upscaled grayscale (for small QR codes)
    upscaled_gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
    variants.append(("upscaled_2x", upscaled_gray))

    # 17. Upscaled + Otsu (best for pixelated small QR codes)
    _, upscaled_otsu = cv2.threshold(upscaled_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("upscaled_2x_otsu", upscaled_otsu))

    # 18. Upscaled + Unsharp
    upscaled_blurred = cv2.GaussianBlur(upscaled_gray, (0, 0), sigmaX=2.0)
    upscaled_unsharp = cv2.addWeighted(upscaled_gray, 1.8, upscaled_blurred, -0.8, 0)
    variants.append(("upscaled_2x_unsharp", upscaled_unsharp))

    # 19. Illumination Normalization (eliminates severe shadow gradients across card)
    bg = cv2.medianBlur(gray, 51)
    norm_illum = cv2.divide(gray, np.maximum(bg, 1), scale=255).astype(np.uint8)
    variants.append(("illumination_normalized", norm_illum))

    return variants


def try_decode_image(
    image: np.ndarray,
    detectors: list,
) -> Tuple[bool, str, str]:
    
    for det in detectors:
        # Try detectAndDecodeBytes (handles binary/null-byte payloads correctly)
        if hasattr(det, "detectAndDecodeBytes"):
            try:
                res = det.detectAndDecodeBytes(image)
                if len(res) >= 2 and res[1] is not None:
                    raw_bytes = res[0]
                    payload = format_decoded_payload(raw_bytes)
                    if payload:
                        return True, payload, "bytes"
            except Exception:
                pass

        # Try detectAndDecode (string)
        try:
            res = det.detectAndDecode(image)
            if len(res) >= 1:
                text = res[0]
                if text and len(str(text).strip()) > 0:
                    return True, str(text).strip(), "text"
        except Exception:
            pass

    return False, "", ""


def try_decode_with_zxing(
    image: np.ndarray,
) -> Tuple[bool, str, str]:
    
    if not _ZXING_AVAILABLE or image is None or image.size == 0:
        return False, "", ""

    try:
        # zxingcpp accepts BGR or grayscale numpy arrays directly
        results = _zxingcpp.read_barcodes(
            image,
            formats=_zxingcpp.barcode_formats_from_str("QRCode|MicroQRCode|RMQRCode"),
            try_rotate=True,
            try_downscale=True,
            try_invert=True,
        )
        for r in results:
            if r.valid and r.text and str(r.text).strip():
                return True, str(r.text).strip(), "zxingcpp"
        # Also try bytes for binary payloads
        for r in results:
            if r.valid and r.bytes:
                payload = format_decoded_payload(bytes(r.bytes))
                if payload:
                    return True, payload, "zxingcpp_bytes"
    except Exception:
        pass

    return False, "", ""


def try_decode_with_zxing_multi(
    image: np.ndarray,
) -> Tuple[bool, List[str], List[np.ndarray], List[str]]:
    """
    Decode multiple QR codes using ZXing with precise bounding quadrilateral extraction.
    """
    if not _ZXING_AVAILABLE or image is None or image.size == 0:
        return False, [], [], []

    try:
        results = _zxingcpp.read_barcodes(
            image,
            formats=_zxingcpp.barcode_formats_from_str("QRCode|MicroQRCode|RMQRCode"),
            try_rotate=True,
            try_downscale=True,
            try_invert=True,
        )
        texts = []
        quads = []
        methods = []
        for r in results:
            if not r.valid:
                continue
            payload = ""
            method = ""
            if r.text and str(r.text).strip():
                payload = str(r.text).strip()
                method = "zxingcpp_multi"
            elif r.bytes:
                payload = format_decoded_payload(bytes(r.bytes))
                if payload:
                    method = "zxingcpp_multi_bytes"

            if payload:
                p = r.position
                if p is not None:
                    quad = np.array([
                        [float(p.top_left.x), float(p.top_left.y)],
                        [float(p.top_right.x), float(p.top_right.y)],
                        [float(p.bottom_right.x), float(p.bottom_right.y)],
                        [float(p.bottom_left.x), float(p.bottom_left.y)],
                    ], dtype=np.float32)
                else:
                    quad = np.zeros((4, 2), dtype=np.float32)
                texts.append(payload)
                quads.append(quad)
                methods.append(method)

        if texts and quads:
            return True, texts, quads, methods
    except Exception:
        pass

    return False, [], [], []


def try_decode_image_multi(
    image: np.ndarray,
    detectors: list,
) -> Tuple[bool, List[str], List[np.ndarray], List[str]]:
    
    for det in detectors:
        # Try detectAndDecodeBytesMulti
        if hasattr(det, "detectAndDecodeBytesMulti"):
            try:
                retval, decoded_bytes_tuple, points, _ = det.detectAndDecodeBytesMulti(image)
                if points is not None and len(points) > 0:
                    quads = _extract_quads(points)
                    texts = []
                    methods = []
                    for b in (decoded_bytes_tuple or []):
                        formatted = format_decoded_payload(b)
                        texts.append(formatted)
                        methods.append("multi_bytes" if formatted else "")
                    while len(texts) < len(quads):
                        texts.append("")
                        methods.append("")
                    if quads:
                        return True, texts, quads, methods
            except Exception:
                pass

        # Try detectAndDecodeMulti (string)
        try:
            retval, decoded_info, points, _ = det.detectAndDecodeMulti(image)
            if points is not None and len(points) > 0:
                quads = _extract_quads(points)
                texts = list(decoded_info) if decoded_info is not None else []
                methods = ["multi_text" if t else "" for t in texts]
                while len(texts) < len(quads):
                    texts.append("")
                    methods.append("")
                if quads:
                    return True, texts, quads, methods
        except Exception:
            pass

    return False, [], [], []


def _extract_quads(points: np.ndarray) -> List[np.ndarray]:
    """Normalize various OpenCV points shapes into a list of (4, 2) quads."""
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


class QRCodeDecoderWrapper:
    """
    Multi-stage QR Code Decoder wrapping OpenCV's QRCodeDetector / QRCodeDetectorAruco.

    Escalation strategy:
    1. Direct detectAndDecode on original image (legacy detector first for rotation tolerance)
    2. Multi-QR detectAndDecodeMulti on original image
    3. ZXing multi-decode with bounding quad extraction
    4. Padded image (prevents edge clipping on rotated/near-border QRs)
    5. Perspective warp → fresh detectAndDecode (correct: re-detect on warped, not decode with known pts)
    6. Multi-variant image processing on each warped size
    """

    def __init__(self, primary_detector=None, legacy_detector=None):
        # Legacy detector handles arbitrary rotations better; ArUco is better for dense codes
        if hasattr(cv2, "QRCodeDetectorAruco"):
            self._aruco_detector = primary_detector or cv2.QRCodeDetectorAruco()
        else:
            self._aruco_detector = legacy_detector or cv2.QRCodeDetector()
        self._legacy_detector = legacy_detector or cv2.QRCodeDetector()

        # For rotation-robust scenarios, try legacy first, ArUco second
        self._detectors_rotation = [self._legacy_detector, self._aruco_detector]
        # For dense/high-density QR, ArUco is better
        self._detectors_dense = [self._aruco_detector, self._legacy_detector]

    def detect_and_decode_multi(
        self, image: np.ndarray
    ) -> Tuple[bool, List[str], List[np.ndarray], List[str]]:
        
        if image is None or image.size == 0:
            return False, [], [], []

        # 1. Try multi-decode with OpenCV detector orderings
        for detectors in (self._detectors_dense, self._detectors_rotation):
            ok, texts, quads, methods = try_decode_image_multi(image, detectors)
            if ok and quads:
                return True, texts, quads, methods

        # 2. Fallback: single detectAndDecode
        ok, payload, method_tag = try_decode_image(image, self._detectors_rotation)
        if ok and payload:
            # Single detection — detect quad separately
            for det in self._detectors_rotation:
                try:
                    det_ok, pts = det.detect(image)
                    if det_ok and pts is not None:
                        p = np.array(pts, dtype=np.float32)
                        if p.ndim == 3:
                            p = p[0]
                        if p.shape == (4, 2):
                            return True, [payload], [p], [f"single_{method_tag}"]
                except Exception:
                    pass
            return True, [payload], [], [f"single_{method_tag}"]

        # 3. ZXing multi-decode with position extraction (finds multiple codes with quads)
        zx_multi_ok, zx_texts, zx_quads, zx_methods = try_decode_with_zxing_multi(image)
        if zx_multi_ok and zx_quads:
            return True, zx_texts, zx_quads, zx_methods

        # 4. Fallback: OpenCV geometry detection + ZXing payload decode
        for det in self._detectors_rotation:
            try:
                det_ok, pts = det.detect(image)
                if det_ok and pts is not None:
                    p = np.array(pts, dtype=np.float32)
                    if p.ndim == 3:
                        p = p[0]
                    if p.shape == (4, 2):
                        zx_ok, zx_payload, zx_method = try_decode_with_zxing(image)
                        if zx_ok and zx_payload:
                            return True, [zx_payload], [p], [zx_method]
            except Exception:
                pass

        # 5. ZXing on full image (finds geometry + decodes independently)
        zx_ok, zx_payload, zx_method = try_decode_with_zxing(image)
        if zx_ok and zx_payload:
            return True, [zx_payload], [], [zx_method]

        return False, [], [], []

    def decode_quad(
        self, image: np.ndarray, quad_points: np.ndarray
    ) -> Tuple[bool, Optional[str], Optional[str], int]:
        """
        Fallback decoding pipeline for detected-but-undecodable QR codes.
        
        Handles cases where detected quadrilateral coordinates are inaccurate or
        image quality is degraded by:
        1. Checking the full image and generously padded full image (OpenCV & ZXing).
        2. Extracting a larger padded region around the detected QR from the full image,
           applying controlled upscaling (1.5x, 2.0x, 3.0x, 4.0x) and image enhancement variants.
        3. Applying perspective rectification with quadrilateral expansion factors to compensate
           for boundary inaccuracies, tested at multiple target sizes and enhancement variants.
        4. Leveraging exhaustive ZXing fallbacks across all padded, upscaled, and rectified views.
        """
        if image is None or image.size == 0 or quad_points is None:
            return False, None, None, 0

        attempts = 0
        pts = np.array(quad_points, dtype=np.float32)
        if pts.shape != (4, 2):
            pts = pts.reshape(4, 2)

        h, w = image.shape[:2]

        # Calculate bounding box and estimated dimension of the QR code
        min_x = float(np.min(pts[:, 0]))
        max_x = float(np.max(pts[:, 0]))
        min_y = float(np.min(pts[:, 1]))
        max_y = float(np.max(pts[:, 1]))
        qr_w = max(1.0, max_x - min_x)
        qr_h = max(1.0, max_y - min_y)
        qr_dim = max(qr_w, qr_h, 30.0)

        # --- Stage 1: Full image & generously padded full image fallbacks ---
        # Add generous padding to full image to prevent corner clipping on rotated or near-border QRs
        full_pad_amount = max(80, int(min(w, h) * 0.25))
        padded_full_img, _ = add_padding_to_image(image, pad=full_pad_amount)

        # Try OpenCV on padded full image
        for detectors, method_tag in [
            (self._detectors_rotation, "padded_rotation"),
            (self._detectors_dense, "padded_dense"),
        ]:
            attempts += 1
            ok, payload, _ = try_decode_image(padded_full_img, detectors)
            if ok and payload:
                return True, payload, f"direct_{method_tag}", attempts

        # Try ZXing directly on the full image and padded full image
        attempts += 1
        zx_ok, zx_payload, zx_method = try_decode_with_zxing(image)
        if zx_ok and zx_payload:
            return True, zx_payload, f"zxingcpp_full_{zx_method}", attempts

        attempts += 1
        zx_ok, zx_payload, zx_method = try_decode_with_zxing(padded_full_img)
        if zx_ok and zx_payload:
            return True, zx_payload, f"zxingcpp_full_padded_{zx_method}", attempts

        # --- Stage 2: Larger padded region crop around detected QR & controlled upscale ---
        # A larger margin (50-80% of QR dimension) guarantees complete quiet zones and accounts for inaccurate quads
        crop_pad = max(80, int(qr_dim * 0.65))
        x_min = max(0, int(min_x - crop_pad))
        y_min = max(0, int(min_y - crop_pad))
        x_max = min(w, int(max_x + crop_pad))
        y_max = min(h, int(max_y + crop_pad))

        if x_max > x_min and y_max > y_min:
            crop = image[y_min:y_max, x_min:x_max]
            if crop.size > 0:
                # Add quiet zone border to crop
                padded_crop, _ = add_padding_to_image(crop, pad=40)

                # Try OpenCV & ZXing on the padded crop directly
                for detectors, method_tag in [
                    (self._detectors_rotation, "padded_crop_rotation"),
                    (self._detectors_dense, "padded_crop_dense"),
                ]:
                    attempts += 1
                    ok, payload, _ = try_decode_image(padded_crop, detectors)
                    if ok and payload:
                        return True, payload, f"direct_{method_tag}", attempts

                attempts += 1
                zx_ok, zx_payload, zx_method = try_decode_with_zxing(padded_crop)
                if zx_ok and zx_payload:
                    return True, zx_payload, f"zxingcpp_padded_crop_{zx_method}", attempts

                # Controlled multi-scale upscaling of the padded crop (1.5x, 2.0x, 3.0x, 4.0x)
                pts_in_crop = pts - np.array([x_min, y_min], dtype=np.float32)

                for scale in (1.5, 2.0, 3.0, 4.0):
                    scaled_crop = cv2.resize(
                        crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
                    )
                    scaled_padded_crop, _ = add_padding_to_image(scaled_crop, pad=int(30 * scale))

                    # OpenCV on scaled crop
                    attempts += 1
                    ok, payload, _ = try_decode_image(scaled_padded_crop, self._detectors_rotation)
                    if ok and payload:
                        return True, payload, f"crop_upscale_{scale}x_rotation", attempts

                    attempts += 1
                    ok, payload, _ = try_decode_image(scaled_padded_crop, self._detectors_dense)
                    if ok and payload:
                        return True, payload, f"crop_upscale_{scale}x_dense", attempts

                    # ZXing on scaled crop
                    attempts += 1
                    zx_ok, zx_payload, zx_method = try_decode_with_zxing(scaled_padded_crop)
                    if zx_ok and zx_payload:
                        return True, zx_payload, f"zxingcpp_crop_{scale}x_{zx_method}", attempts

                    # Preprocessing variants on scaled crop (grayscale, CLAHE, Otsu, Sharpened)
                    if len(scaled_crop.shape) == 3 and scaled_crop.shape[2] == 3:
                        gray_crop = cv2.cvtColor(scaled_crop, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_crop = scaled_crop.copy()

                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    clahe_crop = clahe.apply(gray_crop)

                    _, otsu_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    sharpen_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                    sharpened_crop = cv2.filter2D(gray_crop, -1, sharpen_k)

                    for var_name, var_crop in [
                        ("clahe", clahe_crop),
                        ("otsu", otsu_crop),
                        ("sharpened", sharpened_crop),
                    ]:
                        padded_var_crop, _ = add_padding_to_image(var_crop, pad=int(25 * scale))

                        attempts += 1
                        ok, payload, _ = try_decode_image(padded_var_crop, self._detectors_rotation)
                        if ok and payload:
                            return True, payload, f"crop_upscale_{scale}x_{var_name}", attempts

                        attempts += 1
                        zx_ok, zx_payload, zx_method = try_decode_with_zxing(padded_var_crop)
                        if zx_ok and zx_payload:
                            return True, zx_payload, f"zxingcpp_crop_{scale}x_{var_name}_{zx_method}", attempts

                    # Perspective rectify on upscaled crop with scaled quad
                    scaled_pts_in_crop = pts_in_crop * scale
                    for exp_factor in (1.0, 1.08):
                        try:
                            exp_crop_pts = (
                                expand_quad_points(scaled_pts_in_crop, factor=exp_factor)
                                if exp_factor != 1.0
                                else scaled_pts_in_crop
                            )
                            warped_scaled = rectify_quad(scaled_crop, exp_crop_pts, target_size=600, quiet_zone=35)
                            attempts += 1
                            ok, payload, _ = try_decode_image(warped_scaled, self._detectors_rotation)
                            if ok and payload:
                                return True, payload, f"crop_upscale_{scale}x_rectified_exp{exp_factor}", attempts

                            attempts += 1
                            zx_ok, zx_payload, zx_method = try_decode_with_zxing(warped_scaled)
                            if zx_ok and zx_payload:
                                return True, zx_payload, f"zxingcpp_crop_{scale}x_rectified_{zx_method}", attempts
                        except Exception:
                            pass

        # --- Stage 3: Perspective rectification with quad margin expansion & multi-variants ---
        # When detected quadrilateral points are slightly tight/skewed, expanding vertices compensates
        expansion_factors = [1.0, 1.06, 1.12]
        target_sizes = [500, 750, 1000, 380]
        quiet_zones = [30, 45, 60, 25]

        for exp in expansion_factors:
            exp_pts = expand_quad_points(pts, factor=exp) if exp != 1.0 else pts

            for size, qz in zip(target_sizes, quiet_zones):
                try:
                    warped = rectify_quad(image, exp_pts, target_size=size, quiet_zone=qz)
                    variants = generate_image_variants(warped)

                    for var_name, var_img in variants:
                        attempts += 1
                        ok, payload, _ = try_decode_image(var_img, self._detectors_rotation)
                        if ok and payload:
                            return True, payload, f"perspective_{size}px_exp{exp}_{var_name}", attempts

                        attempts += 1
                        ok, payload, _ = try_decode_image(var_img, self._detectors_dense)
                        if ok and payload:
                            return True, payload, f"perspective_{size}px_exp{exp}_{var_name}_dense", attempts

                        # ZXing fallback on each perspective variant
                        attempts += 1
                        zx_ok, zx_payload, zx_method = try_decode_with_zxing(var_img)
                        if zx_ok and zx_payload:
                            return True, zx_payload, f"zxingcpp_perspective_{size}px_exp{exp}_{var_name}_{zx_method}", attempts

                except Exception:
                    continue

        # --- Stage 4: Final full-image & enhanced fallback with ZXing ---
        try:
            # Grayscale & CLAHE full image with ZXing
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_full = image.copy()

            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            clahe_full = clahe.apply(gray_full)
            attempts += 1
            zx_ok, zx_payload, zx_method = try_decode_with_zxing(clahe_full)
            if zx_ok and zx_payload:
                return True, zx_payload, f"zxingcpp_clahe_full_{zx_method}", attempts

            _, otsu_full = cv2.threshold(gray_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            attempts += 1
            zx_ok, zx_payload, zx_method = try_decode_with_zxing(otsu_full)
            if zx_ok and zx_payload:
                return True, zx_payload, f"zxingcpp_otsu_full_{zx_method}", attempts
        except Exception:
            pass

        return False, None, None, attempts
