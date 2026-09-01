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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)
    variants.append(("clahe", clahe_gray))

    # 4. Otsu Threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))

    # 5. Adaptive Threshold (good for uneven lighting)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    variants.append(("adaptive", adaptive))

    # 6. Sharpened grayscale
    sharpen_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(gray, -1, sharpen_k)
    variants.append(("sharpened", sharpened))

    # 7. Inverted (for dark-background / light-module QR codes)
    inverted = cv2.bitwise_not(gray)
    variants.append(("inverted", inverted))

    # 8. Upscaled grayscale (for small QR codes)
    upscaled_gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
    variants.append(("upscaled_2x", upscaled_gray))

    # 9. Upscaled + Otsu (best for pixelated small QR codes)
    _, upscaled_otsu = cv2.threshold(upscaled_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("upscaled_2x_otsu", upscaled_otsu))

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
            if r.valid and r.text:
                return True, r.text, "zxingcpp"
        # Also try bytes for binary payloads
        for r in results:
            if r.valid and r.bytes:
                payload = format_decoded_payload(bytes(r.bytes))
                if payload:
                    return True, payload, "zxingcpp_bytes"
    except Exception:
        pass

    return False, "", ""


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
    3. Padded image (prevents edge clipping on rotated/near-border QRs)
    4. Perspective warp → fresh detectAndDecode (correct: re-detect on warped, not decode with known pts)
    5. Multi-variant image processing on each warped size
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

        # Try multi-decode with both detector orderings
        for detectors in (self._detectors_dense, self._detectors_rotation):
            ok, texts, quads, methods = try_decode_image_multi(image, detectors)
            if ok and quads:
                return True, texts, quads, methods

        # Fallback: single detectAndDecode
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

        # --- Final fallback: zxingcpp ---
        # Try to detect quad with OpenCV first (geometry only), then decode payload with zxingcpp
        for det in self._detectors_rotation:
            try:
                det_ok, pts = det.detect(image)
                if det_ok and pts is not None:
                    p = np.array(pts, dtype=np.float32)
                    if p.ndim == 3:
                        p = p[0]
                    if p.shape == (4, 2):
                        # OpenCV found the QR geometry but couldn't decode — try zxingcpp
                        zx_ok, zx_payload, zx_method = try_decode_with_zxing(image)
                        if zx_ok and zx_payload:
                            return True, [zx_payload], [p], [zx_method]
            except Exception:
                pass

        # zxingcpp on full image (finds geometry + decodes independently)
        zx_ok, zx_payload, zx_method = try_decode_with_zxing(image)
        if zx_ok and zx_payload:
            # No quad available from zxingcpp in this path — return without bbox
            return True, [zx_payload], [], [zx_method]

        return False, [], [], []

    def decode_quad(
        self, image: np.ndarray, quad_points: np.ndarray
    ) -> Tuple[bool, Optional[str], Optional[str], int]:
        
        if image is None or image.size == 0 or quad_points is None:
            return False, None, None, 0

        attempts = 0
        pts = np.array(quad_points, dtype=np.float32)
        if pts.shape != (4, 2):
            pts = pts.reshape(4, 2)

        # --- Stage 1: Padded original image (fixes edge-clipping for rotated QRs) ---
        # Add enough padding to prevent corners of rotated QR from being clipped
        h, w = image.shape[:2]
        pad_amount = max(60, min(w, h) // 4)
        padded_img, pad = add_padding_to_image(image, pad=pad_amount)
        pts_padded = pts + np.array([pad, pad], dtype=np.float32)

        for detectors, method_tag in [
            (self._detectors_rotation, "padded_rotation"),
            (self._detectors_dense, "padded_dense"),
        ]:
            attempts += 1
            ok, payload, _ = try_decode_image(padded_img, detectors)
            if ok and payload:
                return True, payload, f"direct_{method_tag}", attempts

        # --- Stage 2: Perspective rectification → fresh detectAndDecode ---
        target_sizes = [500, 700, 350]
        quiet_zones = [30, 40, 20]

        for size, qz in zip(target_sizes, quiet_zones):
            try:
                warped = rectify_quad(image, pts, target_size=size, quiet_zone=qz)

                # Generate variants and try fresh decode on each
                variants = generate_image_variants(warped)
                for var_name, var_img in variants:
                    attempts += 1
                    ok, payload, method_tag = try_decode_image(var_img, self._detectors_rotation)
                    if ok and payload:
                        return True, payload, f"perspective_{size}px_{var_name}", attempts

                    # Also try ArUco on each variant (better for dense codes)
                    attempts += 1
                    ok, payload, method_tag = try_decode_image(var_img, self._detectors_dense)
                    if ok and payload:
                        return True, payload, f"perspective_{size}px_{var_name}_aruco", attempts

            except Exception:
                continue

        # --- Stage 3: Multi-scale upscale of original with padded QR region crop ---
        for scale in [2.0, 3.0, 4.0]:
            try:
                # Crop to the QR region with extra context, then upscale
                x_min = max(0, int(np.min(pts[:, 0])) - pad_amount // 2)
                y_min = max(0, int(np.min(pts[:, 1])) - pad_amount // 2)
                x_max = min(w, int(np.max(pts[:, 0])) + pad_amount // 2)
                y_max = min(h, int(np.max(pts[:, 1])) + pad_amount // 2)

                if x_max <= x_min or y_max <= y_min:
                    continue

                crop = image[y_min:y_max, x_min:x_max]
                if crop.size == 0:
                    continue

                # Upscale the crop
                scaled_crop = cv2.resize(
                    crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
                )

                attempts += 1
                ok, payload, _ = try_decode_image(scaled_crop, self._detectors_rotation)
                if ok and payload:
                    return True, payload, f"crop_upscale_{scale}x", attempts

                # Also try otsu on the scaled crop
                if len(scaled_crop.shape) == 3:
                    gray_crop = cv2.cvtColor(scaled_crop, cv2.COLOR_BGR2GRAY)
                else:
                    gray_crop = scaled_crop
                _, otsu_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                attempts += 1
                ok, payload, _ = try_decode_image(otsu_crop, self._detectors_rotation)
                if ok and payload:
                    return True, payload, f"crop_upscale_{scale}x_otsu", attempts

                # Try perspective rectify on the upscaled crop
                pts_in_crop = (pts - np.array([x_min, y_min])) * scale
                try:
                    warped_scaled = rectify_quad(scaled_crop, pts_in_crop, target_size=500, quiet_zone=30)
                    attempts += 1
                    ok, payload, _ = try_decode_image(warped_scaled, self._detectors_rotation)
                    if ok and payload:
                        return True, payload, f"crop_upscale_{scale}x_perspective", attempts
                except Exception:
                    pass

                # zxingcpp on the upscaled crop (plain, otsu, and perspective-rectified)
                # Tried only after all OpenCV attempts for this scale fail.
                attempts += 1
                zx_ok, zx_payload, zx_method = try_decode_with_zxing(scaled_crop)
                if zx_ok and zx_payload:
                    return True, zx_payload, f"zxingcpp_crop_{scale}x_{zx_method}", attempts

                attempts += 1
                zx_ok, zx_payload, zx_method = try_decode_with_zxing(otsu_crop)
                if zx_ok and zx_payload:
                    return True, zx_payload, f"zxingcpp_crop_{scale}x_otsu_{zx_method}", attempts

                try:
                    warped_scaled_zx = rectify_quad(scaled_crop, pts_in_crop, target_size=500, quiet_zone=30)
                    attempts += 1
                    zx_ok, zx_payload, zx_method = try_decode_with_zxing(warped_scaled_zx)
                    if zx_ok and zx_payload:
                        return True, zx_payload, f"zxingcpp_crop_{scale}x_perspective_{zx_method}", attempts
                except Exception:
                    pass

            except Exception:
                continue

        # --- Final fallback: zxingcpp on perspective-rectified crop ---
        try:
            warped_final = rectify_quad(image, pts, target_size=500, quiet_zone=30)
            zx_ok, zx_payload, zx_method = try_decode_with_zxing(warped_final)
            if zx_ok and zx_payload:
                attempts += 1
                return True, zx_payload, f"zxingcpp_perspective_{zx_method}", attempts

            # Also try zxingcpp on the padded original
            zx_ok, zx_payload, zx_method = try_decode_with_zxing(padded_img)
            if zx_ok and zx_payload:
                attempts += 1
                return True, zx_payload, f"zxingcpp_padded_{zx_method}", attempts
        except Exception:
            pass

        return False, None, None, attempts
