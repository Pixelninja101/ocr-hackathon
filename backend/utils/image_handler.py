"""
Image processing utilities using OpenCV, NumPy, and Pillow.
"""

import base64
import io
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image


def bytes_to_cv2_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes in-memory into an OpenCV BGR numpy array.
    Uses cv2.imdecode with a fallback to Pillow for unusual color spaces or formats.
    """
    if not image_bytes:
        raise ValueError("Empty image byte buffer received.")

    # Primary decode using OpenCV
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    cv2_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if cv2_img is not None and cv2_img.size > 0:
        return cv2_img

    # Fallback to Pillow for CMYK, TIFF, or special color profile images
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        rgb_arr = np.array(pil_img)
        cv2_img = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        return cv2_img
    except Exception as exc:
        raise ValueError(f"Failed to decode image buffer into valid image array: {exc}") from exc


def cv2_to_base64_data_url(cv2_img: np.ndarray, max_dim: int = 1600, quality: int = 85) -> str:
    """
    Encode an OpenCV BGR image into a lightweight Base64 JPEG data URL.
    Used specifically for rendering PDF page previews to the frontend.
    """
    if cv2_img is None or cv2_img.size == 0:
        return ""

    h, w = cv2_img.shape[:2]
    # Scale down preview if excessively large to keep payload lightweight
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        preview_img = cv2.resize(cv2_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        preview_img = cv2_img

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_buf = cv2.imencode(".jpg", preview_img, encode_params)
    if not success:
        return ""

    b64_str = base64.b64encode(encoded_buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64_str}"


def normalize_points(points: np.ndarray) -> List[List[float]]:
    """
    Convert OpenCV QR detector points array to a standard list of 4 coordinates:
    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    if points is None:
        return []

    pts = np.array(points, dtype=np.float32)
    # Squeeze extra dimensions if shape is (1, 4, 2) or (4, 2)
    if pts.ndim == 3 and pts.shape[0] == 1:
        pts = pts[0]

    if pts.shape == (4, 2):
        return [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in pts]

    return []


def calculate_bbox_metrics(points: List[List[float]]) -> dict:
    """
    Calculate center point, width, height, and area from 4 corner points.
    """
    if not points or len(points) < 4:
        return {"center": [0, 0], "width": 0, "height": 0, "area": 0}

    pts = np.array(points, dtype=np.float32)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))

    # Width: average of top and bottom edge lengths
    w1 = np.linalg.norm(pts[1] - pts[0])
    w2 = np.linalg.norm(pts[2] - pts[3])
    width = float((w1 + w2) / 2.0)

    # Height: average of left and right edge lengths
    h1 = np.linalg.norm(pts[3] - pts[0])
    h2 = np.linalg.norm(pts[2] - pts[1])
    height = float((h1 + h2) / 2.0)

    # Area using shoelace formula
    x = pts[:, 0]
    y = pts[:, 1]
    area = float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    return {
        "center": [round(cx, 2), round(cy, 2)],
        "width": round(width, 2),
        "height": round(height, 2),
        "area": round(area, 2),
    }


def enhance_for_qr(cv2_img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate standard OpenCV image processing variations to aid QR detection
    on low-contrast, dim, overexposed, blurry, or noisy document scans.
    Strictly uses OpenCV and NumPy image processing.
    """
    if cv2_img is None or cv2_img.size == 0:
        return []

    variants = []

    # 1. Grayscale base
    if len(cv2_img.shape) == 3 and cv2_img.shape[2] == 3:
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2_img.copy()
    variants.append(("grayscale", gray))

    # 2. Standard CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    variants.append(("clahe", clahe_img))

    # 3. Strong CLAHE (for very low-contrast / washed-out document scans)
    clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe_strong_img = clahe_strong.apply(gray)
    variants.append(("clahe_strong", clahe_strong_img))

    # 4. Unsharp Mask Deblurring (for blurry camera or scanner captures)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    unsharp = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
    variants.append(("unsharp_deblur", unsharp))

    # 5. Bilateral Filter Denoising (preserves sharp QR module edges while smoothing noise)
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    variants.append(("bilateral_denoise", denoised))

    # 6. Gamma Correction for Dim / Underexposed Images (gamma=0.5 boost)
    table_dim = np.array([((i / 255.0) ** (1.0 / 0.5)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    brightened = cv2.LUT(gray, table_dim)
    variants.append(("brightened_gamma", brightened))

    # 7. Gamma Correction for Overexposed / Washed-out Images (gamma=1.8)
    table_bright = np.array([((i / 255.0) ** (1.0 / 1.8)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    darkened = cv2.LUT(gray, table_bright)
    variants.append(("darkened_gamma", darkened))

    # 8. Min-Max Contrast Normalization
    normalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    variants.append(("contrast_normalized", normalized))

    # 9. Otsu Thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))

    # 10. CLAHE + Otsu Thresholding
    _, clahe_otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_otsu", clahe_otsu))

    # 11. Adaptive Gaussian Thresholding
    adaptive_gauss = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    variants.append(("adaptive_thresh", adaptive_gauss))

    # 12. Adaptive Mean Thresholding (robust for gradient lighting across card)
    adaptive_mean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 7
    )
    variants.append(("adaptive_mean", adaptive_mean))

    # 13. Sharpening kernel
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    variants.append(("sharpened", sharpened))

    # 14. Morphological Close (reconnects broken / degraded QR module pixels)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_close)
    variants.append(("morph_close", morph_close))

    # 15. Illumination Normalization (eliminates severe shadow gradients across card)
    bg = cv2.medianBlur(gray, 51)
    norm_illum = cv2.divide(gray, np.maximum(bg, 1), scale=255).astype(np.uint8)
    variants.append(("illumination_normalized", norm_illum))

    return variants
