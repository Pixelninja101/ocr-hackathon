"""
Image Normalization and OpenCV Preprocessing Layer.
Produces standardized, high-quality image representations and variants
tailored for bilingual OCR and QR code detection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from document_processor.config import (
    DEBUG_SAVE_PREPROCESSED,
    ENABLE_DESKEW,
    MAX_DESKEW_ANGLE,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_INPUT_DIMENSION,
    MAX_IMAGE_WIDTH,
    MAX_PREPROCESSING_VARIANTS,
    MIN_IMAGE_WIDTH,
    UPSCALE_FACTOR,
)


class PreprocessingError(Exception):
    """Raised when image validation or preprocessing fails."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        """Returns safe, structured error dictionary without internal traces."""
        return {
            "success": False,
            "error": {
                "code": self.code,
                "message": self.message,
            },
        }


@dataclass
class PreprocessingVariant:
    """Represents a single preprocessed image variant."""

    name: str
    image: np.ndarray
    description: str
    target: str = "general"  # "ocr" | "qr" | "general"

    def to_dict(self, include_image: bool = True) -> dict[str, Any]:
        res = {
            "name": self.name,
            "description": self.description,
            "target": self.target,
            "shape": list(self.image.shape) if self.image is not None else [],
        }
        if include_image:
            res["image"] = self.image
        return res


@dataclass
class PreprocessedDocument:
    """
    Complete container for normalized and preprocessed document variants.
    Always retains access to the original image.
    """

    success: bool
    original: np.ndarray
    normalized: np.ndarray
    variants: list[PreprocessingVariant] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_variant(self, name: str) -> Optional[np.ndarray]:
        """Returns image array of the named variant if found."""
        for var in self.variants:
            if var.name == name:
                return var.image
        return None

    def get_ocr_images(self) -> list[Tuple[str, np.ndarray]]:
        """Returns list of (variant_name, image) variants suitable for OCR."""
        return [
            (v.name, v.image)
            for v in self.variants
            if v.target in ("ocr", "general")
        ]

    def get_qr_images(self) -> list[Tuple[str, np.ndarray]]:
        """Returns list of (variant_name, image) variants suitable for QR detection."""
        # Include original / normalized first as QR detection often performs best on unaltered images
        qr_list: list[Tuple[str, np.ndarray]] = [("normalized", self.normalized)]
        for v in self.variants:
            if v.target in ("qr", "general"):
                qr_list.append((v.name, v.image))
        return qr_list

    def to_dict(self, include_images: bool = True) -> dict[str, Any]:
        """Converts to dictionary representation."""
        res: dict[str, Any] = {
            "success": self.success,
            "metadata": self.metadata,
            "variants": [v.to_dict(include_image=include_images) for v in self.variants],
        }
        if include_images:
            res["original"] = self.original
            res["normalized"] = self.normalized
        return res


def validate_image_input(
    image: Any,
    max_input_dimension: Optional[int] = None,
) -> np.ndarray:
    """
    Validates that the input is a valid, non-empty OpenCV/NumPy image array.
    Guards against decompression bombs / excessive memory consumption by checking
    dimensions against MAX_IMAGE_INPUT_DIMENSION.
    """
    if image is None:
        raise PreprocessingError("INVALID_IMAGE", "Image input is None.")

    if not isinstance(image, np.ndarray):
        raise PreprocessingError(
            "INVALID_IMAGE",
            f"Expected numpy ndarray image, received '{type(image).__name__}'.",
        )

    if image.size == 0 or len(image.shape) < 2:
        raise PreprocessingError("EMPTY_IMAGE", "Image has zero dimensions or empty data.")

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise PreprocessingError("EMPTY_IMAGE", "Image height or width is zero.")

    max_dim = max_input_dimension if max_input_dimension is not None else MAX_IMAGE_INPUT_DIMENSION
    if h > max_dim or w > max_dim:
        raise PreprocessingError(
            "IMAGE_TOO_LARGE",
            f"Image dimensions ({w}x{h}) exceed maximum allowed dimension ({max_dim}px).",
        )

    return image


def normalize_color_format(image: np.ndarray) -> np.ndarray:
    """
    Standardizes image color format to 3-channel BGR (uint8).
    Preserves original without in-place mutation.
    """
    img = image.copy()

    # Convert floating point images to uint8 [0, 255]
    if img.dtype != np.uint8:
        if img.max() <= 1.0 and img.dtype in (np.float32, np.float64):
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert 1-channel Grayscale to 3-channel BGR
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Convert 4-channel BGRA to 3-channel BGR
    if len(img.shape) == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Ensure 3-channel BGR
    if len(img.shape) == 3 and img.shape[2] == 3:
        return img

    raise PreprocessingError(
        "INVALID_IMAGE",
        f"Unsupported image shape {img.shape}. Expected 2D or 3D color image.",
    )


def adjust_resolution(
    image: np.ndarray,
    min_width: int = MIN_IMAGE_WIDTH,
    max_width: int = MAX_IMAGE_WIDTH,
    upscale_factor: float = UPSCALE_FACTOR,
) -> Tuple[np.ndarray, bool, bool]:
    """
    Resizes image maintaining aspect ratio:
    - Downscales oversized images so BOTH width and height are <= max_width to bound processing time and memory.
    - Upscales small images (w < min_width and max(w, h) <= max_width) to improve character clarity for OCR.
    - Preserves aspect ratio and leaves normal resolution images unchanged (no upscaling for normal images).

    Returns:
        (resized_image, is_upscaled, is_downscaled)
    """
    h, w = image.shape[:2]
    is_upscaled = False
    is_downscaled = False

    max_dimension = max(w, h)

    # 1. Downscale oversized images (both width and height must be <= max_width)
    if max_dimension > max_width and max_dimension > 0:
        scale = max_width / float(max_dimension)
        target_w = max(1, int(round(w * scale)))
        target_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        is_downscaled = True
        return resized, is_upscaled, is_downscaled

    # 2. Upscale small images (only if width is below min_width and not already oversized)
    elif w < min_width and w > 0:
        scale = max(upscale_factor, min_width / float(w))
        target_w = int(min(max_width, round(w * scale)))
        target_h = max(1, int(round(h * (target_w / float(w)))))
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        is_upscaled = True
        return resized, is_upscaled, is_downscaled

    # 3. Normal resolution within bounds: leave unchanged (never upscale)
    return image.copy(), is_upscaled, is_downscaled


def estimate_skew_angle(
    gray: np.ndarray, max_angle: float = MAX_DESKEW_ANGLE
) -> float:
    """
    Estimates the document rotation angle using Otsu thresholding and minimum bounding rectangle.
    Returns angle in degrees [-max_angle, max_angle].
    """
    try:
        # Invert binary threshold to highlight text foreground
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return 0.0

        angle = cv2.minAreaRect(coords)[-1]
        # Format angle to [-45, 45] degrees
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle

        if abs(angle) <= max_angle:
            return float(round(angle, 2))
    except Exception:
        pass
    return 0.0


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an image around its center by the specified angle with border replication.
    """
    if abs(angle) < 0.5:
        return image.copy()

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def create_variant_grayscale(
    image: np.ndarray, deskew_angle: float = 0.0
) -> np.ndarray:
    """
    Variant 1 — Grayscale:
    Clean 8-bit grayscale image with optional rotation deskewing applied.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if abs(deskew_angle) >= 0.5:
        gray = rotate_image(gray, deskew_angle)

    return gray


def create_variant_contrast(grayscale_image: np.ndarray) -> np.ndarray:
    """
    Variant 2 — Contrast Enhanced & Sharpened:
    Bilateral filter denoising + CLAHE (Contrast Limited Adaptive Histogram Equalization)
    + unsharp masking. Outstanding for low-contrast scans and shadow-affected photographs.
    """
    # Edge-preserving light noise reduction
    denoised = cv2.bilateralFilter(grayscale_image, d=5, sigmaColor=50, sigmaSpace=50)

    # Localized adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Unsharp mask for character edge sharpening
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    return sharpened


def create_variant_threshold(grayscale_image: np.ndarray) -> np.ndarray:
    """
    Variant 3 — Adaptive / Otsu Binarization:
    Clean black-on-white thresholded mask for high-contrast character isolation.
    """
    denoised = cv2.medianBlur(grayscale_image, 3)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11
    )
    return adaptive


def preprocess_image(
    image: Any,
    enable_deskew: Optional[bool] = None,
    max_variants: Optional[int] = None,
) -> PreprocessedDocument:
    """
    Main preprocessing pipeline entrypoint.
    Accepts an OpenCV/NumPy image array, validates it, standardizes resolution/color,
    and produces 2–3 high-quality preprocessing variants for OCR and QR detection.

    Returns:
        PreprocessedDocument container retaining original image, normalized image,
        and preprocessing variants.
    """
    # Step 1: Validate input
    validated_img = validate_image_input(image)
    original_img = validated_img.copy()

    # Step 2: Normalize color format
    color_normalized = normalize_color_format(validated_img)

    # Step 3: Normalize resolution
    res_normalized, is_upscaled, is_downscaled = adjust_resolution(color_normalized)

    # Step 4: Deskew estimation
    deskew_enabled = enable_deskew if enable_deskew is not None else ENABLE_DESKEW
    orig_gray = cv2.cvtColor(res_normalized, cv2.COLOR_BGR2GRAY)

    skew_angle = 0.0
    if deskew_enabled:
        skew_angle = estimate_skew_angle(orig_gray, max_angle=MAX_DESKEW_ANGLE)

    # Step 5: Build Preprocessing Variants (2-3 variants)
    # Variant 1: Grayscale (with deskewing if applicable)
    var_gray = create_variant_grayscale(res_normalized, deskew_angle=skew_angle)

    # Variant 2: Contrast Enhanced (CLAHE + Sharpening) - Primary OCR recommendation
    var_contrast = create_variant_contrast(var_gray)

    # Variant 3: Adaptive Threshold Binarization
    var_threshold = create_variant_threshold(var_gray)

    variant_objects: list[PreprocessingVariant] = [
        PreprocessingVariant(
            name="contrast_enhanced",
            image=var_contrast,
            description="Grayscale with CLAHE contrast enhancement and unsharp mask sharpening (Optimal for OCR)",
            target="ocr",
        ),
        PreprocessingVariant(
            name="grayscale",
            image=var_gray,
            description="Normalized grayscale image with deskew correction",
            target="general",
        ),
        PreprocessingVariant(
            name="adaptive_threshold",
            image=var_threshold,
            description="Adaptive Gaussian binarization thresholding",
            target="qr",
        ),
    ]

    limit = max_variants if max_variants is not None else MAX_PREPROCESSING_VARIANTS
    selected_variants = variant_objects[:limit]

    metadata = {
        "original_shape": list(original_img.shape),
        "normalized_shape": list(res_normalized.shape),
        "is_upscaled": is_upscaled,
        "is_downscaled": is_downscaled,
        "skew_angle": skew_angle,
        "deskew_applied": abs(skew_angle) >= 0.5,
        "variant_count": len(selected_variants),
    }

    doc = PreprocessedDocument(
        success=True,
        original=original_img,
        normalized=res_normalized,
        variants=selected_variants,
        metadata=metadata,
    )

    # Optional developer debug export
    if DEBUG_SAVE_PREPROCESSED:
        save_debug_variants(doc)

    return doc


def get_ocr_variants(image: Any) -> list[Tuple[str, np.ndarray]]:
    """
    Convenience helper for OCR modules: returns list of (name, image) variants tailored for OCR.
    """
    preprocessed = preprocess_image(image)
    return preprocessed.get_ocr_images()


def get_qr_variants(image: Any) -> list[Tuple[str, np.ndarray]]:
    """
    Convenience helper for QR modules: returns list of (name, image) variants tailored for QR detection.
    """
    preprocessed = preprocess_image(image)
    return preprocessed.get_qr_images()


def save_debug_variants(
    doc: PreprocessedDocument,
    output_dir: Union[str, Path] = "debug",
    prefix: str = "preprocessed",
) -> list[str]:
    """
    Developer debug utility to inspect preprocessing variants on disk.
    Disabled in production and excluded from Git.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    try:
        orig_file = out_path / f"{prefix}_0_original.png"
        cv2.imwrite(str(orig_file), doc.original)
        saved_files.append(str(orig_file))

        norm_file = out_path / f"{prefix}_1_normalized.png"
        cv2.imwrite(str(norm_file), doc.normalized)
        saved_files.append(str(norm_file))

        for idx, var in enumerate(doc.variants, start=2):
            var_file = out_path / f"{prefix}_{idx}_{var.name}.png"
            cv2.imwrite(str(var_file), var.image)
            saved_files.append(str(var_file))
    except Exception:
        pass

    return saved_files
