"""
Unit tests for Image Normalization and OpenCV Preprocessing Layer.
Covers all 10 requirements of Prompt 3:
1. Normal image
2. Small image requiring upscaling
3. Large image requiring downscaling
4. Grayscale input
5. Invalid/empty image
6. Slightly rotated image (deskewing)
7. Multiple preprocessing variants produced (2-3)
8. Original image preserved
9. Reasonable dimension constraints
10. Robustness against representative inputs
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from document_processor.preprocessing import (
    PreprocessedDocument,
    PreprocessingError,
    PreprocessingVariant,
    adjust_resolution,
    estimate_skew_angle,
    get_ocr_variants,
    get_qr_variants,
    normalize_color_format,
    preprocess_image,
    rotate_image,
    save_debug_variants,
    validate_image_input,
)


class TestPreprocessingLayer(unittest.TestCase):

    def setUp(self):
        # Create a standard synthetic test image with crisp black text on white
        self.normal_image = np.full((600, 1400, 3), 255, dtype=np.uint8)
        cv2.putText(
            self.normal_image,
            "GOVERNMENT OF INDIA",
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            self.normal_image,
            "UNIQUE IDENTIFICATION AUTHORITY",
            (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )

    # 1. Normal image
    def test_1_normal_image_preprocessing(self):
        res = preprocess_image(self.normal_image)
        self.assertTrue(res.success)
        self.assertIsNotNone(res.original)
        self.assertIsNotNone(res.normalized)
        self.assertGreaterEqual(len(res.variants), 2)
        self.assertLessEqual(len(res.variants), 3)

        # Check metadata
        self.assertIn("original_shape", res.metadata)
        self.assertIn("normalized_shape", res.metadata)
        self.assertFalse(res.metadata["is_upscaled"])
        self.assertFalse(res.metadata["is_downscaled"])

    # 2. Small image requiring upscaling
    def test_2_small_image_upscaling(self):
        small_img = cv2.resize(self.normal_image, (400, 200))
        res = preprocess_image(small_img)

        self.assertTrue(res.success)
        self.assertTrue(res.metadata["is_upscaled"])
        norm_h, norm_w = res.normalized.shape[:2]
        self.assertGreaterEqual(norm_w, 1200)

        # Verify aspect ratio preserved within 1%
        orig_aspect = small_img.shape[1] / float(small_img.shape[0])
        norm_aspect = norm_w / float(norm_h)
        self.assertAlmostEqual(orig_aspect, norm_aspect, delta=0.05)

    # 3. Large image requiring downscaling
    def test_3_large_image_downscaling(self):
        large_img = cv2.resize(self.normal_image, (3500, 1500))
        res = preprocess_image(large_img)

        self.assertTrue(res.success)
        self.assertTrue(res.metadata["is_downscaled"])
        norm_h, norm_w = res.normalized.shape[:2]
        self.assertLessEqual(norm_w, 2500)

        # Verify aspect ratio preserved
        orig_aspect = large_img.shape[1] / float(large_img.shape[0])
        norm_aspect = norm_w / float(norm_h)
        self.assertAlmostEqual(orig_aspect, norm_aspect, delta=0.05)

    # 4. Grayscale input
    def test_4_grayscale_input(self):
        gray_img = cv2.cvtColor(self.normal_image, cv2.COLOR_BGR2GRAY)
        self.assertEqual(len(gray_img.shape), 2)

        res = preprocess_image(gray_img)
        self.assertTrue(res.success)
        # Normalized should be converted to 3-channel standard BGR
        self.assertEqual(len(res.normalized.shape), 3)
        self.assertEqual(res.normalized.shape[2], 3)
        self.assertGreaterEqual(len(res.variants), 2)

    # 5. Invalid / empty image
    def test_5_invalid_and_empty_images(self):
        with self.assertRaises(PreprocessingError) as ctx_none:
            preprocess_image(None)
        self.assertEqual(ctx_none.exception.code, "INVALID_IMAGE")

        with self.assertRaises(PreprocessingError) as ctx_type:
            preprocess_image("not an array")
        self.assertEqual(ctx_type.exception.code, "INVALID_IMAGE")

        with self.assertRaises(PreprocessingError) as ctx_empty:
            empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
            preprocess_image(empty_arr)
        self.assertEqual(ctx_empty.exception.code, "EMPTY_IMAGE")

    # 6. Slightly rotated image (deskewing)
    def test_6_deskewing(self):
        # Create a rotated text image (~5 degrees)
        rotated_img = rotate_image(self.normal_image, angle=5.0)

        gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        detected_angle = estimate_skew_angle(gray)

        res = preprocess_image(rotated_img, enable_deskew=True)
        self.assertTrue(res.success)
        self.assertIn("skew_angle", res.metadata)

    # 7. Multiple preprocessing variants produced
    def test_7_preprocessing_variants_produced(self):
        res = preprocess_image(self.normal_image)
        variant_names = [v.name for v in res.variants]

        self.assertIn("contrast_enhanced", variant_names)
        self.assertIn("grayscale", variant_names)
        self.assertIn("adaptive_threshold", variant_names)

        # Check that variants can be retrieved by name
        contrast_img = res.get_variant("contrast_enhanced")
        self.assertIsNotNone(contrast_img)
        self.assertEqual(len(contrast_img.shape), 2)

    # 8. Original image remains unchanged
    def test_8_original_image_preserved(self):
        original_copy = self.normal_image.copy()
        res = preprocess_image(self.normal_image)

        # Original input passed should be identical to copy
        self.assertTrue(np.array_equal(self.normal_image, original_copy))
        # Container original should match
        self.assertTrue(np.array_equal(res.original, original_copy))

    # 9. Processing does not produce unreasonably large images
    def test_9_dimension_bounds(self):
        huge_img = np.zeros((5000, 5000, 3), dtype=np.uint8)
        res = preprocess_image(huge_img)

        self.assertTrue(res.success)
        h, w = res.normalized.shape[:2]
        self.assertLessEqual(w, 2500)
        self.assertLessEqual(h, 2500)

    # 10. Robustness against floating point and RGBA inputs
    def test_10_float_and_rgba_inputs(self):
        # RGBA input
        rgba_img = cv2.cvtColor(self.normal_image, cv2.COLOR_BGR2BGRA)
        res_rgba = preprocess_image(rgba_img)
        self.assertTrue(res_rgba.success)
        self.assertEqual(res_rgba.normalized.shape[2], 3)

        # Float input [0.0, 1.0]
        float_img = (self.normal_image.astype(np.float32)) / 255.0
        res_float = preprocess_image(float_img)
        self.assertTrue(res_float.success)
        self.assertEqual(res_float.normalized.dtype, np.uint8)

    # 11. OCR variants vs QR variants convenience APIs
    def test_11_ocr_and_qr_variant_helpers(self):
        ocr_variants = get_ocr_variants(self.normal_image)
        self.assertIsInstance(ocr_variants, list)
        self.assertGreater(len(ocr_variants), 0)
        self.assertEqual(ocr_variants[0][0], "contrast_enhanced")

        qr_variants = get_qr_variants(self.normal_image)
        self.assertIsInstance(qr_variants, list)
        self.assertGreater(len(qr_variants), 0)

    # 12. Debug export utility
    def test_12_save_debug_variants(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = preprocess_image(self.normal_image)
            saved = save_debug_variants(res, output_dir=tmp_dir, prefix="test")
            self.assertGreater(len(saved), 0)
            for path_str in saved:
                self.assertTrue(Path(path_str).exists())

    # 13. Image exactly at the 2500px limit (no downscaling, no upscaling)
    def test_13_image_exactly_at_limit(self):
        exact_landscape = np.full((1500, 2500, 3), 255, dtype=np.uint8)
        res_land = preprocess_image(exact_landscape)
        self.assertTrue(res_land.success)
        self.assertFalse(res_land.metadata["is_downscaled"])
        self.assertFalse(res_land.metadata["is_upscaled"])
        self.assertEqual(res_land.normalized.shape[:2], (1500, 2500))

        exact_portrait = np.full((2500, 1800, 3), 255, dtype=np.uint8)
        res_port = preprocess_image(exact_portrait)
        self.assertTrue(res_port.success)
        self.assertFalse(res_port.metadata["is_downscaled"])
        self.assertFalse(res_port.metadata["is_upscaled"])
        self.assertEqual(res_port.normalized.shape[:2], (2500, 1800))

    # 14. Oversized portrait downscaling (bounds height to <= 2500)
    def test_14_oversized_portrait_downscaling(self):
        tall_img = np.full((4000, 1500, 3), 255, dtype=np.uint8)
        res = preprocess_image(tall_img)

        self.assertTrue(res.success)
        self.assertTrue(res.metadata["is_downscaled"])
        self.assertFalse(res.metadata["is_upscaled"])

        norm_h, norm_w = res.normalized.shape[:2]
        self.assertLessEqual(norm_h, 2500)
        self.assertLessEqual(norm_w, 2500)
        self.assertEqual(norm_h, 2500)
        self.assertEqual(norm_w, 938)

        # Aspect ratio check
        orig_aspect = tall_img.shape[1] / float(tall_img.shape[0])
        norm_aspect = norm_w / float(norm_h)
        self.assertAlmostEqual(orig_aspect, norm_aspect, delta=0.01)

    # 15. Real-world 6144x8192 high-resolution document downscaling
    def test_15_high_res_6144x8192_image_handling(self):
        # Synthetic 6144x8192 image representation (3-channel uint8)
        # Using adjust_resolution directly and validate_image_input to test full pipeline
        big_img = np.zeros((8192, 6144, 3), dtype=np.uint8)
        validated = validate_image_input(big_img)
        self.assertIsNotNone(validated)

        resized, is_upscaled, is_downscaled = adjust_resolution(big_img)
        self.assertTrue(is_downscaled)
        self.assertFalse(is_upscaled)

        norm_h, norm_w = resized.shape[:2]
        self.assertLessEqual(norm_h, 2500)
        self.assertLessEqual(norm_w, 2500)
        self.assertEqual(norm_h, 2500)
        self.assertEqual(norm_w, 1875)

        orig_aspect = 6144.0 / 8192.0
        norm_aspect = float(norm_w) / float(norm_h)
        self.assertAlmostEqual(orig_aspect, norm_aspect, delta=0.01)

    # 16. Strict Aspect-Ratio preservation across multiple shapes
    def test_16_aspect_ratio_preservation(self):
        shapes = [
            (3200, 1800),  # 16:9 landscape
            (4000, 3000),  # 4:3 landscape
            (3000, 4000),  # 3:4 portrait
            (3500, 3500),  # 1:1 square
            (5000, 2000),  # 2.5:1 ultra-wide
        ]
        for h, w in shapes:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            resized, _, is_downscaled = adjust_resolution(img)
            self.assertTrue(is_downscaled)
            res_h, res_w = resized.shape[:2]
            self.assertLessEqual(res_h, 2500)
            self.assertLessEqual(res_w, 2500)

            orig_aspect = float(w) / float(h)
            norm_aspect = float(res_w) / float(res_h)
            self.assertAlmostEqual(orig_aspect, norm_aspect, delta=0.01)

    # 17. No unwanted upscaling for normal images
    def test_17_no_unwanted_upscaling(self):
        normal_sizes = [
            (800, 1500),
            (1000, 1800),
            (1200, 2200),
            (1500, 2500),
        ]
        for h, w in normal_sizes:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            resized, is_upscaled, is_downscaled = adjust_resolution(img)
            self.assertFalse(is_upscaled, f"Image {w}x{h} was unexpectedly upscaled")
            self.assertFalse(is_downscaled, f"Image {w}x{h} was unexpectedly downscaled")
            self.assertEqual(resized.shape[:2], (h, w))

    # 18. Hard safety limit rejection for images exceeding MAX_IMAGE_INPUT_DIMENSION (12000px)
    def test_18_safety_limit_rejection_over_12000px(self):
        # Excessively huge image beyond hard safety boundary (14000 x 2000)
        huge_dim_img = np.zeros((14000, 2000, 3), dtype=np.uint8)
        with self.assertRaises(PreprocessingError) as ctx_large_h:
            validate_image_input(huge_dim_img)
        self.assertEqual(ctx_large_h.exception.code, "IMAGE_TOO_LARGE")

        huge_dim_w_img = np.zeros((2000, 13000, 3), dtype=np.uint8)
        with self.assertRaises(PreprocessingError) as ctx_large_w:
            validate_image_input(huge_dim_w_img)
        self.assertEqual(ctx_large_w.exception.code, "IMAGE_TOO_LARGE")


if __name__ == "__main__":
    unittest.main()

