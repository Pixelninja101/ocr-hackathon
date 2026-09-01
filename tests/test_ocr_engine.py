"""
Unit tests for Bilingual Tesseract OCR Engine.
Covers all requirements from Prompt 5:
1. Basic English OCR
2. Hindi OCR (Unicode Devanagari preservation)
3. Combined 'eng+hin' OCR
4. Confidence calculation
5. Empty / blank image handling
6. Invalid image handling
7. Unicode preservation
8. Word-level bounding boxes
9. Preprocessing integration
10. Tesseract health check & diagnostics
11. Language fallback & unavailable language data
12. Timeout & resource safety
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from document_processor.ocr.engine import (
    OCREngineError,
    OCRResult,
    OCRWord,
    _convert_to_pil_image,
    _execute_single_ocr,
    check_ocr_health,
    get_available_tesseract_languages,
    is_tesseract_available,
    run_ocr,
)
from document_processor.preprocessing import preprocess_image


def create_synthetic_text_image(text_lines: list[str], width: int = 800, height: int = 300) -> np.ndarray:
    """Creates a high-contrast synthetic image containing the given text lines."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = 30
    for line in text_lines:
        draw.text((40, y), line, fill=(0, 0, 0))
        y += 50

    return np.array(img)[:, :, ::-1]  # RGB to OpenCV BGR


class TestOCREngine(unittest.TestCase):

    # 1. Basic English OCR (Mocked Tesseract Engine test)
    def test_1_basic_english_ocr(self):
        mock_data = {
            "text": ["", "AADHAAR", "NAME", "DATE", "OF", "BIRTH"],
            "conf": [-1, 95.0, 92.0, 88.0, 90.0, 94.0],
            "left": [0, 40, 150, 40, 100, 140],
            "top": [0, 30, 30, 80, 80, 80],
            "width": [0, 80, 50, 40, 30, 50],
            "height": [0, 25, 25, 25, 25, 25],
            "line_num": [0, 1, 1, 2, 2, 2],
            "block_num": [0, 1, 1, 1, 1, 1],
            "par_num": [0, 1, 1, 1, 1, 1],
            "word_num": [0, 1, 2, 1, 2, 3],
        }

        test_img = create_synthetic_text_image(["AADHAAR NAME", "DATE OF BIRTH"])

        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng", "hin"]), \
             patch("pytesseract.image_to_data", return_value=mock_data):

            res = run_ocr(test_img, lang="eng")

        self.assertTrue(res.success)
        self.assertEqual(res.language, "eng")
        self.assertIn("AADHAAR", res.text)
        self.assertIn("BIRTH", res.text)
        self.assertEqual(res.word_count, 5)
        self.assertEqual(res.line_count, 2)
        self.assertGreaterEqual(res.confidence, 0.90)

    # 2. Hindi OCR with Unicode Devanagari text
    def test_2_hindi_ocr_unicode_preservation(self):
        mock_hindi_data = {
            "text": ["", "आधार", "नाम", "जन्म", "तिथि", "पता"],
            "conf": [-1, 94.0, 91.0, 89.0, 92.0, 88.0],
            "left": [0, 40, 120, 40, 100, 40],
            "top": [0, 30, 30, 80, 80, 130],
            "width": [0, 60, 50, 50, 40, 40],
            "height": [0, 25, 25, 25, 25, 25],
            "line_num": [0, 1, 1, 2, 2, 3],
            "block_num": [0, 1, 1, 1, 1, 1],
            "par_num": [0, 1, 1, 1, 1, 1],
            "word_num": [0, 1, 2, 1, 2, 1],
        }

        test_img = create_synthetic_text_image(["आधार नाम", "जन्म तिथि", "पता"])

        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng", "hin"]), \
             patch("pytesseract.image_to_data", return_value=mock_hindi_data):

            res = run_ocr(test_img, lang="hin")

        self.assertTrue(res.success)
        self.assertEqual(res.language, "hin")
        self.assertIn("आधार", res.text)
        self.assertIn("जन्म तिथि", res.text)
        self.assertIn("पता", res.text)
        self.assertEqual(res.word_count, 5)

    # 3. Combined OCR (eng+hin)
    def test_3_combined_bilingual_ocr(self):
        mock_bilingual_data = {
            "text": ["", "AADHAAR", "आधार", "NAME", "नाम"],
            "conf": [-1, 96.0, 93.0, 95.0, 91.0],
            "left": [0, 40, 140, 40, 140],
            "top": [0, 30, 30, 80, 80],
            "width": [0, 80, 60, 60, 50],
            "height": [0, 25, 25, 25, 25],
            "line_num": [0, 1, 1, 2, 2],
            "block_num": [0, 1, 1, 1, 1],
            "par_num": [0, 1, 1, 1, 1],
            "word_num": [0, 1, 2, 1, 2],
        }

        test_img = create_synthetic_text_image(["AADHAAR आधार", "NAME नाम"])

        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng", "hin"]), \
             patch("pytesseract.image_to_data", return_value=mock_bilingual_data):

            res = run_ocr(test_img, lang="eng+hin")

        self.assertTrue(res.success)
        self.assertEqual(res.language, "eng+hin")
        self.assertIn("AADHAAR", res.text)
        self.assertIn("आधार", res.text)
        self.assertIn("NAME", res.text)
        self.assertIn("नाम", res.text)

    # 4. Confidence Calculation Logic (ignoring -1 layout blocks)
    def test_4_confidence_calculation(self):
        mock_data = {
            "text": ["", "WordOne", "", "WordTwo"],
            "conf": [-1, 80.0, -1, 100.0],
            "left": [0, 10, 0, 80],
            "top": [0, 10, 0, 10],
            "width": [0, 50, 0, 50],
            "height": [0, 20, 0, 20],
            "line_num": [0, 1, 0, 1],
            "block_num": [0, 1, 0, 1],
            "par_num": [0, 1, 0, 1],
            "word_num": [0, 1, 0, 2],
        }

        test_img = np.full((100, 200, 3), 255, dtype=np.uint8)

        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng"]), \
             patch("pytesseract.image_to_data", return_value=mock_data):

            res = run_ocr(test_img, lang="eng")

        # Average of 80.0 and 100.0 is 90.0% -> 0.90
        self.assertEqual(res.confidence, 0.90)
        self.assertEqual(res.word_count, 2)

    # 5. Empty / Blank image handling (OCR_NO_TEXT)
    def test_5_empty_blank_image(self):
        mock_empty_data = {
            "text": ["", ""],
            "conf": [-1, -1],
            "left": [0, 0],
            "top": [0, 0],
            "width": [0, 0],
            "height": [0, 0],
            "line_num": [0, 0],
            "block_num": [0, 0],
            "par_num": [0, 0],
            "word_num": [0, 0],
        }

        blank_img = np.full((200, 200, 3), 255, dtype=np.uint8)

        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng"]), \
             patch("pytesseract.image_to_data", return_value=mock_empty_data):

            res = run_ocr(blank_img, lang="eng")

        self.assertTrue(res.success)
        self.assertEqual(res.text, "")
        self.assertEqual(res.confidence, 0.0)
        self.assertEqual(res.word_count, 0)
        self.assertGreater(len(res.warnings), 0)

    # 6. Invalid image inputs
    def test_6_invalid_image_inputs(self):
        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng"]):

            with self.assertRaises(OCREngineError) as ctx_none:
                run_ocr(None)
            self.assertIn(ctx_none.exception.code, ["EMPTY_IMAGE", "INVALID_IMAGE"])

            empty_array = np.zeros((0, 0, 3), dtype=np.uint8)
            with self.assertRaises(OCREngineError) as ctx_empty:
                run_ocr(empty_array)
            self.assertIn(ctx_empty.exception.code, ["EMPTY_IMAGE", "INVALID_IMAGE"])

    # 7. Word-level bounding boxes and attributes
    def test_7_word_bounding_boxes(self):
        mock_data = {
            "text": ["", "Aadhaar"],
            "conf": [-1, 94.2],
            "left": [0, 100],
            "top": [0, 120],
            "width": [0, 80],
            "height": [0, 25],
            "line_num": [0, 2],
            "block_num": [0, 1],
            "par_num": [0, 1],
            "word_num": [0, 1],
        }

        test_img = np.full((200, 300, 3), 255, dtype=np.uint8)

        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng"]), \
             patch("pytesseract.image_to_data", return_value=mock_data):

            res = run_ocr(test_img, lang="eng")

        self.assertEqual(len(res.words), 1)
        word = res.words[0]
        self.assertIsInstance(word, OCRWord)
        self.assertEqual(word.text, "Aadhaar")
        self.assertEqual(word.x, 100)
        self.assertEqual(word.y, 120)
        self.assertEqual(word.width, 80)
        self.assertEqual(word.height, 25)
        self.assertEqual(word.line_num, 2)

    # 8. Preprocessing integration (accepts PreprocessedDocument)
    def test_8_preprocessing_integration(self):
        test_img = create_synthetic_text_image(["Aadhaar Document"])
        preprocessed = preprocess_image(test_img)

        mock_data = {
            "text": ["", "Aadhaar", "Document"],
            "conf": [-1, 92.0, 95.0],
            "left": [0, 40, 140],
            "top": [0, 30, 30],
            "width": [0, 80, 80],
            "height": [0, 25, 25],
            "line_num": [0, 1, 1],
            "block_num": [0, 1, 1],
            "par_num": [0, 1, 1],
            "word_num": [0, 1, 2],
        }

        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng"]), \
             patch("pytesseract.image_to_data", return_value=mock_data):

            res = run_ocr(preprocessed, lang="eng")

        self.assertTrue(res.success)
        self.assertEqual(res.word_count, 2)
        self.assertIn("variant_used", res.metadata)

    # 9. Diagnostic Health Check API
    def test_9_check_ocr_health(self):
        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("pytesseract.get_tesseract_version", return_value="5.3.3"), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng", "hin", "osd"]):

            health = check_ocr_health()

        self.assertTrue(health["tesseract_installed"])
        self.assertEqual(health["tesseract_version"], "5.3.3")
        self.assertTrue(health["has_english"])
        self.assertTrue(health["has_hindi"])
        self.assertEqual(health["status"], "HEALTHY")

    # 10. Language unavailability handling
    def test_10_language_unavailability_handling(self):
        test_img = np.full((100, 200, 3), 255, dtype=np.uint8)

        # Case A: Hindi requested but only English installed with allow_fallback=True
        mock_eng_data = {
            "text": ["", "EnglishText"],
            "conf": [-1, 90.0],
            "left": [0, 10], "top": [0, 10], "width": [0, 50], "height": [0, 20],
            "line_num": [0, 1], "block_num": [0, 1], "par_num": [0, 1], "word_num": [0, 1],
        }

        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng"]), \
             patch("pytesseract.image_to_data", return_value=mock_eng_data):

            res = run_ocr(test_img, lang="eng+hin", allow_fallback=True)

        self.assertTrue(res.success)
        self.assertEqual(res.language, "eng")
        self.assertTrue(any("hin" in w for w in res.warnings))

        # Case B: Hindi requested with allow_fallback=False -> raises OCREngineError
        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng"]):

            with self.assertRaises(OCREngineError) as ctx_err:
                run_ocr(test_img, lang="eng+hin", allow_fallback=False)
            self.assertEqual(ctx_err.exception.code, "LANGUAGE_DATA_UNAVAILABLE")

    # 12. Multi-variant fusion retains fine text (e.g. secondary variant detects missing token)
    def test_12_multi_variant_fusion_retains_fine_text(self):
        # Variant 1 misses FEMALE, Variant 2 detects FEMALE
        mock_var1_data = {
            "text": ["", "Government", "of", "India", "Name", "AMAVI", "TOMAR"],
            "conf": [-1, 90.0, 90.0, 90.0, 85.0, 85.0, 85.0],
            "left": [0, 10, 80, 110, 10, 50, 100],
            "top": [0, 10, 10, 10, 50, 50, 50],
            "width": [0, 60, 20, 40, 30, 40, 50],
            "height": [0, 20, 20, 20, 20, 20, 20],
            "line_num": [0, 1, 1, 1, 2, 2, 2],
            "block_num": [0, 1, 1, 1, 1, 1, 1],
            "par_num": [0, 1, 1, 1, 1, 1, 1],
            "word_num": [0, 1, 2, 3, 1, 2, 3],
        }

        mock_var2_data = {
            "text": ["", "Government", "of", "India", "FEMALE"],
            "conf": [-1, 85.0, 85.0, 85.0, 95.0],
            "left": [0, 10, 80, 110, 10],
            "top": [0, 10, 10, 10, 100],
            "width": [0, 60, 20, 40, 60],
            "height": [0, 20, 20, 20, 20],
            "line_num": [0, 1, 1, 1, 3],
            "block_num": [0, 1, 1, 1, 1],
            "par_num": [0, 1, 1, 1, 1],
            "word_num": [0, 1, 2, 3, 1],
        }

        img1 = np.full((150, 200, 3), 255, dtype=np.uint8)
        img2 = np.full((150, 200, 3), 255, dtype=np.uint8)
        candidates = [("contrast_enhanced", img1), ("grayscale", img2)]

        def side_effect(pil_img, *args, **kwargs):
            # Return var1 for first call, var2 for second call
            if not hasattr(side_effect, "call_count"):
                side_effect.call_count = 0
            side_effect.call_count += 1
            if side_effect.call_count == 1:
                return mock_var1_data
            return mock_var2_data

        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng", "hin"]), \
             patch("pytesseract.image_to_data", side_effect=side_effect):

            res = run_ocr(candidates, lang="eng+hin")

        self.assertTrue(res.success)
        self.assertIn("AMAVI TOMAR", res.text)
        self.assertIn("FEMALE", res.text)
        self.assertTrue(any(w.text == "FEMALE" for w in res.words))

    # 13. Multi-variant fusion deduplicates overlapping lines without text bloat
    def test_13_multi_variant_fusion_deduplicates_overlapping_lines(self):
        # Both variants have exact same header lines
        mock_data = {
            "text": ["", "Unique", "Identification", "Authority"],
            "conf": [-1, 92.0, 92.0, 92.0],
            "left": [0, 10, 60, 150],
            "top": [0, 10, 10, 10],
            "width": [0, 40, 80, 70],
            "height": [0, 20, 20, 20],
            "line_num": [0, 1, 1, 1],
            "block_num": [0, 1, 1, 1],
            "par_num": [0, 1, 1, 1],
            "word_num": [0, 1, 2, 3],
        }

        img1 = np.full((100, 300, 3), 255, dtype=np.uint8)
        img2 = np.full((100, 300, 3), 255, dtype=np.uint8)
        candidates = [("contrast_enhanced", img1), ("grayscale", img2)]

        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng"]), \
             patch("pytesseract.image_to_data", return_value=mock_data):

            res = run_ocr(candidates, lang="eng")

        self.assertTrue(res.success)
        # Should appear exactly once, not twice
        self.assertEqual(res.text.count("Unique Identification Authority"), 1)
        self.assertEqual(len(res.words), 3)

    # 14. Multi-variant fusion preserves bilingual Hindi tokens
    def test_14_multi_variant_fusion_preserves_bilingual_hindi_tokens(self):
        mock_var1_data = {
            "text": ["", "Government", "of", "India"],
            "conf": [-1, 90.0, 90.0, 90.0],
            "left": [0, 10, 80, 110],
            "top": [0, 10, 10, 10],
            "width": [0, 60, 20, 40],
            "height": [0, 20, 20, 20],
            "line_num": [0, 1, 1, 1],
            "block_num": [0, 1, 1, 1],
            "par_num": [0, 1, 1, 1],
            "word_num": [0, 1, 2, 3],
        }

        mock_var2_data = {
            "text": ["", "महिला/", "FEMALE"],
            "conf": [-1, 88.0, 92.0],
            "left": [0, 10, 60],
            "top": [0, 50, 50],
            "width": [0, 40, 50],
            "height": [0, 20, 20],
            "line_num": [0, 1, 1],
            "block_num": [0, 1, 1],
            "par_num": [0, 1, 1],
            "word_num": [0, 1, 2],
        }

        img1 = np.full((100, 200, 3), 255, dtype=np.uint8)
        img2 = np.full((100, 200, 3), 255, dtype=np.uint8)
        candidates = [("contrast_enhanced", img1), ("grayscale", img2)]

        def side_effect(pil_img, *args, **kwargs):
            if not hasattr(side_effect, "call_count"):
                side_effect.call_count = 0
            side_effect.call_count += 1
            if side_effect.call_count == 1:
                return mock_var1_data
            return mock_var2_data

        with patch("document_processor.ocr.engine.is_tesseract_available", return_value=True), \
             patch("document_processor.ocr.engine.get_available_tesseract_languages", return_value=["eng", "hin"]), \
             patch("pytesseract.image_to_data", side_effect=side_effect):

            res = run_ocr(candidates, lang="eng+hin")

        self.assertTrue(res.success)
        self.assertIn("महिला/", res.text)
        self.assertIn("FEMALE", res.text)


if __name__ == "__main__":
    unittest.main()

