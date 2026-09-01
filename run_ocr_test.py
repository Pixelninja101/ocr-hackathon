"""
Minimal executable test script for document_processor OCR engine.
Runs OCR on test_document.png using the project's native pipeline APIs.
"""

import re
import sys
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows consoles for Hindi/Devanagari text
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from document_processor.file_handler import load_document
from document_processor.ocr.engine import check_ocr_health, run_ocr
from document_processor.preprocessing import preprocess_image


def main():
    image_path = Path("test_document.png")
    if not image_path.exists():
        print(f"[-] Error: Image '{image_path}' not found.")
        return

    print("=" * 60)
    print(" 1. Tesseract OCR Environment Diagnostics")
    print("=" * 60)
    health = check_ocr_health()
    print(f"Tesseract Installed : {health.get('tesseract_installed')}")
    print(f"Tesseract Version   : {health.get('tesseract_version')}")
    print(f"Tesseract Executable: {health.get('tesseract_cmd')}")
    print(f"Available Languages : {health.get('available_languages')}")
    print(f"Status              : {health.get('status')}")

    if not health.get("tesseract_installed"):
        print("[-] Tesseract is not accessible. Please check installation.")
        return

    print("\n" + "=" * 60)
    print(" 2. Loading Document using file_handler API")
    print("=" * 60)
    load_res = load_document(str(image_path))
    if not load_res.get("success"):
        print(f"[-] Failed to load document: {load_res.get('error')}")
        return

    raw_image = load_res["images"][0]
    print(f"[+] Document loaded successfully: {load_res['metadata']['width']}x{load_res['metadata']['height']} px")

    print("\n" + "=" * 60)
    print(" 3. Preprocessing Image using preprocessing API")
    print("=" * 60)
    preprocessed = preprocess_image(raw_image)
    print(f"[+] Normalized Resolution: {preprocessed.metadata['normalized_shape']}")
    print(f"[+] Estimated Skew Angle : {preprocessed.metadata['skew_angle']}°")
    print(f"[+] Generated Variants   : {[v.name for v in preprocessed.variants]}")

    print("\n" + "=" * 60)
    print(" 4. Executing OCR using run_ocr API (Language: eng+hin)")
    print("=" * 60)
    # run_ocr automatically evaluates PreprocessedDocument variants to select the highest-accuracy output
    ocr_result = run_ocr(preprocessed, lang="eng+hin")

    print(f"OCR Success         : {ocr_result.success}")
    print(f"OCR Language Used   : {ocr_result.language}")
    print(f"Variant Selected    : {ocr_result.metadata.get('variant_used')}")
    print(f"Execution Time      : {ocr_result.metadata.get('execution_time_ms')} ms")

    print("\n" + "=" * 60)
    print(" 5. Raw Extracted Text")
    print("=" * 60)
    print(ocr_result.text if ocr_result.text else "[No text detected]")

    print("\n" + "=" * 60)
    print(" 6. Metrics & Language Detection")
    print("=" * 60)
    print(f"Aggregate Confidence: {ocr_result.confidence * 100:.2f}% (normalized: {ocr_result.confidence})")
    print(f"Detected Word Count : {ocr_result.word_count}")
    print(f"Detected Line Count : {ocr_result.line_count}")

    # Devanagari Unicode range check (\u0900 - \u097F)
    devanagari_matches = re.findall(r"[\u0900-\u097F]+", ocr_result.text)
    has_devanagari = len(devanagari_matches) > 0
    print(f"Devanagari/Hindi Detected: {has_devanagari}")
    if has_devanagari:
        print(f"Sample Hindi Words       : {devanagari_matches[:10]}")

    if ocr_result.warnings:
        print(f"Warnings                 : {ocr_result.warnings}")

    print("\n" + "=" * 60)
    print(" 7. Word-Level Bounding Boxes (First 15 Words Sample)")
    print("=" * 60)
    if ocr_result.words:
        print(f"{'Word / Token':<25} | {'Conf':<6} | {'Box (x, y, w, h)':<20} | {'Line#'}")
        print("-" * 65)
        for word in ocr_result.words[:15]:
            box_str = f"({word.x}, {word.y}, {word.width}, {word.height})"
            print(f"{word.text:<25} | {word.confidence * 100:5.1f}% | {box_str:<20} | L{word.line_num}")
        if len(ocr_result.words) > 15:
            print(f"... and {len(ocr_result.words) - 15} more words.")
    else:
        print("[-] No word bounding boxes available.")


if __name__ == "__main__":
    main()
