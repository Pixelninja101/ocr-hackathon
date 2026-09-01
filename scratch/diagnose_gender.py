import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pytesseract
from PIL import Image

from document_processor.file_handler import load_document
from document_processor.ocr.engine import run_ocr
from document_processor.preprocessing import preprocess_image

doc = load_document("test_document1.png")
img = doc["images"][0]
print(f"Original image shape: {img.shape}")

prep = preprocess_image(img)
print(f"Preprocessed image shape: {prep.normalized.shape}")

# Full OCR
ocr_res = run_ocr(prep, lang="eng+hin")
print("Full OCR text:")
print("=" * 60)
print(ocr_res.text)
print("=" * 60)

# Check all words and find any potential gender clues
data = pytesseract.image_to_data(prep.normalized, lang="eng+hin", output_type=pytesseract.Output.DICT)

print("\nSearching for gender terms in words:")
found_any = False
for i, w in enumerate(data["text"]):
    wl = w.strip().lower()
    if not wl:
        continue
    if any(g in wl for g in ["male", "fem", "पुरुष", "महिल", "स्त्री", "लिंग", "gend", "sex", "अन्य"]):
        found_any = True
        surrounding = [data["text"][j] for j in range(max(0, i-5), min(len(data["text"]), i+6)) if data["text"][j].strip()]
        print(f"Match: word='{w}' (conf={data['conf'][i]}) | Context: {' '.join(surrounding)}")

if not found_any:
    print("NO gender terms found anywhere in OCR word list!")

# Let's check single letter tokens 'M' or 'F'
print("\nChecking single-character tokens 'M' or 'F':")
for i, w in enumerate(data["text"]):
    wl = w.strip()
    if wl in ["M", "F", "m", "f", "T", "t"]:
        surrounding = [data["text"][j] for j in range(max(0, i-5), min(len(data["text"]), i+6)) if data["text"][j].strip()]
        print(f"Token '{wl}' (conf={data['conf'][i]}): Context: {' '.join(surrounding)}")
