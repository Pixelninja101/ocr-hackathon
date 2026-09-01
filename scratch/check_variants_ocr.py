import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pytesseract
from PIL import Image
from document_processor.file_handler import load_document
from document_processor.preprocessing import preprocess_image

doc = load_document("test_document1.png")
prep = preprocess_image(doc["images"][0])

print("Normalized shape:", prep.normalized.shape)
ocr_images = prep.get_ocr_images()
for name, img in ocr_images:
    text = pytesseract.image_to_string(img, lang="eng+hin")
    has_female = "FEMALE" in text or "महिला" in text
    has_male = "MALE" in text or "पुरुष" in text
    has_dob = "2007" in text or "22/09" in text
    print(f"Variant '{name}': has_female={has_female}, has_male={has_male}, has_dob={has_dob}")
    if has_female or has_male or has_dob:
        for l in text.splitlines():
            if any(k in l for k in ["FEMALE", "महिला", "MALE", "पुरुष", "2007", "22/09", "जन्म"]):
                print(f"   [{name}] Line: {l}")
