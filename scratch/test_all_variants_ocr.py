import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from document_processor.file_handler import load_document
from document_processor.preprocessing import preprocess_image
from document_processor.ocr.engine import _execute_single_ocr, _convert_to_pil_image

doc = load_document("test_document1.png")
prep = preprocess_image(doc["images"][0])
variants = prep.get_ocr_images()

for name, arr in variants:
    pil_img = _convert_to_pil_image(arr)
    text, conf, words, warnings = _execute_single_ocr(pil_img, lang="eng+hin")
    print(f"Variant: {name:20s} | words={len(words):3d} | conf={conf:.2f}")
    has_female = "FEMALE" in text or "महिला" in text
    has_male = "MALE" in text or "पुरुष" in text
    has_dob = "2007" in text or "22/09" in text
    print(f"  -> has_female={has_female}, has_male={has_male}, has_dob={has_dob}")
    for l in text.splitlines():
        if any(k in l for k in ["FEMALE", "महिला", "MALE", "पुरुष", "22/09", "2007"]):
            print(f"     Match: {l}")
