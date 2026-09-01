import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from document_processor.file_handler import load_document
from document_processor.preprocessing import preprocess_image
from document_processor.ocr.engine import _execute_single_ocr, _convert_to_pil_image
from document_processor.ocr.fields import extract_aadhaar_fields

doc = load_document("test_document1.png")
prep = preprocess_image(doc["images"][0])
variants = prep.get_ocr_images()

for name, arr in variants:
    pil_img = _convert_to_pil_image(arr)
    text, conf, words, warnings = _execute_single_ocr(pil_img, lang="eng+hin")
    print("=" * 70)
    print(f"Variant: {name} (Words: {len(words)}, Conf: {conf:.2f})")
    print("=" * 70)
    f_res = extract_aadhaar_fields(text)
    for k, v in f_res.fields.items():
        print(f"  {k:15s}: value={v.value}, status={v.status}, conf={v.confidence}")
