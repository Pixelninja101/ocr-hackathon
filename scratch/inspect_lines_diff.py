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
    print("=" * 60)
    print(f"VARIANT: {name}")
    print("=" * 60)
    for idx, l in enumerate(text.splitlines()[:25]):
        print(f"  {idx:2d}: {l}")
