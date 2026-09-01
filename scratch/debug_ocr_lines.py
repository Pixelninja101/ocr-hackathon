import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from document_processor.file_handler import load_document
from document_processor.preprocessing import preprocess_image
from document_processor.ocr.engine import run_ocr

doc = load_document("test_document1.png")
prep = preprocess_image(doc["images"][0])
res = run_ocr(prep, lang="eng+hin")

for idx, line in enumerate(res.text.splitlines()):
    print(f"{idx:3d}: {line}")
