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

print(f"Total lines: {len(res.text.splitlines())}")
for idx, line in enumerate(res.text.splitlines()):
    if any(k in line.lower() for k in ["amavi", "tomar", "female", "male", "dob", "2007", "22/09", "gender", "लिंग", "जन्म"]):
        print(f"Line {idx:2d}: {line}")
