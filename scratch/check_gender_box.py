import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pytesseract
from document_processor.file_handler import load_document
from document_processor.preprocessing import preprocess_image

doc = load_document("test_document1.png")
img = doc["images"][0]
prep = preprocess_image(img)

data = pytesseract.image_to_data(prep.normalized, lang="eng+hin", output_type=pytesseract.Output.DICT)

for i, w in enumerate(data["text"]):
    if "FEMALE" in w or "महिला" in w:
        print(f"Index {i}: word='{w}', conf={data['conf'][i]}")
        print(f"  Box: x={data['left'][i]}, y={data['top'][i]}, w={data['width'][i]}, h={data['height'][i]}")
        print(f"  Line num: {data['line_num'][i]}, Block num: {data['block_num'][i]}, Par num: {data['par_num'][i]}")
        # Print all words in this block/paragraph
        block_words = [data["text"][j] for j in range(len(data["text"])) if data["block_num"][j] == data["block_num"][i] and data["text"][j].strip()]
        print("  Full block text:", " ".join(block_words))
