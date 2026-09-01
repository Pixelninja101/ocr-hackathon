import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from document_processor.file_handler import load_document
from document_processor.preprocessing import preprocess_image
from document_processor.ocr.engine import _execute_single_ocr, _convert_to_pil_image, OCRWord
from document_processor.ocr.fields import extract_aadhaar_fields

doc = load_document("test_document1.png")
prep = preprocess_image(doc["images"][0])
variants = prep.get_ocr_images()

# Run OCR on all variants
variant_results = []
for name, arr in variants:
    pil_img = _convert_to_pil_image(arr)
    text, conf, words, warnings = _execute_single_ocr(pil_img, lang="eng+hin")
    variant_results.append((name, text, conf, words))

# Base is variant with highest score
variant_results.sort(key=lambda item: len(item[3]) * 2 + item[2] * 10, reverse=True)
base_name, base_text, base_conf, base_words = variant_results[0]
print(f"Base variant: {base_name} ({len(base_words)} words, conf={base_conf:.2f})")

# Merge non-overlapping high-confidence words from other variants
merged_words = list(base_words)

def compute_overlap_ratio(w1: OCRWord, w2: OCRWord) -> float:
    x_overlap = max(0, min(w1.x + w1.width, w2.x + w2.width) - max(w1.x, w2.x))
    y_overlap = max(0, min(w1.y + w1.height, w2.y + w2.height) - max(w1.y, w2.y))
    intersection = x_overlap * y_overlap
    if intersection <= 0:
        return 0.0
    min_area = min(w1.width * w1.height, w2.width * w2.height)
    if min_area <= 0:
        return 0.0
    return intersection / min_area

added_words = []
for var_name, text, conf, words in variant_results[1:]:
    for w2 in words:
        if w2.confidence < 0.50 or not w2.text.strip():
            continue
        # Check if overlaps with any word in merged_words
        overlaps = False
        for w1 in merged_words:
            if compute_overlap_ratio(w1, w2) > 0.30:
                overlaps = True
                break
        if not overlaps:
            merged_words.append(w2)
            added_words.append(w2)

print(f"Added {len(added_words)} missing words from other variants:")
for w in added_words:
    print(f"  + Added: '{w.text}' (conf={w.confidence:.2f}, box=({w.x},{w.y},{w.width},{w.height}))")

# Sort merged words top-to-bottom, left-to-right to reconstruct lines cleanly
# Cluster into lines based on y-coordinate proximity
merged_words.sort(key=lambda w: (w.y, w.x))

# Line clustering
lines = []
current_line = []
current_y = None
line_height_threshold = 15

for w in merged_words:
    if current_y is None:
        current_y = w.y
        current_line.append(w)
    elif abs(w.y - current_y) <= line_height_threshold:
        current_line.append(w)
        # update running average y
        current_y = int(sum(item.y for item in current_line) / len(current_line))
    else:
        # Sort current line words left to right
        current_line.sort(key=lambda item: item.x)
        lines.append(" ".join(item.text for item in current_line))
        current_line = [w]
        current_y = w.y

if current_line:
    current_line.sort(key=lambda item: item.x)
    lines.append(" ".join(item.text for item in current_line))

final_text = "\n".join(lines)
print("\n--- Extracted Fields from Merged OCR ---")
res = extract_aadhaar_fields(final_text)
for k, v in res.fields.items():
    print(f"  {k}: value={v.value}, status={v.status}, conf={v.confidence}")
