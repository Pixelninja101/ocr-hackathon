import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from difflib import SequenceMatcher
import re
from document_processor.file_handler import load_document
from document_processor.preprocessing import preprocess_image
from document_processor.ocr.engine import _execute_single_ocr, _convert_to_pil_image
from document_processor.ocr.fields import extract_aadhaar_fields

def clean_text_for_comparison(s: str) -> str:
    return re.sub(r"[^\w\u0900-\u097F]+", "", s.lower())

def is_line_duplicate(line: str, existing_lines: list[str], threshold: float = 0.65) -> bool:
    clean_target = clean_text_for_comparison(line)
    if not clean_target or len(clean_target) < 2:
        return True
    for el in existing_lines:
        clean_el = clean_text_for_comparison(el)
        if not clean_el:
            continue
        if clean_target in clean_el or clean_el in clean_target:
            return True
        sim = SequenceMatcher(None, clean_target, clean_el).ratio()
        if sim >= threshold:
            return True
    return False

doc = load_document("test_document.png")
prep = preprocess_image(doc["images"][0])
variants = prep.get_ocr_images()

ocr_results = []
for name, arr in variants:
    pil_img = _convert_to_pil_image(arr)
    text, conf, words, warnings = _execute_single_ocr(pil_img, lang="eng+hin")
    ocr_results.append({
        "name": name,
        "text": text,
        "conf": conf,
        "words": words,
        "warnings": warnings,
    })

primary = ocr_results[0]
print(f"Primary variant: {primary['name']} (words={len(primary['words'])}, conf={primary['conf']:.2f})")

merged_lines = [l for l in primary["text"].splitlines() if l.strip()]
merged_words = list(primary["words"])

for secondary in ocr_results[1:]:
    sec_line_word_map = {}
    for w in secondary["words"]:
        lid = (w.block_num, w.par_num, w.line_num)
        sec_line_word_map.setdefault(lid, []).append(w)

    for lid, line_words in sec_line_word_map.items():
        line_str = " ".join(w.text for w in line_words).strip()
        if not line_str:
            continue
        avg_line_conf = sum(w.confidence for w in line_words) / len(line_words)
        if avg_line_conf < 0.40:
            continue
        if not is_line_duplicate(line_str, merged_lines):
            merged_lines.append(line_str)
            merged_words.extend(line_words)
            print(f"  + Added supplemental line from '{secondary['name']}': '{line_str}' (conf={avg_line_conf:.2f})")

merged_text = "\n".join(merged_lines)
print("\n" + "=" * 70)
print(f"Total merged lines: {len(merged_lines)}, Total merged words: {len(merged_words)}")
print("=" * 70)

print("\n--- Field Extraction on Merged OCR Result (test_document.png) ---")
f_res = extract_aadhaar_fields(merged_text)
for k, v in f_res.fields.items():
    print(f"  {k:15s}: value={v.value}, status={v.status}, conf={v.confidence}")
