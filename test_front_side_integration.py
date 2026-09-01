"""
Real front-side Aadhaar integration test using existing production APIs:
1. file_handler (load_document)
2. preprocessing (preprocess_image)
3. bilingual OCR (run_ocr)
4. document detection (detect_document_type)
5. Aadhaar field extraction (extract_aadhaar_fields)
6. normalization
"""

import sys
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows consoles for Hindi/Devanagari text
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from document_processor.config import mask_sensitive_number
from document_processor.document_detector import detect_document_type
from document_processor.file_handler import load_document
from document_processor.ocr.engine import run_ocr
from document_processor.ocr.fields import extract_aadhaar_fields
from document_processor.preprocessing import preprocess_image
from document_processor.qr.decoder import process_qr_code


def main():
    image_path = Path("test_document1.png")
    if not image_path.exists():
        print(f"[-] Error: Image '{image_path}' not found.")
        return

    print("=" * 75)
    print(" 1. Loading & Preprocessing test_document1.png")
    print("=" * 75)
    load_res = load_document(str(image_path))
    if not load_res.get("success"):
        print(f"[-] Failed to load document: {load_res.get('error')}")
        return

    raw_image = load_res["images"][0]
    print(f"[+] Loaded dimensions   : {load_res['metadata']['width']}x{load_res['metadata']['height']} px")

    preprocessed = preprocess_image(raw_image)
    print(f"[+] Normalized shape    : {preprocessed.metadata['normalized_shape']}")
    print(f"[+] Estimated skew angle: {preprocessed.metadata['skew_angle']}°")
    print(f"[+] Variants produced   : {[v.name for v in preprocessed.variants]}")

    print("\n" + "=" * 75)
    print(" 2. QR Code Processing")
    print("=" * 75)
    qr_result = process_qr_code(raw_image)
    print(f"[+] QR Detected: {qr_result.get('detected')}")
    print(f"[+] QR Decoded : {qr_result.get('decoded')}")

    print("\n" + "=" * 75)
    print(" 3. Executing Bilingual OCR (eng+hin)")
    print("=" * 75)
    ocr_result = run_ocr(preprocessed, lang="eng+hin")
    print(f"[+] OCR Success         : {ocr_result.success}")
    print(f"[+] OCR Language Used   : {ocr_result.language}")
    print(f"[+] Aggregate Confidence: {ocr_result.confidence * 100:.2f}% (normalized: {ocr_result.confidence})")
    print(f"[+] Detected Word Count : {ocr_result.word_count}")
    print(f"[+] Detected Line Count : {ocr_result.line_count}")

    print("\n" + "=" * 75)
    print(" 4. Raw OCR Text (Privacy-Safe Masked)")
    print("=" * 75)
    # Mask any potential 12-digit / 4-4-4 number sequences in raw text output
    import re
    def mask_text_uids(txt: str) -> str:
        t = re.sub(r"\b(\d{4})[\s\-]+(\d{4})[\s\-]+(\d{4})\b", r"XXXX XXXX \3", txt)
        t = re.sub(r"\b(\d{8})(\d{4})\b", r"XXXX XXXX \2", t)
        return t

    masked_raw_text = mask_text_uids(ocr_result.text) if ocr_result.text else "[No text detected]"
    print(masked_raw_text)

    print("\n" + "=" * 75)
    print(" 5. Document Type Detection")
    print("=" * 75)
    doc_detection = detect_document_type(
        image=raw_image,
        ocr_text=ocr_result.text,
        qr_detected=qr_result.get("detected", False),
        qr_is_aadhaar=bool(qr_result.get("fields")),
    )
    print(f"  document_type : {doc_detection.document_type}")
    print(f"  status        : {doc_detection.status}")
    print(f"  confidence    : {doc_detection.confidence * 100:.2f}% (normalized: {doc_detection.confidence})")
    print(f"  signals       : {doc_detection.signals_detected}")
    print(f"  evidence items: {len(doc_detection.evidence)} evidence items evaluated")

    print("\n" + "=" * 75)
    print(" 6. Aadhaar Field Extraction (extract_aadhaar_fields)")
    print("=" * 75)
    field_result = extract_aadhaar_fields(ocr_result)
    fields = field_result.fields

    # Name
    name_f = fields.get("name")
    print("\n[Field: Name]")
    print(f"  Value       : {name_f.value if name_f else None}")
    print(f"  Status      : {name_f.status if name_f else 'NOT_FOUND'}")
    print(f"  Confidence  : {name_f.confidence * 100:.2f}%" if name_f and name_f.value else "  Confidence  : 0.00%")
    print(f"  Raw Text    : {name_f.raw_value if name_f else None}")
    if name_f and name_f.metadata:
        print(f"  English     : {name_f.metadata.get('english')}")
        print(f"  Hindi       : {name_f.metadata.get('hindi')}")

    # DOB
    dob_f = fields.get("dob")
    print("\n[Field: DOB]")
    print(f"  Value (ISO) : {dob_f.value if dob_f else None}")
    print(f"  Status      : {dob_f.status if dob_f else 'NOT_FOUND'}")
    print(f"  Confidence  : {dob_f.confidence * 100:.2f}%" if dob_f and dob_f.value else "  Confidence  : 0.00%")
    print(f"  Raw Text    : {dob_f.raw_value if dob_f else None}")
    if dob_f and dob_f.metadata:
        print(f"  Precision   : {dob_f.metadata.get('precision')}")

    # YOB
    yob_f = fields.get("year_of_birth")
    print("\n[Field: YOB]")
    print(f"  Value       : {yob_f.value if yob_f else None}")
    print(f"  Status      : {yob_f.status if yob_f else 'NOT_FOUND'}")
    print(f"  Confidence  : {yob_f.confidence * 100:.2f}%" if yob_f and yob_f.value else "  Confidence  : 0.00%")
    print(f"  Raw Text    : {yob_f.raw_value if yob_f else None}")

    # Gender
    gender_f = fields.get("gender")
    print("\n[Field: Gender]")
    print(f"  Value       : {gender_f.value if gender_f else None}")
    print(f"  Status      : {gender_f.status if gender_f else 'NOT_FOUND'}")
    print(f"  Confidence  : {gender_f.confidence * 100:.2f}%" if gender_f and gender_f.value else "  Confidence  : 0.00%")
    print(f"  Raw Text    : {gender_f.raw_value if gender_f else None}")

    # Aadhaar Number (MASKED)
    aadhaar_f = fields.get("aadhaar_number")
    print("\n[Field: Aadhaar Number (MASKED)]")
    if aadhaar_f and aadhaar_f.value:
        masked_val = aadhaar_f.metadata.get("masked_value", mask_sensitive_number(aadhaar_f.value))
        print(f"  Masked Value: {masked_val}")
        print(f"  Status      : {aadhaar_f.status}")
        print(f"  Confidence  : {aadhaar_f.confidence * 100:.2f}%")
        print(f"  Raw Text    : {mask_sensitive_number(aadhaar_f.raw_value)}")
    else:
        print("  Value       : None (NOT_FOUND)")
        print(f"  Status      : {aadhaar_f.status if aadhaar_f else 'NOT_FOUND'}")

    # Address
    address_f = fields.get("address")
    print("\n[Field: Address]")
    print(f"  Value       : {address_f.value if address_f else None}")
    print(f"  Status      : {address_f.status if address_f else 'NOT_FOUND'}")
    print(f"  Confidence  : {address_f.confidence * 100:.2f}%" if address_f and address_f.value else "  Confidence  : 0.00%")

    if field_result.warnings:
        print("\n[Extraction Warnings]")
        for w in field_result.warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
