"""
Real integration test for existing Aadhaar field extraction layer.
Runs OCR on test_document.png and passes the output to the existing field extraction API.
"""

import sys
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows consoles for Hindi/Devanagari text
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from document_processor.config import mask_sensitive_number
from document_processor.file_handler import load_document
from document_processor.ocr.engine import run_ocr
from document_processor.ocr.fields import extract_aadhaar_fields, extract_all_fields_from_ocr
from document_processor.preprocessing import preprocess_image


def mask_text_uids(txt: str) -> str:
    """Helper to mask 12-digit sequences in multi-line text."""
    import re
    if not txt:
        return ""
    t = re.sub(r"\b(\d{4})[\s\-]+(\d{4})[\s\-]+(\d{4})\b", r"XXXX XXXX \3", txt)
    t = re.sub(r"\b(\d{8})(\d{4})\b", r"XXXX XXXX \2", t)
    return t


def main():
    image_path = Path("test_document1.png")
    if not image_path.exists():
        print(f"[-] Error: Image '{image_path}' not found.")
        return

    print("=" * 70)
    print(" 1. Loading & Preprocessing test_document1.png")
    print("=" * 70)
    load_res = load_document(str(image_path))
    if not load_res.get("success"):
        print(f"[-] Failed to load document: {load_res.get('error')}")
        return

    raw_image = load_res["images"][0]
    print(f"[+] Loaded document dimensions: {load_res['metadata']['width']}x{load_res['metadata']['height']} px")

    preprocessed = preprocess_image(raw_image)
    print(f"[+] Normalized resolution    : {preprocessed.metadata['normalized_shape']}")

    print("\n" + "=" * 70)
    print(" 2. Executing Bilingual OCR (eng+hin)")
    print("=" * 70)
    ocr_result = run_ocr(preprocessed, lang="eng+hin")
    print(f"[+] OCR Success     : {ocr_result.success}")
    print(f"[+] Language Used   : {ocr_result.language}")
    print(f"[+] Aggregate Conf  : {ocr_result.confidence * 100:.2f}%")
    print(f"[+] Word Count      : {ocr_result.word_count}")

    print("\n" + "=" * 70)
    print(" 3. Raw OCR Text for Debugging (Privacy-Safe Masked)")
    print("=" * 70)
    print(mask_text_uids(ocr_result.text) if ocr_result.text else "[No text detected]")

    print("\n" + "=" * 70)
    print(" 4. Field Extraction using Existing API (extract_aadhaar_fields)")
    print("=" * 70)
    # Call existing primary field extraction API
    extracted_result = extract_aadhaar_fields(ocr_result)
    fields = extracted_result.fields

    # Name
    name_field = fields.get("name")
    print("\n[Field: Name]")
    if name_field and name_field.value:
        print(f"  Value       : {name_field.value}")
        print(f"  English     : {name_field.metadata.get('english')}")
        print(f"  Hindi       : {name_field.metadata.get('hindi')}")
        print(f"  Confidence  : {name_field.confidence * 100:.2f}%")
        print(f"  Status      : {name_field.status}")
    else:
        print("  Value       : None (NOT_FOUND)")
        print(f"  Status      : {name_field.status if name_field else 'NOT_FOUND'}")

    # DOB
    dob_field = fields.get("dob")
    print("\n[Field: DOB]")
    if dob_field and dob_field.value:
        print(f"  Value (ISO) : {dob_field.value}")
        print(f"  Raw Text    : {dob_field.raw_value}")
        print(f"  Precision   : {dob_field.metadata.get('precision')}")
        print(f"  Confidence  : {dob_field.confidence * 100:.2f}%")
        print(f"  Status      : {dob_field.status}")
    else:
        print("  Value       : None (NOT_FOUND)")
        print(f"  Raw Text    : {dob_field.raw_value if dob_field else None}")
        print(f"  Status      : {dob_field.status if dob_field else 'NOT_FOUND'}")

    # YOB
    yob_field = fields.get("year_of_birth")
    print("\n[Field: YOB]")
    if yob_field and yob_field.value:
        print(f"  Value       : {yob_field.value}")
        print(f"  Raw Text    : {yob_field.raw_value}")
        print(f"  Confidence  : {yob_field.confidence * 100:.2f}%")
        print(f"  Status      : {yob_field.status}")
    else:
        print("  Value       : None (NOT_FOUND)")
        print(f"  Status      : {yob_field.status if yob_field else 'NOT_FOUND'}")

    # Gender
    gender_field = fields.get("gender")
    print("\n[Field: Gender]")
    if gender_field and gender_field.value:
        print(f"  Value       : {gender_field.value}")
        print(f"  Raw Text    : {gender_field.raw_value}")
        print(f"  Confidence  : {gender_field.confidence * 100:.2f}%")
        print(f"  Status      : {gender_field.status}")
    else:
        print("  Value       : None (NOT_FOUND)")
        print(f"  Status      : {gender_field.status if gender_field else 'NOT_FOUND'}")

    # Aadhaar Number
    aadhaar_field = fields.get("aadhaar_number")
    print("\n[Field: Aadhaar Number (MASKED)]")
    if aadhaar_field and aadhaar_field.value:
        masked_val = aadhaar_field.metadata.get("masked_value", mask_sensitive_number(aadhaar_field.value))
        print(f"  Masked Value: {masked_val}")
        print(f"  Format      : {aadhaar_field.metadata.get('format')}")
        print(f"  Confidence  : {aadhaar_field.confidence * 100:.2f}%")
        print(f"  Status      : {aadhaar_field.status}")
    else:
        print("  Value       : None (NOT_FOUND)")
        print(f"  Status      : {aadhaar_field.status if aadhaar_field else 'NOT_FOUND'}")

    # Address
    address_field = fields.get("address")
    print("\n[Field: Address]")
    if address_field and address_field.value:
        print(f"  Value       : {address_field.value}")
        print(f"  Confidence  : {address_field.confidence * 100:.2f}%")
        print(f"  Status      : {address_field.status}")
    else:
        print("  Value       : None (NOT_FOUND)")
        print(f"  Status      : {address_field.status if address_field else 'NOT_FOUND'}")

    # Warnings
    if extracted_result.warnings:
        print("\n[Extraction Warnings]")
        for w in extracted_result.warnings:
            print(f"  - {w}")

    print("\n" + "=" * 70)
    print(" 5. Legacy Extraction Schema Compatibility (extract_all_fields_from_ocr)")
    print("=" * 70)
    legacy_dict = extract_all_fields_from_ocr(
        raw_text=ocr_result.text,
        overall_ocr_confidence=ocr_result.confidence,
    )
    # Mask UID in legacy dict for safe logging
    if legacy_dict.get("aadhaar_number") and legacy_dict["aadhaar_number"].get("value"):
        legacy_dict["aadhaar_number"]["value"] = mask_sensitive_number(legacy_dict["aadhaar_number"]["value"])
    print(f"Legacy Extracted Dictionary: {legacy_dict}")


if __name__ == "__main__":
    main()
