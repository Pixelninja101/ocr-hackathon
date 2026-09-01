"""
Real QR Integration Test.
Validates QR Detection, Decoding, Payload Classification, Parsing, Normalization,
and OCR <-> QR Cross-Validation across documents using existing production APIs.
"""

import sys
from pathlib import Path
from typing import Union

# Ensure UTF-8 stdout encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from document_processor.config import mask_sensitive_number
from document_processor.file_handler import load_document
from document_processor.ocr.engine import run_ocr
from document_processor.ocr.fields import extract_aadhaar_fields
from document_processor.preprocessing import preprocess_image
from document_processor.processor import process_document
from document_processor.qr.decoder import (
    mask_payload_uid,
    parse_qr_payload,
    process_qr_code,
)
from document_processor.qr.detector import detect_qr_code
from document_processor.verification.matcher import (
    calculate_string_similarity,
    cross_validate_ocr_and_qr,
    match_dob,
    match_gender,
    match_names,
)
from tests.test_helpers import create_synthetic_aadhaar_image


def mask_text_uids(txt: str) -> str:
    """Helper to mask 12-digit sequences in multi-line text."""
    import re
    if not txt:
        return ""
    t = re.sub(r"\b(\d{4})[\s\-]+(\d{4})[\s\-]+(\d{4})\b", r"XXXX XXXX \3", txt)
    t = re.sub(r"\b(\d{8})(\d{4})\b", r"XXXX XXXX \2", t)
    return t


def run_single_qr_integration_test(doc_name: str, doc_input: Union[str, Path, np.ndarray]):
    print("=" * 80)
    print(f" INTEGRATION TEST: {doc_name}")
    print("=" * 80)

    if isinstance(doc_input, (str, Path)):
        load_res = load_document(doc_input)
        if not load_res.get("success"):
            print(f"[-] Load failed: {load_res.get('error')}")
            return
        doc_image = load_res["images"][0]
        pipe_input = doc_input
    else:
        doc_image = doc_input
        success, buf = cv2.imencode(".png", doc_image)
        pipe_input = buf.tobytes() if success else doc_image

    # 1. QR Detection
    print("\n--- [QR DETECTION] ---")
    has_qr_pattern, qr_points = detect_qr_code(doc_image)
    print(f"  Visual QR Pattern Detected : {has_qr_pattern}")
    print(f"  Decoder Engine             : OpenCV QRCodeDetector / Multi-pass")
    print(f"  QR Codes Detected Count    : {1 if has_qr_pattern else 0}")
    if qr_points is not None:
        print(f"  Bounding Points Shape      : {qr_points.shape}")

    # 2. QR Processing & Decoding
    print("\n--- [QR PAYLOAD & DECODING] ---")
    qr_result = process_qr_code(doc_image)
    detected = qr_result.get("detected", False)
    decoded = qr_result.get("decoded", False)
    payload_format = qr_result.get("format", "none")
    qr_fields = qr_result.get("fields")
    is_aadhaar_payload = bool(qr_fields and any(k in qr_fields for k in ["name", "dob", "gender", "aadhaar_number"]))

    print(f"  QR Detected                : {detected}")
    print(f"  QR Decoded                 : {decoded}")
    print(f"  Aadhaar Classification     : {'AADHAAR_QR' if is_aadhaar_payload else ('NON_AADHAAR_QR' if decoded else 'NO_PAYLOAD')}")
    print(f"  Payload Format             : {payload_format}")
    if qr_fields:
        print(f"  Parsed Field Names ONLY    : {list(qr_fields.keys())}")
    else:
        print(f"  Parsed Field Names ONLY    : None")

    # 3. QR Fields (Privacy-Safe Masked)
    print("\n--- [QR FIELDS (PRIVACY MASKED)] ---")
    if qr_fields:
        print(f"  Name           : {qr_fields.get('name', 'NOT_AVAILABLE')}")
        print(f"  DOB/YOB        : {qr_fields.get('dob', 'NOT_AVAILABLE')}")
        print(f"  Gender         : {qr_fields.get('gender', 'NOT_AVAILABLE')}")
        print(f"  Aadhaar Number : {mask_sensitive_number(qr_fields.get('aadhaar_number', '')) if qr_fields.get('aadhaar_number') else 'NOT_AVAILABLE'}")
        print(f"  Address        : {qr_fields.get('address', 'NOT_AVAILABLE')}")
    else:
        print("  [No QR fields extracted from document]")

    # 4. OCR Execution & Fields
    print("\n--- [OCR FIELDS] ---")
    preprocessed = preprocess_image(doc_image)
    ocr_result = run_ocr(preprocessed, lang="eng+hin")
    field_result = extract_aadhaar_fields(ocr_result)
    ocr_fields = field_result.fields

    name_f = ocr_fields.get("name")
    dob_f = ocr_fields.get("dob")
    yob_f = ocr_fields.get("year_of_birth")
    gender_f = ocr_fields.get("gender")
    aadhaar_f = ocr_fields.get("aadhaar_number")
    address_f = ocr_fields.get("address")

    print(f"  Name           : {name_f.value if name_f and name_f.value else 'NOT_FOUND'} (Status: {name_f.status if name_f else 'NOT_FOUND'})")
    print(f"  DOB            : {dob_f.value if dob_f and dob_f.value else 'NOT_FOUND'} (Status: {dob_f.status if dob_f else 'NOT_FOUND'})")
    print(f"  YOB            : {yob_f.value if yob_f and yob_f.value else 'NOT_FOUND'} (Status: {yob_f.status if yob_f else 'NOT_FOUND'})")
    print(f"  Gender         : {gender_f.value if gender_f and gender_f.value else 'NOT_FOUND'} (Status: {gender_f.status if gender_f else 'NOT_FOUND'})")
    if aadhaar_f and aadhaar_f.value:
        print(f"  Aadhaar Number : {mask_sensitive_number(aadhaar_f.value)} (Status: {aadhaar_f.status})")
    else:
        print(f"  Aadhaar Number : NOT_FOUND (Status: {aadhaar_f.status if aadhaar_f else 'NOT_FOUND'})")
    print(f"  Address        : {address_f.value[:80] + '...' if address_f and address_f.value and len(address_f.value) > 80 else (address_f.value if address_f and address_f.value else 'NOT_FOUND')}")

    # 5. Cross Validation
    print("\n--- [CROSS VALIDATION (OCR <-> QR)] ---")
    fields_to_compare = ["name", "dob", "gender", "aadhaar_number", "address"]

    for f_name in fields_to_compare:
        ocr_val = None
        ocr_status = "NOT_FOUND"
        if f_name == "name" and name_f and name_f.value:
            ocr_val = name_f.value
            ocr_status = name_f.status
        elif f_name == "dob" and dob_f and dob_f.value:
            ocr_val = dob_f.value
            ocr_status = dob_f.status
        elif f_name == "gender" and gender_f and gender_f.value:
            ocr_val = gender_f.value
            ocr_status = gender_f.status
        elif f_name == "aadhaar_number" and aadhaar_f and aadhaar_f.value:
            ocr_val = aadhaar_f.value
            ocr_status = aadhaar_f.status
        elif f_name == "address" and address_f and address_f.value:
            ocr_val = address_f.value
            ocr_status = address_f.status

        qr_val = qr_fields.get(f_name) if qr_fields else None
        qr_status = "FOUND" if qr_val else "NOT_AVAILABLE"

        print(f"\n  [Field: {f_name.upper()}]")
        masked_ocr_display = mask_sensitive_number(ocr_val) if f_name == "aadhaar_number" and ocr_val else ocr_val
        masked_qr_display = mask_sensitive_number(qr_val) if f_name == "aadhaar_number" and qr_val else qr_val
        print(f"    OCR Value  : {masked_ocr_display} (Status: {ocr_status})")
        print(f"    QR Value   : {masked_qr_display} (Status: {qr_status})")

        if not ocr_val or not qr_val:
            print(f"    Comparison : NOT_COMPARABLE (missing in {'OCR' if not ocr_val else 'QR'})")
        else:
            if f_name == "name":
                n_match = match_names(ocr_val, qr_val)
                if n_match:
                    match_str = "MATCH" if n_match["match"] else "MISMATCH"
                    print(f"    Comparison : {match_str} (Similarity: {n_match['similarity'] * 100:.1f}%)")
            elif f_name == "dob":
                d_match = match_dob(ocr_val, qr_val)
                if d_match:
                    match_str = "MATCH" if d_match["match"] else "MISMATCH"
                    print(f"    Comparison : {match_str} (Precision: {d_match.get('comparison', 'full')})")
            elif f_name == "gender":
                g_match = match_gender(ocr_val, qr_val)
                if g_match:
                    match_str = "MATCH" if g_match["match"] else "MISMATCH"
                    print(f"    Comparison : {match_str}")
            elif f_name == "aadhaar_number":
                clean_ocr_uid = "".join(ch for ch in str(ocr_val) if ch.isdigit())
                clean_qr_uid = "".join(ch for ch in str(qr_val) if ch.isdigit())
                uid_match = clean_ocr_uid == clean_qr_uid
                print(f"    Comparison : {'MATCH' if uid_match else 'MISMATCH'}")
            elif f_name == "address":
                sim = calculate_string_similarity(ocr_val.lower(), qr_val.lower())
                print(f"    Comparison : {'MATCH' if sim >= 0.60 else 'PARTIAL_MATCH'} (Similarity: {sim * 100:.1f}%)")

    # 6. Overall Pipeline Validation
    print("\n--- [PROCESS_DOCUMENT PIPELINE EXECUTION] ---")
    pipeline_res = process_document(pipe_input)
    print(f"  Pipeline Success           : {pipeline_res.get('success')}")
    print(f"  Document Classification    : {pipeline_res.get('document')}")
    print(f"  Cross-Validation Included  : {bool(pipeline_res.get('cross_validation'))}")
    if pipeline_res.get("cross_validation"):
        print(f"  Cross-Validation Details   : {pipeline_res.get('cross_validation')}")
    print()


def main():
    print("*" * 80)
    print(" AADHAAR QR PIPELINE INTEGRATION TEST SUITE")
    print("*" * 80)

    # Test 1: Real test_document.png
    if Path("test_document.png").exists():
        run_single_qr_integration_test("test_document.png (Real Document Upload)", Path("test_document.png"))

    # Test 2: Real test_document1.png
    if Path("test_document1.png").exists():
        run_single_qr_integration_test("test_document1.png (Real Document Upload)", Path("test_document1.png"))

    # Test 3: Synthetic Aadhaar Document with Legitimate UIDAI XML QR Code
    synth_img = create_synthetic_aadhaar_image(
        name="RAHUL KUMAR",
        dob="12/04/2002",
        gender="MALE",
        aadhaar_num="9876 5432 1098",
        include_qr=True,
    )
    run_single_qr_integration_test("Synthetic Aadhaar Card with Valid UIDAI XML QR Code", synth_img)


if __name__ == "__main__":
    main()
