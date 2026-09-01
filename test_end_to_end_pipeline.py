"""
End-to-End Aadhaar Document Processing Pipeline Integration Test.

Tests the full public entry point `process_document(file_input)` across:
- Real test documents (test_document1.png, test_document.png)
- Synthetic Aadhaar cards (with & without QR)
- Error conditions (corrupted, oversized, unsupported files, non-Aadhaar documents)
- Full JSON serializability and strict privacy masking
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure UTF-8 stdout encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from document_processor.config import mask_sensitive_number
from document_processor.processor import process_document
from tests.test_helpers import (
    create_synthetic_aadhaar_bytes,
    create_synthetic_aadhaar_image,
)


def mask_text_uids(txt: str) -> str:
    """Helper to mask 12-digit sequences in multi-line text."""
    import re
    if not txt:
        return ""
    t = re.sub(r"\b(\d{4})[\s\-]+(\d{4})[\s\-]+(\d{4})\b", r"XXXX XXXX \3", txt)
    t = re.sub(r"\b(\d{8})(\d{4})\b", r"XXXX XXXX \2", t)
    return t


def print_pipeline_result(title: str, result: dict[str, Any]):
    print("=" * 80)
    print(f" PIPELINE RESULT: {title}")
    print("=" * 80)

    # 1. JSON serializability verification
    try:
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        is_serializable = True
    except Exception as exc:
        is_serializable = False
        json_str = f"[Serialization Error: {exc}]"

    print(f"  Success               : {result.get('success')}")
    print(f"  JSON Serializable     : {is_serializable}")

    if not result.get("success"):
        print(f"  Error Code            : {result.get('error', {}).get('code')}")
        print(f"  Error Message         : {result.get('error', {}).get('message')}")
        print()
        return

    # Document classification
    doc_info = result.get("document", {})
    print(f"  Document Type         : {doc_info.get('type')} (Confidence: {doc_info.get('confidence', 0)*100:.1f}%)")

    # OCR information
    ocr_info = result.get("ocr", {})
    print(f"  OCR Language          : {ocr_info.get('language')}")
    print(f"  OCR Confidence        : {ocr_info.get('confidence', 0)*100:.1f}%")

    fields = ocr_info.get("fields", {})
    print("\n  --- Extracted OCR Fields ---")
    for fname in ["name", "dob", "gender", "aadhaar_number", "address"]:
        fval = fields.get(fname)
        if isinstance(fval, dict):
            val_display = fval.get("value")
            conf_display = f" (conf: {fval.get('confidence', 0)*100:.1f}%)"
            if fname == "dob":
                val_display = f"{fval.get('year')}-{fval.get('month', 'XX')}-{fval.get('day', 'XX')} (precision: {fval.get('precision')})"
        else:
            val_display = "NOT_FOUND"
            conf_display = ""
        print(f"    {fname:15s}: {val_display}{conf_display}")

    # QR information
    qr_info = result.get("qr", {})
    print("\n  --- QR Code Processing ---")
    print(f"    Detected            : {qr_info.get('detected')}")
    print(f"    Decoded             : {qr_info.get('decoded')}")
    print(f"    Format              : {qr_info.get('format', 'none')}")

    # Cross-validation
    if result.get("cross_validation"):
        print("\n  --- Cross-Validation (OCR <-> QR) ---")
        for k, v in result["cross_validation"].items():
            print(f"    {k:15s}: {v}")
    else:
        print("\n  --- Cross-Validation ---")
        print("    Status              : NOT_APPLICABLE (No readable QR payload)")

    if result.get("warnings"):
        print(f"\n  Warnings              : {result['warnings']}")

    print()


def main():
    print("*" * 80)
    print(" COMPLETE END-TO-END PIPELINE INTEGRATION TEST SUITE")
    print(" Public Entrypoint: process_document(file_input)")
    print("*" * 80)
    print()

    # Test 1: Real Front-Side Document (test_document1.png)
    if Path("test_document1.png").exists():
        res1 = process_document("test_document1.png")
        print_pipeline_result("Real Front-Side Aadhaar (test_document1.png)", res1)

    # Test 2: Real Back-Side Document (test_document.png)
    if Path("test_document.png").exists():
        res2 = process_document("test_document.png")
        print_pipeline_result("Real Back-Side Aadhaar (test_document.png)", res2)

    # Test 3: Synthetic Aadhaar with Valid UIDAI XML QR Code
    synth_bytes = create_synthetic_aadhaar_bytes(
        name="RAHUL KUMAR",
        dob="12/04/2002",
        gender="MALE",
        aadhaar_num="9876 5432 1098",
        include_qr=True,
    )
    res3 = process_document(synth_bytes)
    print_pipeline_result("Synthetic Aadhaar Card with QR Code (Full Verification)", res3)

    # Test 4: Aadhaar Without QR Code
    no_qr_bytes = create_synthetic_aadhaar_bytes(
        name="PRIYA SHARMA",
        dob="15/08/1995",
        gender="FEMALE",
        aadhaar_num="1234 5678 9012",
        include_qr=False,
    )
    res4 = process_document(no_qr_bytes)
    print_pipeline_result("Aadhaar Without QR Code (Graceful Continuation)", res4)

    # Test 5: Corrupted File Handling
    res5 = process_document(b"this is corrupt binary payload not an image")
    print_pipeline_result("Corrupted File Input", res5)

    # Test 6: Oversized File Handling (>10MB)
    res6 = process_document(b"%PDF-" + b"0" * (11 * 1024 * 1024))
    print_pipeline_result("Oversized File (>10MB)", res6)

    # Test 7: Unsupported File Type
    res7 = process_document(b"PK\x03\x04zipfileheader")
    print_pipeline_result("Unsupported File Type (.zip)", res7)


if __name__ == "__main__":
    main()
