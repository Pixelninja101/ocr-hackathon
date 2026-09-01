"""
Dedicated Front-Side Aadhaar Field Extraction Integration Test.

Executes real end-to-end integration:
Image -> file_handler -> preprocessing -> Tesseract OCR (eng+hin) -> extract_aadhaar_fields() -> Evaluation

Tests front-side documents containing explicit DOB/YOB, Gender, Name, and Aadhaar numbers.
Uses synthetic test documents with clearly fictional data to maintain zero PII exposure.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure UTF-8 stdout encoding on Windows consoles for Hindi/Devanagari text
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from document_processor.config import mask_sensitive_number
from document_processor.file_handler import load_document
from document_processor.ocr.engine import run_ocr
from document_processor.ocr.fields import extract_aadhaar_fields
from document_processor.preprocessing import preprocess_image


def mask_text_uids(txt: str) -> str:
    """Helper to mask 12-digit sequences in multi-line text."""
    import re
    if not txt:
        return ""
    t = re.sub(r"\b(\d{4})[\s\-]+(\d{4})[\s\-]+(\d{4})\b", r"XXXX XXXX \3", txt)
    t = re.sub(r"\b(\d{8})(\d{4})\b", r"XXXX XXXX \2", t)
    return t


def generate_front_side_image(
    name_en: str = "TEST PERSON",
    name_hi: Optional[str] = None,
    dob: Optional[str] = "12/04/2002",
    yob: Optional[str] = None,
    gender: str = "Male",
    gender_hi: Optional[str] = "पुरुष",
    aadhaar_num: str = "1234 5678 9012",
    guardian_line: Optional[str] = None,
    footer_metadata_dates: Optional[list[str]] = None,
    image_width: int = 1200,
    image_height: int = 750,
) -> np.ndarray:
    """
    Generates a clean synthetic front-side Aadhaar card image fixture
    with crisp typography for reliable Tesseract OCR integration.
    """
    img = Image.new("RGB", (image_width, image_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Load system fonts if available
    try:
        font_hi = ImageFont.truetype("C:/Windows/Fonts/Nirmala.ttc", 28)
        font_hi_bold = ImageFont.truetype("C:/Windows/Fonts/Nirmala.ttc", 32)
    except Exception:
        font_hi = ImageFont.load_default()
        font_hi_bold = font_hi

    try:
        font_en = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
        font_en_bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 32)
        font_num = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 36)
    except Exception:
        font_en = ImageFont.load_default()
        font_en_bold = font_en
        font_num = font_en

    # Card outer border
    draw.rectangle([(15, 15), (image_width - 15, image_height - 15)], outline=(180, 180, 180), width=3)

    # Header section
    draw.text((45, 30), "भारत सरकार", fill=(0, 0, 0), font=font_hi_bold)
    draw.text((220, 30), "/ Government of India", fill=(0, 0, 0), font=font_en_bold)
    draw.text((45, 75), "भारतीय विशिष्ट पहचान प्राधिकरण / UIDAI", fill=(0, 0, 0), font=font_hi)

    y_pos = 140

    # Optional Guardian line (e.g. S/O Guardian Name)
    if guardian_line:
        draw.text((45, y_pos), guardian_line, fill=(0, 0, 0), font=font_en)
        y_pos += 45

    # Name section (Hindi & English)
    if name_hi:
        draw.text((45, y_pos), name_hi, fill=(0, 0, 0), font=font_hi)
        y_pos += 40
    draw.text((45, y_pos), name_en, fill=(0, 0, 0), font=font_en_bold)
    y_pos += 55

    # DOB / YOB section
    if dob:
        dob_str = f"जन्म तिथि / DOB: {dob}"
        draw.text((45, y_pos), dob_str, fill=(0, 0, 0), font=font_hi)
        y_pos += 55
    elif yob:
        yob_str = f"जन्म वर्ष / Year of Birth: {yob}"
        draw.text((45, y_pos), yob_str, fill=(0, 0, 0), font=font_hi)
        y_pos += 55

    # Gender section
    gender_str = f"{gender_hi} / {gender.upper()}" if gender_hi else gender.upper()
    draw.text((45, y_pos), gender_str, fill=(0, 0, 0), font=font_hi)
    y_pos += 70

    # Aadhaar Number section
    draw.text((45, y_pos), aadhaar_num, fill=(0, 0, 0), font=font_num)
    y_pos += 60

    # Optional footer dates (issue date, download date, details as on)
    if footer_metadata_dates:
        for f_date in footer_metadata_dates:
            draw.text((45, y_pos), f_date, fill=(80, 80, 80), font=font_en)
            y_pos += 35

    # Convert to OpenCV BGR
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_bgr


def run_single_front_side_test(
    test_id: str,
    description: str,
    img_bgr: np.ndarray,
    expected_values: dict[str, Any],
) -> dict[str, str]:
    print("=" * 80)
    print(f" FRONT-SIDE TEST {test_id}: {description}")
    print("=" * 80)

    # 1. Encode image to temporary PNG bytes to test file_handler integration
    success, img_bytes = cv2.imencode(".png", img_bgr)
    if not success:
        print("[-] Error: Failed to encode test image.")
        return {}

    load_res = load_document(img_bytes.tobytes())
    if not load_res.get("success"):
        print(f"[-] Load failed: {load_res.get('error')}")
        return {}

    raw_image = load_res["images"][0]
    print(f"[+] Loaded document dimensions: {load_res['metadata']['width']}x{load_res['metadata']['height']} px")

    # 2. Preprocessing
    preprocessed = preprocess_image(raw_image)
    print(f"[+] Normalized resolution    : {preprocessed.metadata['normalized_shape']}")

    # 3. Bilingual Tesseract OCR
    ocr_result = run_ocr(preprocessed, lang="eng+hin")
    print(f"[+] OCR Success              : {ocr_result.success}")
    print(f"[+] OCR Language Used        : {ocr_result.language}")
    print(f"[+] Aggregate Confidence     : {ocr_result.confidence * 100:.2f}%")
    print(f"[+] Detected Word Count      : {ocr_result.word_count}")

    print("\n--- Raw OCR Text (Privacy-Safe Masked) ---")
    masked_raw_ocr = mask_text_uids(ocr_result.text)
    for line in masked_raw_ocr.splitlines():
        if line.strip():
            print(f"  {line.strip()}")

    # 4. Field Extraction
    field_result = extract_aadhaar_fields(ocr_result)
    fields = field_result.fields

    name_f = fields.get("name")
    dob_f = fields.get("dob")
    yob_f = fields.get("year_of_birth")
    gender_f = fields.get("gender")
    aadhaar_f = fields.get("aadhaar_number")
    address_f = fields.get("address")

    # 5. Field Evaluation & Classification
    print("\n--- Extracted Fields & Classification ---")
    classifications: dict[str, str] = {}

    # Name
    exp_name = expected_values.get("name")
    act_name = name_f.value if name_f else None
    if exp_name is None:
        name_status = "FOUND_CORRECT" if act_name is None else "FOUND_INCORRECT"
    else:
        if act_name == exp_name:
            name_status = "FOUND_CORRECT"
        elif act_name is not None:
            name_status = "FOUND_INCORRECT"
        else:
            name_status = "NOT_FOUND"
    classifications["name"] = name_status
    print(f"  Name           : {act_name} -> {name_status} (Expected: {exp_name})")
    if name_f and name_f.metadata:
        print(f"    English: {name_f.metadata.get('english')}, Hindi: {name_f.metadata.get('hindi')}")

    # DOB
    exp_dob = expected_values.get("dob")
    act_dob = dob_f.value if dob_f else None
    if exp_dob is None:
        dob_status = "FOUND_CORRECT" if act_dob is None else "FOUND_INCORRECT"
    else:
        if act_dob == exp_dob:
            dob_status = "FOUND_CORRECT"
        elif act_dob is not None:
            dob_status = "FOUND_INCORRECT"
        else:
            dob_status = "NOT_FOUND"
    classifications["dob"] = dob_status
    print(f"  DOB            : {act_dob} -> {dob_status} (Expected: {exp_dob})")

    # YOB
    exp_yob = expected_values.get("yob")
    act_yob = yob_f.value if yob_f else None
    if exp_yob is None:
        yob_status = "FOUND_CORRECT" if act_yob is None else "FOUND_INCORRECT"
    else:
        if act_yob == exp_yob:
            yob_status = "FOUND_CORRECT"
        elif act_yob is not None:
            yob_status = "FOUND_INCORRECT"
        else:
            yob_status = "NOT_FOUND"
    classifications["yob"] = yob_status
    print(f"  YOB            : {act_yob} -> {yob_status} (Expected: {exp_yob})")

    # Gender
    exp_gender = expected_values.get("gender")
    act_gender = gender_f.value if gender_f else None
    if exp_gender is None:
        gender_status = "FOUND_CORRECT" if act_gender is None else "FOUND_INCORRECT"
    else:
        if act_gender == exp_gender:
            gender_status = "FOUND_CORRECT"
        elif act_gender is not None:
            gender_status = "FOUND_INCORRECT"
        else:
            gender_status = "NOT_FOUND"
    classifications["gender"] = gender_status
    print(f"  Gender         : {act_gender} -> {gender_status} (Expected: {exp_gender})")

    # Aadhaar Number
    exp_num_masked = expected_values.get("aadhaar_masked", "XXXX XXXX 9012")
    act_num_masked = aadhaar_f.metadata.get("masked_value", mask_sensitive_number(aadhaar_f.value)) if aadhaar_f and aadhaar_f.value else None
    if act_num_masked == exp_num_masked:
        num_status = "FOUND_CORRECT"
    elif act_num_masked is not None:
        num_status = "FOUND_INCORRECT"
    else:
        num_status = "NOT_FOUND"
    classifications["aadhaar_number"] = num_status
    print(f"  Aadhaar Number : {act_num_masked} -> {num_status} (Expected: {exp_num_masked})")

    # Address (expected None on front-side card)
    act_address = address_f.value if address_f else None
    addr_status = "FOUND_CORRECT" if act_address is None else "FOUND_INCORRECT"
    classifications["address"] = addr_status
    print(f"  Address        : {act_address} -> {addr_status} (Expected: None)")

    print()
    return classifications


def main():
    print("*" * 80)
    print(" FRONT-SIDE AADHAAR FIELD EXTRACTION INTEGRATION SUITE")
    print("*" * 80)
    print("Pipeline: Image -> file_handler -> preprocessing -> Tesseract OCR (eng+hin) -> extract_aadhaar_fields()")
    print()

    results: list[dict[str, Any]] = []

    # Scenario 1: Standard Bilingual Front-Side Card (English + Hindi Name, DOB, Gender, Number)
    img1 = generate_front_side_image(
        name_en="TEST PERSON",
        name_hi="राहुल कुमार",
        dob="12/04/2002",
        gender="Male",
        gender_hi="पुरुष",
        aadhaar_num="1234 5678 9012",
    )
    c1 = run_single_front_side_test(
        "1",
        "Standard Bilingual Front-Side (English + Hindi Name, Full DOB, Male Gender)",
        img1,
        expected_values={
            "name": "TEST PERSON",
            "dob": "2002-04-12",
            "yob": "2002",
            "gender": "MALE",
            "aadhaar_masked": "XXXX XXXX 9012",
        },
    )
    results.append({"scenario": "1. Bilingual Standard", "classifications": c1})

    # Scenario 2: Female Gender & Different DOB
    img2 = generate_front_side_image(
        name_en="PRIYA SHARMA",
        name_hi="प्रिया शर्मा",
        dob="15/08/1995",
        gender="Female",
        gender_hi="महिला",
        aadhaar_num="9876 5432 1098",
    )
    c2 = run_single_front_side_test(
        "2",
        "Female Gender & 15/08/1995 DOB",
        img2,
        expected_values={
            "name": "PRIYA SHARMA",
            "dob": "1995-08-15",
            "yob": "1995",
            "gender": "FEMALE",
            "aadhaar_masked": "XXXX XXXX 1098",
        },
    )
    results.append({"scenario": "2. Female Gender & DOB", "classifications": c2})

    # Scenario 3: Year of Birth Only (YOB) Front-Side Layout
    img3 = generate_front_side_image(
        name_en="AMIT VERMA",
        name_hi="अमित वर्मा",
        dob=None,
        yob="1988",
        gender="Male",
        gender_hi="पुरुष",
        aadhaar_num="5555 6666 7777",
    )
    c3 = run_single_front_side_test(
        "3",
        "Year of Birth Only (YOB: 1988) Layout",
        img3,
        expected_values={
            "name": "AMIT VERMA",
            "dob": None,
            "yob": "1988",
            "gender": "MALE",
            "aadhaar_masked": "XXXX XXXX 7777",
        },
    )
    results.append({"scenario": "3. YOB Only", "classifications": c3})

    # Scenario 4: Front-Side Card with Guardian Line (S/O: FATHER PERSON)
    img4 = generate_front_side_image(
        name_en="RAHUL KUMAR",
        guardian_line="S/O: RAJESH KUMAR",
        dob="12/04/2002",
        gender="Male",
        aadhaar_num="1234 5678 9012",
    )
    c4 = run_single_front_side_test(
        "4",
        "Guardian S/O Line Present (Must extract RAHUL KUMAR, not RAJESH KUMAR)",
        img4,
        expected_values={
            "name": "RAHUL KUMAR",
            "dob": "2002-04-12",
            "yob": "2002",
            "gender": "MALE",
            "aadhaar_masked": "XXXX XXXX 9012",
        },
    )
    results.append({"scenario": "4. Guardian Line S/O", "classifications": c4})

    # Scenario 5: Front-Side Card with Non-DOB Footer Metadata Dates
    img5 = generate_front_side_image(
        name_en="TEST PERSON",
        dob="12/04/2002",
        gender="Male",
        aadhaar_num="1234 5678 9012",
        footer_metadata_dates=[
            "Details as on: 30/12/2025",
            "Issue Date: 10/01/2023",
            "Date of Download: 01/08/2024",
        ],
    )
    c5 = run_single_front_side_test(
        "5",
        "Non-DOB Metadata Dates in Footer (Ensures 12/04/2002 extracted, non-DOB rejected)",
        img5,
        expected_values={
            "name": "TEST PERSON",
            "dob": "2002-04-12",
            "yob": "2002",
            "gender": "MALE",
            "aadhaar_masked": "XXXX XXXX 9012",
        },
    )
    results.append({"scenario": "5. Non-DOB Metadata Dates", "classifications": c5})

    print("=" * 80)
    print(" SUMMARY OF FRONT-SIDE INTEGRATION TESTS")
    print("=" * 80)
    all_correct = True
    for r in results:
        scen = r["scenario"]
        c = r["classifications"]
        statuses = list(c.values())
        scen_pass = all(s == "FOUND_CORRECT" for s in statuses)
        if not scen_pass:
            all_correct = False
        print(f"  {scen:35}: {'ALL FOUND_CORRECT' if scen_pass else 'ISSUES DETECTED'}")
        for f, st in c.items():
            if st != "FOUND_CORRECT":
                print(f"    - {f}: {st}")

    print(f"\n[+] Overall Integration Status: {'SUCCESS' if all_correct else 'FAILURE'}")


if __name__ == "__main__":
    main()
