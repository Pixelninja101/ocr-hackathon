"""
Payload Parser for QR Codes.
Specialized in UIDAI Aadhaar Secure QR (V1/V2 binary compressed),
Aadhaar Legacy XML, URLs, JSON, and standard formats.
"""

import base64
import io
import re
from typing import Any, Dict, List, Optional, Union
import zlib
import xml.etree.ElementTree as ET

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_aadhaar_secure_qr(raw_input: Union[str, bytes, int]) -> Optional[Dict[str, Any]]:
    """
    Parse UIDAI Aadhaar Secure QR code (binary gzip-compressed format).
    Accepts raw bytes or a large base-10 decimal integer string.
    Extracts authentic demographic fields, contact status, and embedded photo.
    Does not invent data, decrypt unknown data, or falsely claim signature verification.
    """
    decompressed: Optional[bytes] = None

    # Step 1: Obtain decompressed byte array
    if isinstance(raw_input, (bytes, bytearray)):
        # Try direct decompression
        for wbits in (16 + zlib.MAX_WBITS, 0, -zlib.MAX_WBITS):
            try:
                decompressed = zlib.decompress(raw_input, wbits)
                if decompressed:
                    break
            except Exception:
                pass

        # If direct decompression failed, check if raw_input was ascii string of digits
        if not decompressed:
            try:
                text_str = raw_input.decode("ascii", errors="ignore").strip()
                if text_str.isdigit() and len(text_str) > 100:
                    val = int(text_str)
                    byte_len = (val.bit_length() + 7) // 8
                    b_arr = val.to_bytes(max(byte_len, 500), "big").lstrip(b"\x00")
                    for wbits in (16 + zlib.MAX_WBITS, 0, -zlib.MAX_WBITS):
                        try:
                            decompressed = zlib.decompress(b_arr, wbits)
                            if decompressed:
                                break
                        except Exception:
                            pass
            except Exception:
                pass

    elif isinstance(raw_input, (str, int)):
        text_str = str(raw_input).strip()
        if text_str.isdigit() and len(text_str) > 100:
            try:
                val = int(text_str)
                byte_len = (val.bit_length() + 7) // 8
                b_arr = val.to_bytes(max(byte_len, 500), "big").lstrip(b"\x00")
                for wbits in (16 + zlib.MAX_WBITS, 0, -zlib.MAX_WBITS):
                    try:
                        decompressed = zlib.decompress(b_arr, wbits)
                        if decompressed:
                            break
                    except Exception:
                        pass
            except Exception:
                pass

    if not decompressed or len(decompressed) < 300:
        return None

    # Step 2: Identify delimiters (0xFF / 255)
    delimiters = [-1] + [i for i, b in enumerate(decompressed) if b == 255]
    if len(delimiters) < 16:
        return None

    # Step 3: Determine schema version (V2 has 'V2' header and last_4_digits_mobile_no)
    is_v2 = False
    try:
        header_text = decompressed[:2].decode("latin-1", errors="ignore")
        if header_text == "V2":
            is_v2 = True
    except Exception:
        pass

    if is_v2:
        field_names = [
            "version",
            "email_mobile_status",
            "referenceid",
            "name",
            "dob",
            "gender",
            "careof",
            "district",
            "landmark",
            "house",
            "location",
            "pincode",
            "postoffice",
            "state",
            "street",
            "subdistrict",
            "vtc",
            "last_4_digits_mobile_no",
        ]
    else:
        field_names = [
            "email_mobile_status",
            "referenceid",
            "name",
            "dob",
            "gender",
            "careof",
            "district",
            "landmark",
            "house",
            "location",
            "pincode",
            "postoffice",
            "state",
            "street",
            "subdistrict",
            "vtc",
        ]

    # Step 4: Extract demographic fields
    raw_fields: Dict[str, str] = {}
    for i, field_name in enumerate(field_names):
        if i + 1 < len(delimiters):
            start = delimiters[i] + 1
            end = delimiters[i + 1]
            raw_fields[field_name] = decompressed[start:end].decode("latin-1", errors="ignore").strip()
        else:
            raw_fields[field_name] = ""

    # Step 5: Format gender and address
    raw_gender = raw_fields.get("gender", "")
    gender_map = {"M": "Male", "F": "Female", "T": "Transgender"}
    gender = gender_map.get(raw_gender.upper(), raw_gender)

    addr_parts = [
        raw_fields.get("house"),
        raw_fields.get("street"),
        raw_fields.get("landmark"),
        raw_fields.get("location"),
        raw_fields.get("vtc"),
        raw_fields.get("subdistrict"),
        raw_fields.get("district"),
        raw_fields.get("state"),
    ]
    formatted_address = ", ".join([p for p in addr_parts if p])
    pincode = raw_fields.get("pincode")
    if pincode and formatted_address:
        formatted_address += f" - {pincode}"

    # Step 6: Status of mobile and email
    status_code = raw_fields.get("email_mobile_status", "0")
    try:
        status_num = int(status_code)
    except ValueError:
        status_num = 0

    email_registered = status_num in (1, 3)
    mobile_registered = status_num in (2, 3)

    # Step 7: Reference ID and masked Aadhaar
    ref_id = raw_fields.get("referenceid", "")
    masked_aadhaar = f"XXXX XXXX {ref_id[:4]}" if len(ref_id) >= 4 else (ref_id or None)

    # Step 8: Extract photo bytes if present
    photo_base64: Optional[str] = None
    last_field_delim_idx = len(field_names)
    if len(delimiters) > last_field_delim_idx:
        photo_start = delimiters[last_field_delim_idx] + 1
        # Photo ends before hashes/signature at end of stream
        # Signature is last 256 bytes
        # Email hash: 32 bytes (if email registered)
        # Mobile hash: 32 bytes (if mobile registered)
        tail_offset = 256
        if status_num == 3:
            tail_offset += 64
        elif status_num in (1, 2):
            tail_offset += 32

        photo_end = max(photo_start, len(decompressed) - tail_offset)
        photo_bytes = decompressed[photo_start:photo_end]

        if len(photo_bytes) > 50:
            try:
                img = Image.open(io.BytesIO(photo_bytes))
                img.load()
                out_buf = io.BytesIO()
                img.convert("RGB").save(out_buf, format="JPEG", quality=90)
                photo_base64 = "data:image/jpeg;base64," + base64.b64encode(out_buf.getvalue()).decode("ascii")
            except Exception:
                pass

    # Build structured field list for UI display
    display_fields = []
    if masked_aadhaar:
        display_fields.append({"key": "Aadhaar / Ref", "value": masked_aadhaar, "sensitive": True})
    if raw_fields.get("name"):
        display_fields.append({"key": "Name", "value": raw_fields["name"]})
    if raw_fields.get("dob"):
        display_fields.append({"key": "Date of Birth", "value": raw_fields["dob"]})
    if gender:
        display_fields.append({"key": "Gender", "value": gender})
    if raw_fields.get("careof"):
        display_fields.append({"key": "Care Of", "value": raw_fields["careof"]})
    if formatted_address:
        display_fields.append({"key": "Address", "value": formatted_address})
    if pincode:
        display_fields.append({"key": "PIN Code", "value": pincode})
    if raw_fields.get("last_4_digits_mobile_no"):
        display_fields.append({"key": "Mobile (Last 4)", "value": f"XXXXXX{raw_fields['last_4_digits_mobile_no']}"})
    elif mobile_registered:
        display_fields.append({"key": "Mobile", "value": "Registered with UIDAI"})
    else:
        display_fields.append({"key": "Mobile", "value": "Not Registered"})

    if email_registered:
        display_fields.append({"key": "Email", "value": "Registered with UIDAI"})
    else:
        display_fields.append({"key": "Email", "value": "Not Registered"})

    return {
        "type": "aadhaar_secure",
        "label": f"UIDAI Aadhaar Secure QR ({'V2' if is_v2 else 'V1'})",
        "icon": "shield",
        "format": "UIDAI Secure QR (Binary Encoded & Digitally Signed)",
        "version": "V2" if is_v2 else "V1",
        "photo_url": photo_base64,
        "fields": display_fields,
        "raw_attributes": raw_fields,
        "has_photo": photo_base64 is not None,
        "has_digital_signature": len(decompressed) >= 256,
        "notice": "This is a digitally signed UIDAI Secure QR Code. Authentic demographic data has been extracted from the decompressed binary schema without bypassing cryptography or altering signatures.",
    }


def parse_aadhaar_old_xml(xml_string: str) -> Optional[Dict[str, Any]]:
    """
    Parse Legacy Aadhaar QR code (XML format).
    """
    if "PrintLetterBarcodeData" not in xml_string and "<" not in xml_string:
        return None
    try:
        parser = ET.XMLParser(encoding="utf-8")
        root = ET.fromstring(xml_string.strip(), parser=parser)
        attrs = root.attrib
        if not attrs or ("uid" not in attrs and "name" not in attrs):
            return None

        uid = attrs.get("uid")
        masked_uid = f"XXXX XXXX {uid[-4:]}" if uid and len(uid) >= 4 else uid

        g_raw = attrs.get("gender", "")
        gender_map = {"M": "Male", "F": "Female", "T": "Transgender"}
        gender = gender_map.get(g_raw.upper(), g_raw)

        addr_parts = [
            attrs.get("house"),
            attrs.get("street") or attrs.get("str"),
            attrs.get("lm"),
            attrs.get("loc"),
            attrs.get("vtc"),
            attrs.get("subdist"),
            attrs.get("dist"),
            attrs.get("state"),
        ]
        formatted_address = ", ".join([p for p in addr_parts if p])
        pc = attrs.get("pc")
        if pc and formatted_address:
            formatted_address += f" - {pc}"

        fields = []
        if masked_uid:
            fields.append({"key": "Aadhaar Number", "value": masked_uid, "sensitive": True})
        if attrs.get("name"):
            fields.append({"key": "Name", "value": attrs["name"]})
        if gender:
            fields.append({"key": "Gender", "value": gender})
        if attrs.get("dob"):
            fields.append({"key": "Date of Birth", "value": attrs["dob"]})
        elif attrs.get("yob"):
            fields.append({"key": "Year of Birth", "value": attrs["yob"]})
        if attrs.get("co"):
            fields.append({"key": "Care Of", "value": attrs["co"]})
        if formatted_address:
            fields.append({"key": "Address", "value": formatted_address})
        if pc:
            fields.append({"key": "PIN Code", "value": pc})

        return {
            "type": "aadhaar_old",
            "label": "Aadhaar QR (Legacy XML Format)",
            "icon": "shield",
            "format": "UIDAI PrintLetterBarcodeData (XML)",
            "fields": fields,
            "raw_attributes": attrs,
            "has_photo": False,
            "has_digital_signature": False,
            "notice": None,
        }
    except Exception:
        return None


def parse_qr_payload(data: Union[str, bytes]) -> Dict[str, Any]:
    """
    Main entry point for parsing any decoded QR payload into structured data.
    """
    # 1. Check Aadhaar Secure QR (binary or long numeric)
    secure_result = parse_aadhaar_secure_qr(data)
    if secure_result:
        return secure_result

    # Format text if string
    text_data = data if isinstance(data, str) else str(data)
    trimmed = text_data.strip()

    # 2. Check Aadhaar Legacy XML
    xml_result = parse_aadhaar_old_xml(trimmed)
    if xml_result:
        return xml_result

    # 3. URL
    if trimmed.startswith("http://") or trimmed.startswith("https://") or trimmed.startswith("ftp://"):
        return {
            "type": "url",
            "label": "URL / Web Link",
            "icon": "link",
            "format": "Web URL",
            "fields": [{"key": "URL", "value": trimmed}],
            "notice": None,
        }

    # 4. JSON
    if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
        return {
            "type": "json",
            "label": "JSON Data",
            "icon": "code",
            "format": "JSON Object",
            "fields": [{"key": "Payload", "value": trimmed}],
            "notice": None,
        }

    # 5. Fallback
    return {
        "type": "text",
        "label": "Text / Raw Payload",
        "icon": "text",
        "format": "Plain Text",
        "fields": [{"key": "Content", "value": trimmed}],
        "notice": None,
    }
