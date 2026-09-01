# OCR Document Verification System

A modular document processing & OCR module for web-based identity verification systems.

## Features

- Safe file validation & PDF page rendering (300 DPI)
- Adaptive image normalization, deskewing & OpenCV preprocessing
- Multi-signal Aadhaar document detection (English/Hindi anchors, 12-digit patterns)
- Bilingual Tesseract OCR Engine (`eng` + `hin` with Unicode Devanagari preservation)
- Word-level bounding boxes and aggregate confidence scoring
- QR code detection & decoding
- Precision-aware DOB and fuzzy name cross-validation

---

## Tech Stack

- **Python 3.12**
- **OpenCV (`cv2`)**
- **Tesseract OCR & `pytesseract`**
- **PyMuPDF (`pymupdf`)**
- **Pillow (`PIL`)**
- **NumPy**

---

## Setup & System Requirements

### 1. Python Environment
```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Tesseract OCR System Dependency

Tesseract OCR is a **system-level dependency** that must be installed separately from Python pip packages.

#### Windows Installation:
1. Download the official installer from [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
2. During installation:
   - Make sure to select **Additional language data (download)** and check **Hindi (`hin`)**.
   - Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. Set the environment variable (if installed in a custom directory):
   ```powershell
   $env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-eng
```

#### macOS:
```bash
brew install tesseract tesseract-lang
```

---

## Checking Tesseract Installation & Available Languages

You can verify installed languages from your terminal:

```bash
tesseract --list-langs
```

Expected output:
```text
List of available languages (3):
eng
hin
osd
```

Or check programmatically via Python:
```python
from document_processor import check_ocr_health

health = check_ocr_health()
print(health)
# Output: {'tesseract_installed': True, 'has_english': True, 'has_hindi': True, 'status': 'HEALTHY', ...}
```

---

## Public API Usage

The pipeline exposes a clean, single entry point: `process_document(file_input)`.
Callers do not need to interact with internal preprocessing, OCR, or matcher modules.

### Basic Example

```python
from pathlib import Path
import io
from document_processor import process_document

# 1. Using a file path (string or Path object)
result = process_document("path/to/aadhaar_document.png")

# 2. Using raw bytes (e.g. from web upload / API payload)
with open("aadhaar.jpg", "rb") as f:
    file_bytes = f.read()
result = process_document(file_bytes)

# 3. Using an in-memory stream (io.BytesIO)
buffer = io.BytesIO(file_bytes)
result = process_document(buffer)
```

---

## Supported Input Types

| Input Type | Format / Types | Limits |
| :--- | :--- | :--- |
| `str` or `pathlib.Path` | Filepath to `.png`, `.jpg`, `.jpeg`, `.pdf` | Max 10.0 MB |
| `bytes` | Raw binary byte payload | Max 10.0 MB |
| `io.BytesIO` | In-memory binary buffer | Max 10.0 MB |

---

## Result Schema

Every successful execution returns a JSON-serializable Python dictionary:

```json
{
  "success": true,
  "document": {
    "type": "aadhaar",
    "confidence": 0.99
  },
  "ocr": {
    "language": "eng+hin",
    "confidence": 0.87,
    "fields": {
      "name": {
        "value": "sample person",
        "confidence": 0.70
      },
      "dob": {
        "year": 2000,
        "month": 1,
        "day": 25,
        "precision": "full",
        "confidence": 0.70
      },
      "gender": {
        "value": "FEMALE",
        "confidence": 0.87
      },
      "aadhaar_number": {
        "value": "XXXX XXXX XXXX",
        "confidence": 0.87
      },
      "address": {
        "value": "SAMPLE ADDRESS, SAMPLE CITY, SAMPLE STATE, PIN XXXXX",
        "confidence": 0.74
      }
    }
  },
  "qr": {
    "detected": false,
    "decoded": false,
    "verified": false,
    "format": "none"
  },
  "cross_validation": {
    "name": {
      "similarity": 1.0,
      "match": true
    },
    "dob": {
      "match": true,
      "comparison": "full"
    },
    "gender": {
      "match": true
    }
  },
  "warnings": []
}
```

*Note: `cross_validation` is present when a valid QR payload is decoded; otherwise, it is omitted and OCR field extraction proceeds uninterrupted.*

---

## Error Handling

Invalid, corrupted, or oversized inputs return structured error dictionaries with `success: False` rather than unhandled exceptions:

```json
{
  "success": false,
  "error": {
    "code": "FILE_TOO_LARGE",
    "message": "File size (11.00 MB) exceeds maximum allowed size (10.0 MB)."
  }
}
```

### Common Error Codes

| Error Code | Description |
| :--- | :--- |
| `FILE_NOT_FOUND` | The specified file path does not exist on disk. |
| `EMPTY_FILE` | The file or byte buffer has a length of 0 bytes. |
| `FILE_TOO_LARGE` | File exceeds the maximum file size limit (10 MB). |
| `IMAGE_TOO_LARGE` | Image dimensions exceed the maximum safety threshold (12,000 px). |
| `UNSUPPORTED_FILE_TYPE` | Extension is not `.pdf`, `.png`, `.jpg`, or `.jpeg`. |
| `CORRUPTED_OR_INVALID_FILE` | Magic bytes or image structure are invalid/corrupt. |
| `OCR_ENGINE_UNAVAILABLE` | Tesseract binary not found or system dependencies missing. |

---

## Privacy & Security Guarantees

- **Masked Aadhaar Numbers**: All extracted 12-digit Aadhaar numbers and VIDs are masked as `XXXX XXXX 1234` in both OCR fields and QR payloads.
- **Zero Raw PII Leaks**: Unmasked Aadhaar numbers are never emitted to logs, exception messages, debug outputs, or serialized JSON.

---

## Running Tests

Run the full automated test suite using `pytest`:

```bash
python -m pytest -v
```

