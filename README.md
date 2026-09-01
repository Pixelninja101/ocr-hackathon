# Risk Engine

Deterministic Risk Scoring & Verification Engine for Identity Document Verification (Aadhaar).

---

## End-to-End Architecture

```text
       Document Upload (Image / PDF / Stream)
                         │
                         ▼
                    FastAPI REST API
           [POST /v1/verify | POST /verify]
     ├── Security Headers (nosniff, DENY, no-store)
     ├── In-Process Rate Limiter (HTTP 429 Retry-After)
     ├── Correlation ID Tracking (X-Request-ID)
     └── Safe Metadata Observability Logging
                         │
                         ▼
             document_processor (C:\icons\ocr)
    [OCR / Document Detection / Text Extraction / QR Processing]
                         │
                         ▼
               OCR Processing Result
                         │
                         ▼
         risk_engine.signals.extract_signals()
       ├── Document Classification Signals
       ├── OCR Quality & Field Presence Signals
       ├── QR Detection, Decode & Verification Signals
       ├── OCR ↔ QR Cross-Validation Comparison Signals
       └── Verhoeff Aadhaar Checksum Validation
                         │
                         ▼
                 Normalized Signals
                         │
                         ▼
          risk_engine.rules.evaluate_rules()
         [13 Explainable Risk Rules & Findings]
                         │
                         ▼
                 Risk Rule Findings
                         │
                         ▼
           risk_engine.scorer.assess_document()
       ├── Heuristic Point Summation
       ├── 0–100 Score Normalization
       ├── Risk Level Classification (LOW/MEDIUM/HIGH)
       ├── Critical Risk Overrides
       └── Decision Recommendation (PASS/REVIEW)
                         │
                         ▼
        integration.verification_service.verify_document()
       ┌───────────────────────────────────────┐
       │ Standardized Backend Response JSON    │
       │ Explicit Overrides & Check Summaries  │
       │ Zero Unmasked PII / Zero Raw Text     │
       └───────────────────────────────────────┘
                         │
                         ▼
             Web Backend & Dashboard UI
```

---

## Installation & Setup

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Install external OCR dependency in editable mode
python -m pip install -e C:\icons\ocr
```

---

## Running the REST API

Launch the production-style FastAPI server with Uvicorn:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`
- **OpenAPI JSON Spec**: `http://127.0.0.1:8000/openapi.json`
- **Liveness Health Check**: `http://127.0.0.1:8000/health`
- **Readiness Probe**: `http://127.0.0.1:8000/ready`

### API Endpoints

| Method | Route | Description | Security & Headers |
|:---:|---|---|---|
| `GET` | `/` | Service root status banner & version info | Security headers, `X-Request-ID` |
| `GET` | `/health` | Lightweight liveness probe (does not run OCR) | Security headers, `X-Request-ID` |
| `GET` | `/ready` | Readiness probe (validates configuration) | Security headers, `X-Request-ID` |
| `POST` | `/v1/verify` | Primary multipart document verification endpoint | Rate limiting, size limits, security headers |
| `POST` | `/verify` | Backwards-compatible alias for `/v1/verify` | Rate limiting, size limits, security headers |

### Example cURL Request

```bash
curl -X POST "http://127.0.0.1:8000/v1/verify" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -H "X-Request-ID: custom-trace-001" \
  -F "file=@sample_aadhaar.jpg"
```

---

## API Security & Operational Hardening

1. **Security HTTP Headers**:
   - `X-Content-Type-Options: nosniff`
   - `X-Frame-Options: DENY`
   - `Referrer-Policy: no-referrer`
   - `Cache-Control: no-store`
2. **In-Process Sliding Window Rate Limiting**:
   - Rejects traffic exceeding limits with HTTP `429 Too Many Requests`.
   - Returns `Retry-After: <seconds>` header and structured error response (`RATE_LIMIT_EXCEEDED`).
3. **Correlation ID & Observability**:
   - Generates or propagates `X-Request-ID` across all endpoints and responses.
   - Logs operational metadata (method, route, client IP, content-length, latency) with zero PII or document text.
4. **Defensive Upload Sanitization**:
   - Limits uploads to `MAX_UPLOAD_SIZE_BYTES` (default 10 MB) returning HTTP `413 FILE_TOO_LARGE`.
   - Rejects empty uploads returning HTTP `400 EMPTY_UPLOAD`.
   - Deep recursive sanitization guarantees continuous 12-digit numbers are masked in all output payloads.

---

## Centralized Configuration

Settings can be customized via environment variables:

| Variable | Default | Description | Valid Range / Bounds |
|---|:---:|---|:---:|
| `MAX_UPLOAD_SIZE_BYTES` | `10485760` (10 MB) | Maximum upload size accepted by API | $\ge 1024$ bytes |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests allowed per window | $\ge 1$ |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding rate limit window duration in seconds | $\ge 1$ |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | Valid log level |
| `ENVIRONMENT` | `development` | Environment name (`development`, `staging`, `production`, `test`) | Valid environment string |
| `ALLOWED_ORIGINS` | `""` | Comma-separated CORS allowed origins | Origin URLs |

---

## Running Automated Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
