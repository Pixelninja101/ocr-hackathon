"""
End-to-End Pipeline Runner for Document Processing & Risk Assessment.
Usage:
    python demo/run_pipeline.py path/to/document.jpg
    python demo/run_pipeline.py path/to/document.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure risk_engine package root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from document_processor import process_document
from risk_engine import assess_document


def format_console_output(assessment: Dict[str, Any]) -> str:
    """
    Formats the complete assessment result into a clean, human-readable,
    PII-safe console presentation.
    """
    lines: list[str] = []
    lines.append("========================================")
    lines.append("DOCUMENT VERIFICATION")
    lines.append("========================================")
    lines.append("")

    signals = assessment.get("signals", {})
    doc_sig = signals.get("document", {})
    ocr_sig = signals.get("ocr", {})
    qr_sig = signals.get("qr", {})
    risk_info = assessment.get("risk", {})
    findings = assessment.get("findings", [])
    success = assessment.get("success", False)

    if not success:
        err = assessment.get("error", {})
        err_code = err.get("code", "UNKNOWN_ERROR")
        err_msg = err.get("message", "Document processing failed.")
        lines.append(f"Status        : FAILED")
        lines.append(f"Error Code    : {err_code}")
        lines.append(f"Error Message : {err_msg}")
        lines.append("")
        lines.append("----------------------------------------")
        lines.append("RISK ASSESSMENT")
        lines.append("----------------------------------------")
        lines.append("")
        lines.append(f"Risk Score    : N/A")
        lines.append(f"Risk Level    : {risk_info.get('level', 'UNKNOWN')}")
        lines.append(f"Decision      : {risk_info.get('decision', 'REVIEW')}")
        lines.append("")
        lines.append("----------------------------------------")
        lines.append("STATUS")
        lines.append("----------------------------------------")
        lines.append("")
        lines.append(risk_info.get("summary", "Document processing failed upstream."))
        lines.append("")
        return "\n".join(lines)

    # Document Classification & Confidence
    doc_type_raw = doc_sig.get("type", "unknown")
    doc_type = "Aadhaar" if doc_type_raw == "aadhaar" else doc_type_raw.replace("_", " ").title()
    doc_conf = int(round(doc_sig.get("confidence", 0.0) * 100))
    lines.append(f"Document Type : {doc_type}")
    lines.append(f"Detection     : {doc_conf}%")
    lines.append("")

    # OCR Quality Confidence
    ocr_conf_val = ocr_sig.get("confidence")
    ocr_conf = f"{int(round(ocr_conf_val * 100))}%" if ocr_conf_val is not None else "N/A"
    lines.append(f"OCR Confidence: {ocr_conf}")
    lines.append("")

    # QR Detection & Verification Status
    qr_detected = "YES" if qr_sig.get("detected") else "NO"
    qr_decoded = "YES" if qr_sig.get("decoded") else "NO"
    qr_verified = "YES" if qr_sig.get("verified") else "NO"
    status_raw = qr_sig.get("verification_status", "QR_NOT_DETECTED")

    if status_raw == "QR_NOT_DETECTED":
        status_desc = "QR not detected"
    elif status_raw == "QR_DETECTED_NOT_DECODED":
        status_desc = "Detected, decode failed"
    elif status_raw == "QR_DECODED_VERIFICATION_UNAVAILABLE":
        status_desc = "Verification unavailable"
    elif status_raw == "QR_VERIFIED":
        status_desc = "Cryptographically verified"
    elif status_raw == "QR_VERIFICATION_FAILED":
        status_desc = "Verification failed"
    else:
        status_desc = status_raw.replace("_", " ").title()

    lines.append("QR:")
    lines.append(f"  Detected    : {qr_detected}")
    lines.append(f"  Decoded     : {qr_decoded}")
    lines.append(f"  Verified    : {qr_verified}")
    lines.append(f"  Status      : {status_desc}")
    lines.append("")

    # Risk Assessment Section
    lines.append("----------------------------------------")
    lines.append("RISK ASSESSMENT")
    lines.append("----------------------------------------")
    lines.append("")
    score_val = risk_info.get("score")
    score_str = f"{score_val} / 100" if score_val is not None else "N/A"
    lines.append(f"Risk Score    : {score_str}")
    lines.append(f"Risk Level    : {risk_info.get('level', 'UNKNOWN')}")
    lines.append(f"Decision      : {risk_info.get('decision', 'REVIEW')}")
    lines.append("")

    # Findings Section
    lines.append("----------------------------------------")
    lines.append("FINDINGS")
    lines.append("----------------------------------------")
    lines.append("")

    if not findings:
        lines.append("No risk findings detected.")
        lines.append("")
    else:
        for finding in findings:
            severity = finding.get("severity", "MEDIUM")
            rule_id = finding.get("rule_id", "UNKNOWN_RULE")
            reason = finding.get("reason", "")
            lines.append(f"[{severity}]")
            lines.append(f"{rule_id}")
            lines.append("")
            lines.append(reason)
            lines.append("")

    # Status & Summary Section
    lines.append("----------------------------------------")
    lines.append("STATUS")
    lines.append("----------------------------------------")
    lines.append("")
    summary = risk_info.get("summary", "Automated checks completed.")
    lines.append(summary)
    lines.append("")

    return "\n".join(lines)


def run_pipeline(file_path: str | Path) -> Dict[str, Any]:
    """
    Executes the end-to-end OCR processing and risk assessment pipeline.
    """
    target = Path(file_path)
    if not target.exists():
        print(f"Error: File not found: {target}", file=sys.stderr)
        sys.exit(1)

    # 1. Run OCR document processor
    processing_result = process_document(str(target))

    # 2. Run Risk Assessment engine
    assessment = assess_document(processing_result)

    return assessment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end OCR Document Processing and Risk Assessment."
    )
    parser.add_argument(
        "file_path",
        help="Path to document file (JPG, PNG, PDF, WEBP, etc.)",
    )
    args = parser.parse_args()

    assessment = run_pipeline(args.file_path)
    formatted = format_console_output(assessment)
    print(formatted)


if __name__ == "__main__":
    main()
