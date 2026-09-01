"""
Aadhaar Document Verification & Heuristic Risk Engine Demo.
Runs synthetic verification scenarios and displays clean, privacy-safe results.

Usage:
    python examples/demo_verification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure risk_engine root is in python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from integration.verification_service import verify_document
from tests.fixtures.verification_cases import DEMO_CASES


def run_demo() -> None:
    print("=" * 65)
    print("      AADHAAR DOCUMENT VERIFICATION & RISK ENGINE DEMO")
    print("=" * 65)
    print("Privacy Guarantee: Zero unmasked PII, OCR text, or QR payloads.\n")

    for idx, case in enumerate(DEMO_CASES, start=1):
        case_name = case["name"]
        description = case["description"]
        payload = case["payload"]

        # Run verification through the backend service layer
        result = verify_document(payload)
        ver = result["verification"]
        checks = result["checks"]
        findings = result["findings"]

        score_display = f"{ver['risk_score']}/100" if ver["risk_score"] is not None else "N/A"
        override_display = (
            f"YES ({', '.join(ver['override_reasons'])})"
            if ver["override_applied"]
            else "None"
        )

        print(f"[{idx}] Scenario: {case_name}")
        print(f"    Description : {description}")
        print(f"    Risk Level  : {ver['risk_level']}")
        print(f"    Risk Score  : {score_display}")
        print(f"    Decision    : {ver['decision']}")
        print(f"    Override    : {override_display}")
        print(f"    Summary     : {ver['summary']}")
        print(f"    Checks      : Checksum={checks['checksum']['valid']} | QR={checks['qr']['status']} | NameMatch={checks['cross_validation']['name_match']}")
        print(f"    Findings    : {len(findings)} explainable rule(s) triggered")
        for f in findings:
            print(f"      - [{f['severity']}] {f['rule_id']} (+{f['points']} pts): {f['reason']}")
        print("-" * 65)

    print("\nDemo completed successfully. All cases verified without PII exposure.")


if __name__ == "__main__":
    run_demo()
