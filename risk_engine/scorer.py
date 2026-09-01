"""
Scoring and document risk assessment orchestrator for Risk Engine.
Converts explainable risk rule findings into a normalized 0-100 score, categorical tier, and decision.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

from risk_engine.models import (
    DocumentRiskSummary,
    RiskAssessmentResult,
    RiskDecision,
    RiskLevel,
    mask_sensitive_number,
)
from risk_engine.rules import evaluate_rules
from risk_engine.signals import extract_signals

logger = logging.getLogger("risk_engine.scorer")

# Scoring Parameters & Thresholds
MAX_RAW_SCORE: int = 200
LOW_RISK_THRESHOLD: int = 30     # 0–29 -> LOW
HIGH_RISK_THRESHOLD: int = 60    # 30–59 -> MEDIUM, 60–100 -> HIGH

# Override Rules that mandate at least HIGH risk level
HIGH_RISK_OVERRIDE_RULES: frozenset[str] = frozenset({
    "NAME_MISMATCH",
    "DOB_MISMATCH",
    "GENDER_MISMATCH",
    "QR_VERIFICATION_FAILED",
    "DOCUMENT_NOT_IDENTIFIED",
})


def calculate_raw_score(findings: List[Dict[str, Any]]) -> int:
    """
    Calculates the sum of points from all triggered risk findings.
    """
    if not findings:
        return 0
    return sum(int(f.get("points", 0)) for f in findings if f.get("triggered", True))


def normalize_risk_score(raw_points: int, max_raw_score: int = MAX_RAW_SCORE) -> int:
    """
    Normalizes raw additive rule points to a 0–100 heuristic risk scale.
    Clamps the result between 0 and 100.
    """
    if raw_points <= 0:
        return 0
    normalized = round((raw_points / max_raw_score) * 100)
    return max(0, min(100, int(normalized)))


def classify_risk_level(score: int) -> str:
    """
    Categorizes numeric risk score into standard risk tiers.
    0–29   -> LOW
    30–59  -> MEDIUM
    60–100 -> HIGH
    """
    if score < LOW_RISK_THRESHOLD:
        return RiskLevel.LOW.value
    elif score < HIGH_RISK_THRESHOLD:
        return RiskLevel.MEDIUM.value
    else:
        return RiskLevel.HIGH.value


def apply_risk_overrides(
    base_level: str,
    flags: List[str],
) -> Tuple[str, Optional[str]]:
    """
    Applies mandatory overrides for critical identity contradictions and non-Aadhaar classifications.
    Returns (effective_level, override_reason).
    """
    triggered_overrides = [flag for flag in flags if flag in HIGH_RISK_OVERRIDE_RULES]

    if not triggered_overrides:
        return base_level, None

    if "DOCUMENT_NOT_IDENTIFIED" in triggered_overrides:
        return RiskLevel.HIGH.value, "DOCUMENT_NOT_IDENTIFIED"

    # Identity mismatch or cryptographic verification failure
    return RiskLevel.HIGH.value, "CRITICAL_SIGNAL_OVERRIDE"


def determine_decision(level: str) -> str:
    """
    Determines actionable verification recommendation based on risk level.
    LOW    -> PASS
    MEDIUM -> REVIEW
    HIGH   -> REVIEW
    """
    if level == RiskLevel.LOW.value:
        return RiskDecision.PASS.value
    else:
        return RiskDecision.REVIEW.value


def generate_risk_summary(
    level: str,
    flags: List[str],
    override_reason: Optional[str] = None,
) -> str:
    """
    Generates a concise, non-accusatory, machine-readable risk summary string.
    """
    if override_reason == "DOCUMENT_NOT_IDENTIFIED" or "DOCUMENT_NOT_IDENTIFIED" in flags:
        return "The uploaded document could not be confidently identified as Aadhaar and requires review."

    if level == RiskLevel.HIGH.value:
        return "Document contains significant verification or identity-consistency risk signals and requires review."
    elif level == RiskLevel.MEDIUM.value:
        return "Document requires additional review due to one or more risk signals."
    elif level == RiskLevel.LOW.value:
        return "Document passed the available automated checks with low observed risk."
    else:
        return "Document assessment requires additional review."


def score_document(
    findings: List[Dict[str, Any]],
    flags: List[str],
) -> DocumentRiskSummary:
    """
    Scores a collection of findings and generates a DocumentRiskSummary.
    """
    raw_points = calculate_raw_score(findings)
    normalized_score = normalize_risk_score(raw_points)
    base_level = classify_risk_level(normalized_score)

    final_level, override_reason = apply_risk_overrides(base_level, flags)
    decision = determine_decision(final_level)
    summary = generate_risk_summary(final_level, flags, override_reason)

    return DocumentRiskSummary(
        score=normalized_score,
        level=final_level,
        decision=decision,
        summary=summary,
    )


def assess_document(
    processing_result: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Main public API entrypoint for assessing document risk from OCR processing results.

    Parameters:
        processing_result (dict | None): Output dictionary from document_processor.process_document()

    Returns:
        dict: Strictly JSON-serializable risk assessment result adhering to security and privacy rules.
    """
    # 1. Handle None or non-dictionary input
    if processing_result is None or not isinstance(processing_result, dict):
        logger.warning("Assessment invoked with invalid or None input.")
        res = RiskAssessmentResult(
            success=False,
            risk=DocumentRiskSummary(
                score=None,
                level=RiskLevel.UNKNOWN.value,
                decision=RiskDecision.REVIEW.value,
                summary="The provided input is None or not a valid dictionary.",
            ),
            findings=[],
            flags=["INVALID_INPUT"],
            signals={},
            warnings=["Input must be a non-empty processing result dictionary."],
            error={
                "code": "INVALID_INPUT",
                "message": "The provided input is None or not a valid dictionary.",
            },
            status="INVALID_INPUT",
        )
        return res.to_dict()

    # 2. Defensive copy to guarantee input immutability
    input_copy = copy.deepcopy(processing_result)

    # 3. Handle upstream processing failure (success == False)
    if not input_copy.get("success", False):
        err = input_copy.get("error")
        err_dict = dict(err) if isinstance(err, dict) else {
            "code": "PROCESSING_FAILED",
            "message": "Upstream document processing reported failure.",
        }
        raw_warnings = input_copy.get("warnings", [])
        warnings_list = [str(w) for w in raw_warnings] if isinstance(raw_warnings, list) else []

        logger.info(
            "Document processing was unsuccessful: error_code=%s",
            err_dict.get("code", "UNKNOWN"),
        )

        res = RiskAssessmentResult(
            success=False,
            risk=DocumentRiskSummary(
                score=None,
                level=RiskLevel.UNKNOWN.value,
                decision=RiskDecision.REVIEW.value,
                summary=f"Upstream document processing failed ({err_dict.get('code', 'PROCESSING_FAILED')}).",
            ),
            findings=[],
            flags=["DOCUMENT_PROCESSING_FAILED"],
            signals={},
            warnings=warnings_list,
            error=err_dict,
            status="PROCESSING_FAILED",
        )
        return res.to_dict()

    # 4. Extract normalized signals safely from the input
    signals = extract_signals(input_copy)
    raw_warnings = input_copy.get("warnings", [])
    warnings_list = [str(w) for w in raw_warnings] if isinstance(raw_warnings, list) else []

    # 5. Evaluate explainable risk rules
    rules_eval = evaluate_rules(signals)
    findings = rules_eval.get("findings", [])
    flags = [f["rule_id"] for f in findings]

    # 6. Compute final risk score, level, decision, and summary
    risk_summary = score_document(findings, flags)

    # Safe summary logging without PII
    logger.info(
        "Document assessed: score=%d, level=%s, decision=%s, findings=%d, flags=%s",
        risk_summary.score if risk_summary.score is not None else -1,
        risk_summary.level,
        risk_summary.decision,
        len(findings),
        flags,
    )

    # 7. Assemble final standardized assessment response
    assessment = RiskAssessmentResult(
        success=True,
        risk=risk_summary,
        findings=findings,
        flags=flags,
        signals=signals,
        warnings=warnings_list,
        error=None,
        status="SUCCESS",
    )

    return assessment.to_dict()
