"""
Bilingual Tesseract OCR Engine.
Executes high-accuracy English + Hindi (eng+hin) OCR using pytesseract.image_to_data.
Preserves Unicode Devanagari text, extracts word-level bounding boxes, and calculates aggregate confidence.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output

from document_processor.config import (
    DEFAULT_OCR_LANGUAGES,
    DEFAULT_OCR_PSM,
    FALLBACK_OCR_LANGUAGES,
    OCR_TIMEOUT_SECONDS,
    TESSERACT_CMD,
)
from document_processor.preprocessing import PreprocessedDocument

logger = logging.getLogger(__name__)

# Configure pytesseract binary path if discovered
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


class OCREngineError(Exception):
    """
    Raised when Tesseract OCR engine encounters an unrecoverable failure.
    Provides structured error codes safe for API exposure.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        """Returns safe, structured error dictionary without internal system traces."""
        return {
            "success": False,
            "error": {
                "code": self.code,
                "message": self.message,
            },
        }


@dataclass
class OCRWord:
    """Represents a single recognized word token with bounding box and confidence."""

    text: str
    confidence: float  # [0.0, 1.0]
    x: int             # Left pixel offset
    y: int             # Top pixel offset
    width: int
    height: int
    line_num: int = 0
    block_num: int = 0
    par_num: int = 0
    word_num: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "line_num": self.line_num,
            "block_num": self.block_num,
            "par_num": self.par_num,
            "word_num": self.word_num,
        }


@dataclass
class OCRResult:
    """
    Complete structured container for OCR recognition results.
    Preserves raw recognized text, Unicode characters, confidence, and word tokens.
    """

    success: bool
    text: str
    language: str
    confidence: float  # [0.0, 1.0]
    word_count: int
    line_count: int
    words: list[OCRWord] = field(default_factory=list)
    tokens: list[dict[str, Any]] = field(default_factory=list)  # Backwards-compatibility alias
    warnings: list[str] = field(default_factory=list)
    error: Optional[dict[str, str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_words: bool = True) -> dict[str, Any]:
        """Converts OCR result to a JSON-serializable dictionary."""
        res: dict[str, Any] = {
            "success": self.success,
            "text": self.text,
            "language": self.language,
            "confidence": round(self.confidence, 2),
            "word_count": self.word_count,
            "line_count": self.line_count,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }
        if self.error:
            res["error"] = self.error
        if include_words:
            res["words"] = [w.to_dict() for w in self.words]
            res["tokens"] = self.tokens
        return res

    def __getitem__(self, item: str) -> Any:
        """Enables dictionary-style access for backwards compatibility (e.g. res['text'])."""
        return self.to_dict(include_words=True)[item]

    def get(self, key: str, default: Any = None) -> Any:
        """Enables dict.get() interface."""
        return self.to_dict(include_words=True).get(key, default)


def is_tesseract_available() -> bool:
    """Checks if Tesseract executable is installed and callable."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def get_available_tesseract_languages() -> list[str]:
    """Returns list of installed Tesseract language models (e.g. ['eng', 'hin', 'osd'])."""
    try:
        langs = pytesseract.get_languages(config="")
        return langs
    except Exception:
        return []


def check_ocr_health() -> dict[str, Any]:
    """
    Performs comprehensive diagnostic health check on Tesseract system installation.
    Returns:
        {
            "tesseract_installed": bool,
            "tesseract_version": str | None,
            "tesseract_cmd": str | None,
            "available_languages": list[str],
            "has_english": bool,
            "has_hindi": bool,
            "status": "HEALTHY" | "DEGRADED" | "UNAVAILABLE"
        }
    """
    installed = is_tesseract_available()
    version = None
    if installed:
        try:
            version = str(pytesseract.get_tesseract_version())
        except Exception:
            version = "unknown"

    langs = get_available_tesseract_languages() if installed else []
    has_eng = "eng" in langs
    has_hin = "hin" in langs

    if not installed:
        status = "UNAVAILABLE"
    elif has_eng and has_hin:
        status = "HEALTHY"
    else:
        status = "DEGRADED"

    return {
        "tesseract_installed": installed,
        "tesseract_version": version,
        "tesseract_cmd": pytesseract.pytesseract.tesseract_cmd if installed else None,
        "available_languages": langs,
        "has_english": has_eng,
        "has_hindi": has_hin,
        "status": status,
    }


def _convert_to_pil_image(image: Any) -> Image.Image:
    """Converts numpy array or image input to PIL Image."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        if image.size == 0 or len(image.shape) < 2:
            raise OCREngineError("EMPTY_IMAGE", "Cannot perform OCR on empty image array.")
        if len(image.shape) == 3 and image.shape[2] == 3:
            # OpenCV BGR -> RGB for PIL
            return Image.fromarray(image[:, :, ::-1])
        elif len(image.shape) == 2:
            return Image.fromarray(image)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
    raise OCREngineError("INVALID_IMAGE", f"Unsupported image input type: '{type(image).__name__}'.")


def _execute_single_ocr(
    pil_image: Image.Image,
    lang: str = DEFAULT_OCR_LANGUAGES,
    psm: int = DEFAULT_OCR_PSM,
    timeout_seconds: int = OCR_TIMEOUT_SECONDS,
) -> Tuple[str, float, list[OCRWord], list[str]]:
    """
    Executes pytesseract.image_to_data on a single PIL Image.
    Returns:
        (full_text, aggregate_confidence, words_list, warnings)
    """
    warnings: list[str] = []
    config_str = f"--psm {psm}"

    try:
        data = pytesseract.image_to_data(
            pil_image,
            lang=lang,
            config=config_str,
            timeout=timeout_seconds,
            output_type=Output.DICT,
        )
    except pytesseract.pytesseract.TesseractTimeoutExpired:
        raise OCREngineError(
            "OCR_TIMEOUT",
            f"Tesseract OCR processing timed out after {timeout_seconds} seconds.",
        )
    except pytesseract.TesseractError as texc:
        raise OCREngineError(
            "OCR_EXECUTION_ERROR", f"Tesseract execution failed: {str(texc)}"
        )
    except Exception as exc:
        raise OCREngineError(
            "OCR_EXECUTION_ERROR", f"Unexpected error during Tesseract OCR: {str(exc)}"
        )

    words: list[OCRWord] = []
    valid_confidences: list[float] = []
    lines: list[str] = []
    current_line_words: list[str] = []
    last_line_id: Optional[Tuple[int, int, int]] = None  # (block, par, line)

    n_boxes = len(data.get("text", []))
    for i in range(n_boxes):
        raw_word = str(data["text"][i]).strip()
        conf_val = float(data["conf"][i])

        if raw_word:
            conf_norm = max(0.0, conf_val) / 100.0 if conf_val >= 0 else 0.0

            word_obj = OCRWord(
                text=raw_word,
                confidence=conf_norm,
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
                line_num=int(data["line_num"][i]),
                block_num=int(data["block_num"][i]),
                par_num=int(data["par_num"][i]),
                word_num=int(data["word_num"][i]),
            )
            words.append(word_obj)

            if conf_val >= 0:
                valid_confidences.append(conf_val)

            line_id = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
            if last_line_id is not None and line_id != last_line_id and current_line_words:
                lines.append(" ".join(current_line_words))
                current_line_words = []
            current_line_words.append(raw_word)
            last_line_id = line_id

    if current_line_words:
        lines.append(" ".join(current_line_words))

    full_text = "\n".join(lines).strip()

    if valid_confidences:
        avg_conf = sum(valid_confidences) / len(valid_confidences)
        aggregate_confidence = round(max(0.0, min(avg_conf / 100.0, 1.0)), 2)
    else:
        aggregate_confidence = 0.0

    if not full_text:
        warnings.append("OCR recognized no text content.")

    return full_text, aggregate_confidence, words, warnings


def _clean_text_for_comparison(s: str) -> str:
    """Strips punctuation, symbols, and whitespace for fuzzy duplicate detection."""
    return re.sub(r"[^\w\u0900-\u097F]+", "", s.lower())


def _is_line_duplicate(line: str, existing_lines: list[str], threshold: float = 0.65) -> bool:
    """
    Checks if a recognized line from a secondary variant is already present
    in the primary recognized lines to avoid duplicate text injection.
    """
    clean_target = _clean_text_for_comparison(line)
    if not clean_target or len(clean_target) < 2:
        return True
    for el in existing_lines:
        clean_el = _clean_text_for_comparison(el)
        if not clean_el:
            continue
        if clean_target in clean_el or clean_el in clean_target:
            return True
        sim = SequenceMatcher(None, clean_target, clean_el).ratio()
        if sim >= threshold:
            return True
    return False


def run_ocr(
    image: Union[np.ndarray, Image.Image, PreprocessedDocument, list[Tuple[str, np.ndarray]]],
    lang: str = DEFAULT_OCR_LANGUAGES,
    languages: Optional[str] = None,  # Backwards-compatible alias for lang
    psm: int = DEFAULT_OCR_PSM,
    timeout_seconds: int = OCR_TIMEOUT_SECONDS,
    allow_fallback: bool = True,
) -> OCRResult:
    """
    Main OCR engine entrypoint.
    Accepts an image or preprocessed document variants, validates Tesseract and language models,
    and extracts structured text, confidence scores, and token bounding boxes.
    Supports multi-variant OCR evaluation and non-duplicate token fusion so small/fine text is preserved.

    Returns:
        OCRResult container with text, word-level bounding boxes, and aggregate confidence.
    """
    start_time = time.time()
    effective_lang = languages if languages is not None else lang
    warnings: list[str] = []

    # Step 1: Verify Tesseract Availability
    if not is_tesseract_available():
        raise OCREngineError(
            "OCR_ENGINE_UNAVAILABLE",
            "Tesseract OCR executable was not found. Please install Tesseract OCR and configure TESSERACT_CMD.",
        )

    # Step 2: Language Availability & Fallback Check
    available_langs = get_available_tesseract_languages()
    lang_to_use = effective_lang

    if "hin" in effective_lang and "hin" not in available_langs:
        if allow_fallback:
            warnings.append(
                "Hindi OCR language data (hin) is unavailable in Tesseract installation. Falling back to English OCR."
            )
            lang_to_use = FALLBACK_OCR_LANGUAGES
        else:
            raise OCREngineError(
                "LANGUAGE_DATA_UNAVAILABLE",
                "Hindi language data ('hin') is not installed in Tesseract.",
            )

    # Step 3: Resolve Image Variants for Evaluation
    candidate_images: list[Tuple[str, Image.Image]] = []

    if isinstance(image, PreprocessedDocument):
        ocr_variants = image.get_ocr_images()
        for name, arr in ocr_variants[:3]:
            candidate_images.append((name, _convert_to_pil_image(arr)))
    elif isinstance(image, list):
        for item in image[:3]:
            if isinstance(item, tuple) and len(item) == 2:
                name, arr = item
                candidate_images.append((name, _convert_to_pil_image(arr)))
            else:
                candidate_images.append(("variant", _convert_to_pil_image(item)))
    else:
        candidate_images.append(("primary", _convert_to_pil_image(image)))

    if not candidate_images:
        raise OCREngineError("INVALID_IMAGE", "No valid image candidates provided for OCR.")

    # Step 4: Execute OCR with Multi-Variant Evaluation & Non-Duplicate Fusion
    all_warnings = list(warnings)
    executed_results: list[dict[str, Any]] = []

    for var_name, pil_img in candidate_images[:3]:
        text, conf, words, var_warnings = _execute_single_ocr(
            pil_image=pil_img,
            lang=lang_to_use,
            psm=psm,
            timeout_seconds=timeout_seconds,
        )
        all_warnings.extend(var_warnings)
        executed_results.append({
            "name": var_name,
            "text": text,
            "conf": conf,
            "words": words,
            "score": len(words) * 2 + conf * 10,
        })

    if not executed_results:
        raise OCREngineError("OCR_FAILED", "No OCR results obtained from candidate images.")

    # Primary variant is the candidate image provided first (or base variant)
    primary_res = executed_results[0]
    merged_lines = [l for l in primary_res["text"].splitlines() if l.strip()]
    merged_words = list(primary_res["words"])
    supplemental_lines_count = 0
    supplemental_variants_used: list[str] = []

    # Merge non-duplicate high-confidence lines from secondary variants (e.g. grayscale fine text recovery)
    for sec_res in executed_results[1:]:
        sec_line_word_map: dict[Tuple[int, int, int], list[OCRWord]] = {}
        for w in sec_res["words"]:
            lid = (w.block_num, w.par_num, w.line_num)
            sec_line_word_map.setdefault(lid, []).append(w)

        variant_added_any = False
        for lid, line_words in sec_line_word_map.items():
            line_str = " ".join(w.text for w in line_words).strip()
            if not line_str:
                continue
            avg_line_conf = sum(w.confidence for w in line_words) / len(line_words)
            if avg_line_conf < 0.40:
                continue
            if not _is_line_duplicate(line_str, merged_lines):
                merged_lines.append(line_str)
                merged_words.extend(line_words)
                supplemental_lines_count += 1
                variant_added_any = True

        if variant_added_any:
            supplemental_variants_used.append(sec_res["name"])

    best_text = "\n".join(merged_lines).strip()
    valid_confidences = [w.confidence * 100.0 for w in merged_words if w.confidence >= 0]
    if valid_confidences:
        best_conf = round(max(0.0, min(sum(valid_confidences) / len(valid_confidences) / 100.0, 1.0)), 2)
    else:
        best_conf = primary_res["conf"]

    best_words = merged_words
    best_variant_name = primary_res["name"]
    if supplemental_variants_used:
        best_variant_name = f"{primary_res['name']}+{'+'.join(supplemental_variants_used)}"

    duration_ms = int((time.time() - start_time) * 1000)
    lines_count = len([l for l in best_text.splitlines() if l.strip()])

    tokens_compat = [
        {
            "text": w.text,
            "confidence": round(w.confidence, 4),
            "left": w.x,
            "top": w.y,
            "width": w.width,
            "height": w.height,
            "line_num": w.line_num,
        }
        for w in best_words
    ]

    metadata = {
        "execution_time_ms": duration_ms,
        "variant_used": best_variant_name,
        "psm": psm,
        "raw_language_requested": effective_lang,
        "supplemental_lines_added": supplemental_lines_count,
        "variants_evaluated": [v["name"] for v in executed_results],
    }

    # Safe log without sensitive PII
    logger.info(
        "OCR executed: lang=%s, words=%d, conf=%.2f, time=%dms, variants=%s",
        lang_to_use,
        len(best_words),
        best_conf,
        duration_ms,
        metadata["variants_evaluated"],
    )

    return OCRResult(
        success=True,
        text=best_text,
        language=lang_to_use,
        confidence=best_conf,
        word_count=len(best_words),
        line_count=lines_count,
        words=best_words,
        tokens=tokens_compat,
        warnings=list(dict.fromkeys(all_warnings)),  # deduplicate warnings
        error=None,
        metadata=metadata,
    )
