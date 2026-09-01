from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from .detector import QRCodeDetectorWrapper
from .decoder import QRCodeDecoderWrapper, add_padding_to_image
from .parser import parse_qr_payload
from ..utils.image_handler import calculate_bbox_metrics, enhance_for_qr, normalize_points


def calculate_quad_iou(pts1: List[List[float]], pts2: List[List[float]]) -> float:
    if not pts1 or not pts2 or len(pts1) < 4 or len(pts2) < 4:
        return 0.0

    p1 = np.array(pts1)
    p2 = np.array(pts2)

    min_x1, min_y1 = np.min(p1, axis=0)
    max_x1, max_y1 = np.max(p1, axis=0)
    min_x2, min_y2 = np.min(p2, axis=0)
    max_x2, max_y2 = np.max(p2, axis=0)

    inter_x1 = max(min_x1, min_x2)
    inter_y1 = max(min_y1, min_y2)
    inter_x2 = min(max_x1, max_x2)
    inter_y2 = min(max_y1, max_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (max_x1 - min_x1) * (max_y1 - min_y1))
    area2 = max(0.0, (max_x2 - min_x2) * (max_y2 - min_y2))

    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class QRProcessor:

    def __init__(self):
        self.detector_wrapper = QRCodeDetectorWrapper()

        self.decoder = QRCodeDecoderWrapper(
            primary_detector=self.detector_wrapper.primary_detector,
            legacy_detector=self.detector_wrapper.legacy_detector,
        )

        self._detect_detectors = [
            self.detector_wrapper.legacy_detector,
            self.detector_wrapper.primary_detector,
        ]

    def process_image(
        self,
        cv2_img: np.ndarray,
        filename: Optional[str] = None,
        file_type: str = "image",
        page_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        if cv2_img is None or cv2_img.size == 0:
            return {
                "success": False,
                "qr_detected": False,
                "qr_count": 0,
                "codes": [],
                "warnings": [],
                "errors": ["Invalid or empty image buffer provided for processing."],
            }

        height, width = cv2_img.shape[:2]

        detected_codes: List[Dict[str, Any]] = []
        warnings: List[str] = []

        max_dim = max(height, width)

        if max_dim > 1800:
            scale_down = 1800.0 / float(max_dim)
            search_canvas = cv2.resize(
                cv2_img,
                (0, 0),
                fx=scale_down,
                fy=scale_down,
                interpolation=cv2.INTER_AREA,
            )
        else:
            scale_down = 1.0
            search_canvas = cv2_img

        detected, texts, quads, methods = (
            self.decoder.detect_and_decode_multi(search_canvas)
        )

        if detected and quads:
            self._append_codes(
                detected_codes,
                quads,
                texts,
                methods,
                scale_down,
            )

        if not detected_codes:
            pad_amount = max(80, min(width, height) // 5)

            padded, pad = add_padding_to_image(
                search_canvas,
                pad=pad_amount,
            )

            p_det, p_texts, p_quads, p_methods = (
                self.decoder.detect_and_decode_multi(padded)
            )

            if p_det and p_quads:
                mapped_quads = [
                    q - np.array([pad, pad], dtype=np.float32)
                    for q in p_quads
                ]

                self._append_codes(
                    detected_codes,
                    mapped_quads,
                    p_texts,
                    p_methods,
                    scale_down,
                )

        # ponytail: Manual 90/180/270 rotations are YAGNI (QR finders are natively rotation invariant).
        # ponytail: Replaced massive scaling/enhancement passes with a simple native 4-quadrant overlap tile scan.
        if not detected_codes:
            h, w = search_canvas.shape[:2]
            tiles = [
                (0, 0, search_canvas[0:h*2//3, 0:w*2//3]),
                (0, w//3, search_canvas[0:h*2//3, w//3:w]),
                (h//3, 0, search_canvas[h//3:h, 0:w*2//3]),
                (h//3, w//3, search_canvas[h//3:h, w//3:w])
            ]
            for dy, dx, tile in tiles:
                t_det, t_texts, t_quads, t_methods = self.decoder.detect_and_decode_multi(tile)
                if t_det and t_quads:
                    mapped_quads = [q + np.array([dx, dy], dtype=np.float32) for q in t_quads]
                    self._append_codes(detected_codes, mapped_quads, t_texts, t_methods, scale_down)

        for code in detected_codes:

            if not code["decoded"]:

                quad_np = code.get("_quad_np")

                if quad_np is None:
                    quad_np = np.array(
                        code["bbox"],
                        dtype=np.float32,
                    )

                try:
                    dec_ok, dec_payload, dec_method, dec_attempts = (
                        self.decoder.decode_quad(
                            cv2_img,
                            quad_np,
                        )
                    )

                    code["attempts"] += dec_attempts

                    if dec_ok and dec_payload:
                        code["decoded"] = True
                        code["data"] = dec_payload
                        code["decode_method"] = dec_method

                except Exception:
                    pass

        unique_codes = self._deduplicate_codes(
            detected_codes
        )

        qr_count = len(unique_codes)
        qr_detected = qr_count > 0

        formatted_codes: List[Dict[str, Any]] = []

        for idx, code in enumerate(unique_codes):
            parsed = None
            if code["decoded"] and code.get("data"):
                try:
                    parsed = parse_qr_payload(code["data"])
                except Exception:
                    pass

            formatted_codes.append(
                {
                    "id": idx + 1,
                    "decoded": code["decoded"],
                    "data": code["data"],
                    "parsed_data": parsed,
                    "bbox": code["bbox"],
                    "center": code["center"],
                    "width": code["width"],
                    "height": code["height"],
                    "area": code["area"],
                    "decode_method": code.get("decode_method"),
                    "attempts": max(
                        1,
                        code.get("attempts", 1),
                    ),
                }
            )

        if not qr_detected:
            warnings.append("No QR code detected")
        else:
            undecoded_count = sum(
                not c["decoded"]
                for c in formatted_codes
            )

            if undecoded_count == qr_count:
                warnings.append(
                    "QR detected but decoding failed"
                )

            elif undecoded_count > 0:
                warnings.append(
                    f"{undecoded_count} of {qr_count} detected QR codes could not be decoded"
                )

        response: Dict[str, Any] = {
            "success": True,
            "qr_detected": qr_detected,
            "qr_count": qr_count,
            "codes": formatted_codes,
            "warnings": warnings,
            "errors": [],
        }

        meta: Dict[str, Any] = {
            "filename": filename or "unknown",
            "file_type": file_type,
            "image_width": width,
            "image_height": height,
        }

        if page_metadata:
            meta.update(page_metadata)

        response["metadata"] = meta

        return response

    def _detect_at_scale(
        self,
        image: np.ndarray,
        scale: float,
    ) -> tuple:

        if image is None or image.size == 0:
            return [], [], []

        scaled = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        all_quads = []
        all_texts = []
        all_methods = []

        try:
            ok, texts, quads, methods = (
                self.decoder.detect_and_decode_multi(
                    scaled
                )
            )

            if quads:
                for i, quad in enumerate(quads):

                    original_quad = (
                        np.asarray(
                            quad,
                            dtype=np.float32,
                        ) / scale
                    )

                    all_quads.append(original_quad)

                    text = (
                        texts[i]
                        if i < len(texts)
                        else ""
                    )

                    method = (
                        methods[i]
                        if i < len(methods)
                        else ""
                    )

                    all_texts.append(text or "")
                    all_methods.append(method or "")

        except Exception:
            pass

        if not all_quads:

            try:
                detected, quads = (
                    self.detector_wrapper.detect_multi(
                        scaled
                    )
                )

                if detected and quads:

                    for quad in quads:

                        original_quad = (
                            np.asarray(
                                quad,
                                dtype=np.float32,
                            ) / scale
                        )

                        all_quads.append(
                            original_quad
                        )

                        all_texts.append("")
                        all_methods.append(
                            f"geometry_{scale}x"
                        )

            except Exception:
                pass

        return (
            all_quads,
            all_texts,
            all_methods,
        )

    def _append_codes(
        self,
        detected_codes: List[Dict[str, Any]],
        quads: List[np.ndarray],
        texts: List[str],
        methods: List[str],
        scale_down: float,
    ) -> None:

        for idx, quad in enumerate(quads):

            orig_quad = (
                quad / scale_down
                if scale_down != 1.0
                else quad
            )

            norm_pts = normalize_points(
                orig_quad
            )

            if not norm_pts:
                continue

            metrics = calculate_bbox_metrics(
                norm_pts
            )

            text = (
                texts[idx]
                if idx < len(texts)
                else ""
            )

            method = (
                methods[idx]
                if idx < len(methods)
                else ""
            )

            is_decoded = bool(
                text and text.strip()
            )

            detected_codes.append(
                {
                    "decoded": is_decoded,
                    "data": (
                        text.strip()
                        if is_decoded
                        else None
                    ),
                    "bbox": norm_pts,
                    "center": metrics["center"],
                    "width": metrics["width"],
                    "height": metrics["height"],
                    "area": metrics["area"],
                    "decode_method": (
                        method
                        if is_decoded
                        else None
                    ),
                    "attempts": 1,
                    "_quad_np": np.asarray(
                        orig_quad,
                        dtype=np.float32,
                    ),
                }
            )

    def _deduplicate_codes(
        self,
        codes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        if len(codes) <= 1:
            return codes

        unique: List[Dict[str, Any]] = []

        sorted_codes = sorted(
            codes,
            key=lambda x: (
                x["decoded"],
                x["area"],
            ),
            reverse=True,
        )

        for code in sorted_codes:

            is_dup = False

            for existing in unique:

                iou = calculate_quad_iou(
                    code["bbox"],
                    existing["bbox"],
                )

                if iou > 0.35:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(code)

        return unique