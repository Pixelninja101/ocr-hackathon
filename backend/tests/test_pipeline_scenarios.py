"""
Comprehensive QR pipeline test script.
Tests all required scenarios and prints pass/fail for each.
"""
import sys
sys.path.insert(0, '.')

import qrcode
import io
import cv2
import numpy as np

from backend.qr.processor import QRProcessor

processor = QRProcessor()


def make_qr(data, box_size=10, error_correction=qrcode.constants.ERROR_CORRECT_H):
    qr = qrcode.QRCode(
        version=1,
        error_correction=error_correction,
        box_size=box_size,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color='black', back_color='white')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    np_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def run_test(label, cv_img, expect_decoded=True, expect_data=None):
    r = processor.process_image(cv_img, label)
    detected = r["qr_detected"]
    codes = r["codes"]
    code = codes[0] if codes else None
    decoded = code["decoded"] if code else False
    data = code["data"] if code else None
    method = code.get("decode_method") if code else None
    attempts = code.get("attempts", 0) if code else 0

    status = "PASS" if (detected == (expect_decoded or not expect_decoded)) else "?"
    if expect_decoded and not decoded:
        status = "FAIL"
    elif not expect_decoded and detected:
        status = "WARN"
    elif expect_data and data != expect_data:
        status = "FAIL (wrong data)"
    else:
        status = "PASS"

    print(f"[{status}] {label}")
    print(f"       detected={detected}, decoded={decoded}, attempts={attempts}")
    if data:
        print(f"       data={repr(data[:60])}")
    if method:
        print(f"       method={method}")
    if r["warnings"]:
        print(f"       warnings={r['warnings']}")
    print()


# ─── Test 1: Known-good QR ────────────────────────────────────────────────────
cv_img = make_qr("HACKVERSE QR TEST 2026")
run_test("1. Clean QR (HACKVERSE QR TEST 2026)", cv_img, expect_decoded=True, expect_data="HACKVERSE QR TEST 2026")

# ─── Test 2: URL QR ───────────────────────────────────────────────────────────
url_qr = make_qr("https://uidai.gov.in/verify")
run_test("2. URL QR", url_qr, expect_decoded=True, expect_data="https://uidai.gov.in/verify")

# ─── Test 3: Rotated 45° (realistic: QR inside larger document canvas) ──────
# NOTE: A 45°-rotated QR in a same-size canvas clips the finder patterns — physically
# impossible to decode. In real usage, the document image is larger than the QR.
h, w = cv_img.shape[:2]
canvas_size = max(w, h) * 3
canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
ox, oy = (canvas_size - w) // 2, (canvas_size - h) // 2
canvas[oy:oy+h, ox:ox+w] = cv_img
M45 = cv2.getRotationMatrix2D((canvas_size/2, canvas_size/2), 45, 1.0)
rotated45 = cv2.warpAffine(canvas, M45, (canvas_size, canvas_size), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
run_test("3. Rotated 45deg (QR in document canvas)", rotated45, expect_decoded=True)

# ─── Test 4: Rotated 180° ────────────────────────────────────────────────────
M180 = cv2.getRotationMatrix2D((w/2, h/2), 180, 1.0)
rotated180 = cv2.warpAffine(cv_img, M180, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
run_test("4. Rotated 180deg", rotated180, expect_decoded=True)

# ─── Test 5: Perspective distorted ───────────────────────────────────────────
src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
dst_pts = np.float32([[30, 0], [w, 30], [w - 20, h], [0, h - 20]])
M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
perspective_distorted = cv2.warpPerspective(cv_img, M_persp, (w, h), borderValue=(255, 255, 255))
run_test("5. Perspective distorted", perspective_distorted, expect_decoded=True)

# ─── Test 6: Small QR (box_size=3) ───────────────────────────────────────────
small_qr = make_qr("HACKVERSE QR TEST 2026", box_size=3)
run_test("6. Small QR (box_size=3)", small_qr, expect_decoded=True)

# ─── Test 7: No QR ────────────────────────────────────────────────────────────
blank = np.full((400, 400, 3), 255, dtype=np.uint8)
cv2.putText(blank, "No QR here", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
run_test("7. No QR image", blank, expect_decoded=False)

# ─── Test 8: Multiple QR codes ───────────────────────────────────────────────
qr_a = make_qr("QR_CODE_A_DATA", box_size=6)
qr_b = make_qr("QR_CODE_B_DATA", box_size=6)
ha, wa = qr_a.shape[:2]
hb, wb = qr_b.shape[:2]
multi_canvas = np.full((max(ha, hb), wa + wb + 20, 3), 255, dtype=np.uint8)
multi_canvas[0:ha, 0:wa] = qr_a
multi_canvas[0:hb, wa + 20:wa + 20 + wb] = qr_b
r_multi = processor.process_image(multi_canvas, "multi_qr.png")
print(f"[{'PASS' if r_multi['qr_count'] >= 2 else 'WARN'}] 8. Multiple QR codes")
print(f"       detected_count={r_multi['qr_count']}")
for ci, c in enumerate(r_multi["codes"]):
    print(f"       QR#{ci+1}: decoded={c['decoded']}, data={repr(c['data'][:40] if c['data'] else None)}")
print()

print("=" * 50)
print("Test run complete.")
