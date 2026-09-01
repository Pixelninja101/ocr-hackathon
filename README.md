# QR Detection and Decoding Module

A robust QR code detection and decoding module built using OpenCV and NumPy, with optional ZXing-C++ fallback support.

## Features

- Single and multiple QR code detection
- OpenCV `QRCodeDetectorAruco` support with legacy `QRCodeDetector` fallback
- Multi-stage QR decoding pipeline
- Text and binary QR payload decoding
- Support for rotated, skewed, low-resolution, noisy, and low-contrast QR codes
- Perspective rectification using detected QR corner points
- Image padding for QR codes near image boundaries
- Quiet-zone generation before decoding
- Grayscale, CLAHE, Otsu, adaptive thresholding, sharpening, and inversion
- Multi-scale QR-region cropping and upscaling
- Optional ZXing-C++ fallback
- Support for QR Code, Micro QR Code, and RMQR Code through ZXing
- UTF-8, Latin-1, and hexadecimal binary payload handling

---

## Detection and Decoding Pipeline

The module uses multiple fallback stages instead of relying on a single QR decoding attempt:

1. Multi-QR detection and decoding
2. Single QR detection and decoding
3. Padded image decoding
4. Perspective rectification
5. Multiple image preprocessing variants
6. Multi-scale QR region cropping and upscaling
7. ZXing-C++ fallback

Each stage is attempted only when the previous stage fails.

---

## Image Preprocessing

The decoder generates multiple variants of the QR image:

- Original image
- Grayscale
- CLAHE contrast enhancement
- Otsu thresholding
- Adaptive thresholding
- Sharpened grayscale
- Inverted image
- 2x upscaled grayscale
- 2x upscaled thresholded image

---

## Perspective Rectification

For rotated or perspective-distorted QR codes, the detected four corner points are reordered and transformed into a square image.

A quiet zone is added around the rectified QR image before decoding is retried.

Multiple target sizes are used:

```text
500px
700px
350px
````

---

## Binary Payload Handling

Binary QR payloads are decoded using:

```python
detectAndDecodeBytes()
detectAndDecodeBytesMulti()
```

Payload data is processed in the following order:

1. UTF-8 decoding
2. Latin-1 decoding
3. Hexadecimal representation

---

## ZXing Fallback

When OpenCV decoding fails, `zxingcpp` is optionally used as a fallback.

Supported formats include:

* QR Code
* Micro QR Code
* RMQR Code

ZXing is applied to the original image, cropped QR regions, thresholded images, upscaled images, and perspective-rectified images.

---

## Installation

```powershell
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install opencv-python numpy
```

Optional ZXing-C++ support:

```powershell
pip install zxing-cpp
```

---

## Basic Usage

```python
import cv2

from qr_detector import QRCodeDetectorWrapper
from qr_decoder import QRCodeDecoderWrapper

image = cv2.imread("image.jpg")

detector = QRCodeDetectorWrapper()
decoder = QRCodeDecoderWrapper()

detected, quads = detector.detect_multi(image)

if detected:
    for quad in quads:
        success, payload, method, attempts = decoder.decode_quad(
            image,
            quad
        )

        if success:
            print("Decoded Data:", payload)
            print("Method:", method)
            print("Attempts:", attempts)
```

---

## Multi-QR Decoding

```python
success, payloads, quads, methods = decoder.detect_and_decode_multi(image)

if success:
    for payload, method in zip(payloads, methods):
        print("Decoded Data:", payload)
        print("Method:", method)
```

---

## Return Values

### Detection

```python
detected, quads = detector.detect_multi(image)
```

* `detected`: Whether one or more QR codes were detected
* `quads`: List of QR corner points

### Single QR Decoding

```python
success, payload, method, attempts = decoder.decode_quad(
    image,
    quad_points
)
```

* `success`: Whether decoding was successful
* `payload`: Decoded QR data
* `method`: Successful decoding strategy
* `attempts`: Number of decoding attempts

### Multiple QR Decoding

```python
success, payloads, quads, methods = decoder.detect_and_decode_multi(image)
```

* `success`: Whether QR codes were detected or decoded
* `payloads`: Decoded QR payloads
* `quads`: QR corner coordinates
* `methods`: Decoding methods used

---

## Limitations

The module uses multiple detection and decoding strategies to improve reliability. However, severely damaged, heavily blurred, extremely low-resolution, or partially missing QR codes may still be impossible to decode.

````
