# Personna: AI-Based Fake Identity and Document Screening System

A modular document processing and verification system for web-based identity screening.

Personna combines OCR, QR code processing, Aadhaar Secure QR parsing, and document validation to extract and analyse information from identity documents. The current system is designed to process uploaded documents through a web interface and return structured results.

## Features

- File validation and document image processing
- OCR-based extraction of identity information
- Aadhaar document detection using multiple signals
- Bilingual OCR support for English and Hindi
- Word-level bounding boxes and confidence scoring
- QR code detection and decoding
- Support for multiple QR codes in a single image
- Robust QR decoding for rotated, skewed, dim, noisy, blurry, low-contrast, and low-resolution images
- Multi-stage image preprocessing and decoding fallbacks
- Aadhaar Secure QR detection and payload parsing
- Structured Aadhaar detail extraction
- Raw QR payload preservation
- Cross-validation of OCR and QR data

---

## QR Detection and Decoding

The QR module uses a multi-stage detection and decoding pipeline to handle different image conditions.

The pipeline includes:

- OpenCV QR code detection
- Support for multiple QR codes
- Direct decoding
- Multi-QR decoding
- Image padding
- Perspective rectification
- Grayscale conversion
- CLAHE contrast enhancement
- Otsu and adaptive thresholding
- Sharpening and inversion
- Multi-scale upscaling
- QR-region cropping
- ZXing-C++ fallback decoding

The decoder progressively applies fallback methods when a QR cannot be decoded directly.

```text
Input Image
     |
     v
QR Detection
     |
     v
Direct QR Decoding
     |
     +----> Success
     |
     v
Padding and Rectification
     |
     v
Image Preprocessing Variants
     |
     v
Multi-Scale Decoding
     |
     v
OpenCV / ZXing Fallback
     |
     v
Decoded QR Payload
