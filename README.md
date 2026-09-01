# AI-Based Identity Document Verification and QR Detection System

A web-based AI-powered system for identity document verification, OCR, QR code detection and decoding, Aadhaar Secure QR handling, payload classification, and information cross-validation.

The system combines document processing, OCR, QR detection, multi-stage QR decoding, Aadhaar Secure QR processing, and structured result generation. It is designed to analyse uploaded identity documents and present extracted and decoded information in a structured format.

The system processes document images and PDF files. Its QR module uses a multi-stage detection and decoding pipeline to handle real-world conditions such as rotation, perspective distortion, low resolution, dense QR payloads, uneven lighting, noise, and camera-captured images.

## Features

- Safe file validation and PDF page rendering
- Document image processing and OpenCV preprocessing
- Aadhaar document detection using text anchors and number patterns
- Bilingual OCR support for English and Hindi
- Word-level OCR bounding boxes and confidence scoring
- QR code detection and decoding
- Single and multiple QR code support
- Detection using OpenCV `QRCodeDetectorAruco` and `QRCodeDetector`
- Support for text and binary QR payloads
- Robust decoding for rotated, skewed, dim, noisy, blurry, and low-resolution QR codes
- Perspective correction and multi-scale QR processing
- Image padding and quiet-zone handling
- CLAHE contrast enhancement
- Otsu and adaptive thresholding
- Image sharpening and inversion
- QR-region cropping and upscaling
- OpenCV-based multi-stage decoding
- Optional ZXing-C++ fallback decoding
- QR bounding boxes and geometry extraction
- Coordinate mapping to the original image
- Duplicate QR detection removal
- Aadhaar and e-Aadhaar Secure QR identification
- Raw QR payload preservation
- Payload classification
- OCR and QR information cross-validation
- Structured JSON results
- Automated backend testing

---

## System Architecture

```text
                    Frontend
                 React + Vite
                       |
                       | HTTP
                       v
                 FastAPI Backend
                       |
                       v
                Document Processor
                       |
          +------------+------------+
          |                         |
          v                         v
      OCR Pipeline              QR Processor
          |                         |
          |              +----------+----------+
          |              |                     |
          v              v                     v
   Document Details   QR Detector          QR Decoder
                         OpenCV            Multi-stage
                                              |
                                              v
                                       Payload Parser
                                              |
                                              v
                                      Result / Validation
                                              |
                                              v
                                     Human-readable UI
