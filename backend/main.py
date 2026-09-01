"""
FastAPI Backend Application for Standalone QR Code Detection & Decoding.
"""

from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .qr.processor import QRProcessor
from .utils.image_handler import bytes_to_cv2_image, cv2_to_base64_data_url
from .utils.pdf_handler import render_pdf_page_to_cv2_image
from .utils.validation import validate_uploaded_file

# Initialize FastAPI application
app = FastAPI(
    title="Standalone QR Code Detection & Decoding API",
    description="High-performance, modular QR detection and decoding service using OpenCV and PyMuPDF.",
    version="1.0.0",
)

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global QR Processor instance
qr_processor = QRProcessor()


@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify backend status."""
    return {
        "status": "ok",
        "service": "QR Code Detection & Decoding Engine",
        "version": "1.0.0",
        "engine": "OpenCV QRCodeDetector",
    }


@app.post("/api/qr/scan")
async def scan_qr_code(
    file: UploadFile = File(...),
    page: Optional[int] = Form(0),
):
    """
    Scan an uploaded Image or PDF document for QR codes.

    Accepts:
        file: Multipart file upload (.png, .jpg, .jpeg, .webp, .bmp, .tiff, .pdf)
        page: Optional 0-indexed page number for multi-page PDF documents.

    Returns:
        Structured JSON with detected QR status, coordinates, decoded payloads, and warnings.
    """
    try:
        # Read file bytes into memory buffer (no disk storage)
        file_bytes = await file.read()
        filename = file.filename or "uploaded_document"

        # Validate file size, extension, and magic bytes
        is_valid, file_type, val_error = validate_uploaded_file(filename, file_bytes)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "qr_detected": False,
                    "qr_count": 0,
                    "codes": [],
                    "warnings": [],
                    "errors": [val_error or "File validation failed."],
                },
            )

        page_number = max(0, page or 0)
        page_metadata = {}

        if file_type == "pdf":
            # Render PDF page to OpenCV BGR at true ~300 DPI
            try:
                cv2_img, total_pages, current_page = render_pdf_page_to_cv2_image(
                    file_bytes, page_number=page_number, dpi=300
                )
                # Generate preview ONLY for PDF visualization so frontend can overlay bounding boxes
                pdf_preview_data_url = cv2_to_base64_data_url(cv2_img, max_dim=1600, quality=85)
                page_metadata = {
                    "page_count": total_pages,
                    "current_page": current_page + 1,  # 1-indexed for display
                    "preview_image": pdf_preview_data_url,
                }
            except Exception as pdf_err:
                return JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    content={
                        "success": False,
                        "qr_detected": False,
                        "qr_count": 0,
                        "codes": [],
                        "warnings": [],
                        "errors": [f"Failed to process PDF page: {str(pdf_err)}"],
                    },
                )
        else:
            # Process standard image format
            try:
                cv2_img = bytes_to_cv2_image(file_bytes)
                page_metadata = {
                    "page_count": 1,
                    "current_page": 1,
                    "preview_image": None,  # Frontend uses its local object URL for image preview
                }
            except Exception as img_err:
                return JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    content={
                        "success": False,
                        "qr_detected": False,
                        "qr_count": 0,
                        "codes": [],
                        "warnings": [],
                        "errors": [f"Failed to decode image data: {str(img_err)}"],
                    },
                )

        # Run QR detection and decoding pipeline
        result = qr_processor.process_image(
            cv2_img=cv2_img,
            filename=filename,
            file_type=file_type,
            page_metadata=page_metadata,
        )

        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

    except Exception as general_err:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "qr_detected": False,
                "qr_count": 0,
                "codes": [],
                "warnings": [],
                "errors": [f"Internal server error: {str(general_err)}"],
            },
        )
