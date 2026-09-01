"""
Comprehensive unit tests for file validation and safe loading layer.
Covers all 8 requirements of Prompt 2:
1. Valid JPG
2. Valid PNG
3. Valid PDF
4. Missing file
5. Unsupported extension
6. Oversized file
7. Corrupted image
8. Invalid PDF
"""

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from document_processor.file_handler import (
    FileValidationError,
    LoadedDocument,
    cleanup_temp_files,
    load_document,
    validate_file,
    validate_file_input,
)
from tests.test_helpers import (
    create_synthetic_aadhaar_bytes,
    create_synthetic_aadhaar_pdf,
)


class TestFileValidationAndLoading(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Valid JPG
    def test_1_valid_jpg(self):
        jpg_bytes = create_synthetic_aadhaar_bytes(fmt="jpeg")
        jpg_file = self.temp_path / "test_doc.jpg"
        jpg_file.write_bytes(jpg_bytes)

        # Test validate_file
        val_res = validate_file(jpg_file)
        self.assertTrue(val_res["success"])
        self.assertTrue(val_res["valid"])
        self.assertEqual(val_res["file_type"], "image")
        self.assertEqual(val_res["format"], "jpeg")
        self.assertGreater(val_res["file_size_bytes"], 0)

        # Test load_document
        load_res = load_document(jpg_file)
        self.assertTrue(load_res["success"])
        self.assertEqual(load_res["file_type"], "image")
        self.assertEqual(load_res["format"], "jpeg")
        self.assertEqual(load_res["pages"], 1)
        self.assertEqual(len(load_res["images"]), 1)
        self.assertIsInstance(load_res["images"][0], np.ndarray)
        self.assertEqual(load_res["metadata"]["channels"], 3)
        self.assertGreater(load_res["metadata"]["width"], 0)
        self.assertGreater(load_res["metadata"]["height"], 0)

    # 2. Valid PNG
    def test_2_valid_png(self):
        png_bytes = create_synthetic_aadhaar_bytes(fmt="png")
        png_file = self.temp_path / "test_doc.png"
        png_file.write_bytes(png_bytes)

        val_res = validate_file(png_file)
        self.assertTrue(val_res["success"])
        self.assertEqual(val_res["format"], "png")

        load_res = load_document(png_file)
        self.assertTrue(load_res["success"])
        self.assertEqual(load_res["format"], "png")
        self.assertEqual(len(load_res["images"]), 1)

    # 3. Valid PDF
    def test_3_valid_pdf(self):
        pdf_bytes = create_synthetic_aadhaar_pdf()
        pdf_file = self.temp_path / "test_doc.pdf"
        pdf_file.write_bytes(pdf_bytes)

        val_res = validate_file(pdf_file)
        self.assertTrue(val_res["success"])
        self.assertEqual(val_res["file_type"], "pdf")
        self.assertEqual(val_res["format"], "pdf")

        load_res = load_document(pdf_file, dpi=300)
        self.assertTrue(load_res["success"])
        self.assertEqual(load_res["file_type"], "pdf")
        self.assertEqual(load_res["format"], "pdf")
        self.assertEqual(load_res["pages"], 1)
        self.assertEqual(load_res["metadata"]["dpi"], 300)
        self.assertIsInstance(load_res["images"][0], np.ndarray)

    # 4. Missing file
    def test_4_missing_file(self):
        missing_path = self.temp_path / "non_existent_file.png"

        val_res = validate_file(missing_path)
        self.assertFalse(val_res["success"])
        self.assertFalse(val_res["valid"])
        self.assertEqual(val_res["error"]["code"], "FILE_NOT_FOUND")

        load_res = load_document(missing_path)
        self.assertFalse(load_res["success"])
        self.assertEqual(load_res["error"]["code"], "FILE_NOT_FOUND")
        # Ensure no internal filesystem full path is leaked in message
        self.assertNotIn(str(self.temp_path), load_res["error"]["message"])

    # 5. Unsupported extension
    def test_5_unsupported_extension(self):
        doc_file = self.temp_path / "document.docx"
        doc_file.write_bytes(b"PK\x03\x04mock docx content")

        val_res = validate_file(doc_file)
        self.assertFalse(val_res["success"])
        self.assertEqual(val_res["error"]["code"], "UNSUPPORTED_FILE_TYPE")

        load_res = load_document(doc_file)
        self.assertFalse(load_res["success"])
        self.assertEqual(load_res["error"]["code"], "UNSUPPORTED_FILE_TYPE")

    # 6. Oversized file
    def test_6_oversized_file(self):
        # 11 MB buffer with PDF header
        oversized_data = b"%PDF-1.4 " + b"0" * (11 * 1024 * 1024)

        # Test configurable max_size_bytes parameter
        val_res = validate_file(oversized_data, max_size_bytes=10 * 1024 * 1024)
        self.assertFalse(val_res["success"])
        self.assertEqual(val_res["error"]["code"], "FILE_TOO_LARGE")

        load_res = load_document(oversized_data, max_size_bytes=10 * 1024 * 1024)
        self.assertFalse(load_res["success"])
        self.assertEqual(load_res["error"]["code"], "FILE_TOO_LARGE")

    # 7. Corrupted image
    def test_7_corrupted_image(self):
        # Named .jpg but contents are completely invalid / corrupt
        fake_jpg = self.temp_path / "corrupt_image.jpg"
        fake_jpg.write_bytes(b"this is definitely not a valid jpeg or png file")

        val_res = validate_file(fake_jpg)
        self.assertFalse(val_res["success"])
        self.assertEqual(val_res["error"]["code"], "CORRUPTED_OR_INVALID_FILE")

        load_res = load_document(fake_jpg)
        self.assertFalse(load_res["success"])
        self.assertEqual(load_res["error"]["code"], "CORRUPTED_OR_INVALID_FILE")

    # 8. Invalid PDF
    def test_8_invalid_pdf(self):
        # Starts with %PDF- header but has corrupted body
        broken_pdf = self.temp_path / "broken.pdf"
        broken_pdf.write_bytes(b"%PDF-1.4\ncorrupted trailing binary data without xref")

        load_res = load_document(broken_pdf)
        self.assertFalse(load_res["success"])
        self.assertIn(load_res["error"]["code"], ["INVALID_PDF", "PDF_READ_ERROR"])

    # 9. In-memory buffer support (BytesIO and bytes)
    def test_in_memory_bytesio_support(self):
        png_bytes = create_synthetic_aadhaar_bytes(fmt="png")
        buffer = io.BytesIO(png_bytes)

        load_res = load_document(buffer)
        self.assertTrue(load_res["success"])
        self.assertEqual(load_res["format"], "png")
        self.assertEqual(len(load_res["images"]), 1)

    # 10. Temporary file cleanup helper
    def test_temp_cleanup_helper(self):
        test_temp = self.temp_path / "temp_to_delete.tmp"
        test_temp.write_bytes(b"temp data")
        self.assertTrue(test_temp.exists())

        cleanup_temp_files(test_temp)
        self.assertFalse(test_temp.exists())


if __name__ == "__main__":
    unittest.main()
