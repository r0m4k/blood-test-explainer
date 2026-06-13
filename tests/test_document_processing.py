import sys
import tempfile
from pathlib import Path

import fitz
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.document_processing import document_intake_metadata, document_to_payload_parts, validate_upload


def test_png_upload_returns_image_url_part():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
        Image.new("RGB", (32, 32), color="white").save(path)

    parts = document_to_payload_parts(path)
    assert len(parts) == 1
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_jpeg_upload_returns_image_url_part():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        path = tmp.name
        Image.new("RGB", (24, 24), color="red").save(path, format="JPEG")

    parts = document_to_payload_parts(path)
    assert len(parts) == 1
    assert parts[0]["type"] == "image_url"


def test_pdf_upload_renders_pages_to_images():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        path = tmp.name
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Hemoglobin 12.5 g/dL")
        document.save(path)
        document.close()

    parts = document_to_payload_parts(path, max_pages=1)
    assert len(parts) == 1
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_text_upload_still_returns_text_part():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write("Hemoglobin 13.1 g/dL")
        path = tmp.name

    parts = document_to_payload_parts(path)
    assert len(parts) == 1
    assert parts[0]["type"] == "text"
    assert "Hemoglobin" in parts[0]["text"]


def test_validate_upload_rejects_unknown_extension():
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        path = tmp.name
    try:
        validate_upload(path)
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert "Unsupported file type" in str(error)


def test_document_intake_metadata_for_pdf():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        path = tmp.name
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Sample")
        document.save(path)
        document.close()

    parts = document_to_payload_parts(path, max_pages=1)
    metadata = document_intake_metadata(path, parts)
    assert metadata["input_modality"] == "vision"
    assert metadata["pages_rendered"] == 1
    assert metadata["image_count"] == 1


if __name__ == "__main__":
    test_png_upload_returns_image_url_part()
    test_jpeg_upload_returns_image_url_part()
    test_pdf_upload_renders_pages_to_images()
    test_text_upload_still_returns_text_part()
    test_validate_upload_rejects_unknown_extension()
    test_document_intake_metadata_for_pdf()
    print("test_document_processing: ok")
