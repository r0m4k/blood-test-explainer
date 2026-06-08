from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path

import fitz
from PIL import Image, ImageOps


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".csv"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_TEXT_EXTENSIONS | {".pdf"}


def validate_upload(path: str) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError("Uploaded file could not be found.")

    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type `{extension}`. Supported types: {allowed}.")

    return file_path


def document_to_payload_parts(path: str, max_pages: int = 3) -> list[dict]:
    file_path = validate_upload(path)
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return _pdf_to_image_parts(file_path, max_pages=max_pages)

    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        return [_image_to_payload_part(file_path)]

    if extension in SUPPORTED_TEXT_EXTENSIONS:
        return [{"type": "text", "text": _read_text_file(file_path)}]

    raise ValueError(f"Unsupported file type `{extension}`.")


def _pdf_to_image_parts(file_path: Path, max_pages: int) -> list[dict]:
    parts: list[dict] = []
    with fitz.open(file_path) as document:
        if document.page_count == 0:
            raise ValueError("The uploaded PDF does not contain any pages.")

        pages_to_read = min(document.page_count, max_pages)
        for page_index in range(pages_to_read):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            parts.append(_image_to_data_url_part(image, "image/jpeg"))

    return parts


def _image_to_payload_part(file_path: Path) -> dict:
    with Image.open(file_path) as image:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "image/jpeg"
        if mime_type == "image/tiff":
            mime_type = "image/jpeg"
        return _image_to_data_url_part(image, mime_type)


def _image_to_data_url_part(image: Image.Image, mime_type: str) -> dict:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((1800, 1800))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def _read_text_file(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError("The uploaded text file is empty.")
    return text[:20000]
