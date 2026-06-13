from __future__ import annotations

import base64
import io
from pathlib import Path

import fitz

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".csv"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}

_DEFAULT_MAX_PAGES = 3
_PDF_RENDER_MATRIX = fitz.Matrix(2.5, 2.5)
_MAX_IMAGE_EDGE = 2048


def validate_upload(path: str) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError("Uploaded file could not be found.")

    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type `{extension}`. Supported types: {allowed}.")

    return file_path


def document_intake_metadata(path: str, parts: list[dict]) -> dict[str, object]:
    """Lightweight intake stats for traces (no base64 payloads)."""
    extension = Path(path).suffix.lower()
    image_count = sum(1 for part in parts if part.get("type") == "image_url")
    text_characters = sum(len(str(part.get("text") or "")) for part in parts if part.get("type") == "text")
    return {
        "source_extension": extension,
        "input_modality": "vision" if image_count else "text",
        "pages_rendered": image_count if extension == ".pdf" else None,
        "image_count": image_count,
        "text_characters": text_characters,
    }


def document_to_payload_parts(path: str, max_pages: int | None = None) -> list[dict]:
    """Build OpenAI-compatible message parts for vision extraction."""
    file_path = validate_upload(path)
    extension = file_path.suffix.lower()
    page_limit = _DEFAULT_MAX_PAGES if max_pages is None else max_pages

    if extension == ".pdf":
        return _pdf_to_image_parts(file_path, max_pages=page_limit)

    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        return [_image_part(file_path)]

    if extension in SUPPORTED_TEXT_EXTENSIONS:
        return [{"type": "text", "text": _read_text_file(file_path)}]

    raise ValueError(f"Unsupported file type `{extension}`.")


def _pdf_to_image_parts(file_path: Path, max_pages: int) -> list[dict]:
    parts: list[dict] = []
    with fitz.open(file_path) as document:
        if document.page_count == 0:
            raise ValueError("The uploaded PDF does not contain any pages.")

        pages_to_render = min(document.page_count, max(1, max_pages))
        for page_index in range(pages_to_render):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=_PDF_RENDER_MATRIX, alpha=False)
            encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )

    return parts


def _image_part(file_path: Path) -> dict:
    from PIL import Image, ImageOps

    with Image.open(file_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((_MAX_IMAGE_EDGE, _MAX_IMAGE_EDGE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
    }


def _read_text_file(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError("The uploaded text file is empty.")
    return text[:20000]
