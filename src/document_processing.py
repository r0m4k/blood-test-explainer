from __future__ import annotations

from pathlib import Path

import fitz


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".csv"}
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | {".pdf"}


def validate_upload(path: str) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError("Uploaded file could not be found.")

    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type `{extension}`. Supported types: {allowed}.")

    return file_path


def document_to_payload_parts(path: str, max_pages: int | None = None) -> list[dict]:
    file_path = validate_upload(path)
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return [{"type": "text", "text": _pdf_to_text(file_path, max_pages=max_pages)}]

    if extension in SUPPORTED_TEXT_EXTENSIONS:
        return [{"type": "text", "text": _read_text_file(file_path)}]

    raise ValueError(f"Unsupported file type `{extension}`.")


def _pdf_to_text(file_path: Path, max_pages: int | None) -> str:
    chunks: list[str] = []
    with fitz.open(file_path) as document:
        if document.page_count == 0:
            raise ValueError("The uploaded PDF does not contain any pages.")

        pages_to_read = document.page_count if max_pages is None else min(document.page_count, max_pages)
        for page_index in range(pages_to_read):
            page = document.load_page(page_index)
            text = page.get_text("text").strip()
            if text:
                chunks.append(f"[Page {page_index + 1}]\n{text}")

    text = "\n\n".join(chunks).strip()
    if not text:
        raise ValueError(
            "The uploaded PDF does not contain extractable text. "
            "Please use a text-based PDF rather than a scanned image."
        )
    return text[:30000]


def _read_text_file(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError("The uploaded text file is empty.")
    return text[:20000]
