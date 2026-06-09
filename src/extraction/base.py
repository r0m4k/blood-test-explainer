"""The extractor interface shared by every backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Reuse the existing result dataclass so there is exactly one definition in the codebase.
from src.openbmb_client import ExtractionResult

__all__ = ["Extractor", "ExtractionResult"]


@runtime_checkable
class Extractor(Protocol):
    """Anything that turns an uploaded document into structured lab values."""

    def extract(self, file_path: str, max_pages: int = 3) -> ExtractionResult: ...
