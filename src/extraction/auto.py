"""Adaptive extraction backend selection."""

from __future__ import annotations

import os

from src.extraction.base import Extractor
from src.extraction.zerogpu_transformers import ZeroGPUTransformersExtractor


class AutoExtractor:
    """Use the local Transformers MiniCPM-V path."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id
        self._selected: Extractor | None = None

    def extract(self, file_path: str, max_pages: int = 3):
        return self._backend().extract(file_path, max_pages=max_pages)

    def _backend(self) -> Extractor:
        if self._selected is None:
            from src.model_paths import resolve_transformers_model_source

            source = resolve_transformers_model_source(self.model_id)
            print(
                "[Blood Test Explainer] using Transformers extractor "
                f"(origin={source.origin}, model={source.model_id})",
                flush=True,
            )
            self._selected = ZeroGPUTransformersExtractor(model_id=self.model_id)
        return self._selected

