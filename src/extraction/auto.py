"""Adaptive extraction backend selection."""

from __future__ import annotations

from src.extraction.base import Extractor
from src.extraction.llamacpp_gpu import LlamaCppGPUExtractor
from src.extraction.zerogpu_transformers import ZeroGPUTransformersExtractor


class AutoExtractor:
    """Use Transformers when CUDA is visible, otherwise fall back to CPU llama.cpp."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id
        self._selected: Extractor | None = None

    def extract(self, file_path: str, max_pages: int = 3):
        return self._backend().extract(file_path, max_pages=max_pages)

    def _backend(self) -> Extractor:
        if self._selected is None:
            if cuda_available():
                self._selected = ZeroGPUTransformersExtractor(model_id=self.model_id)
            else:
                self._selected = LlamaCppGPUExtractor()
        return self._selected


def cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False
