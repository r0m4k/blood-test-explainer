"""Adaptive extraction backend selection."""

from __future__ import annotations

import os

from src.extraction.base import Extractor
from src.extraction.llamacpp_gpu import LlamaCppGPUExtractor
from src.extraction.zerogpu_transformers import ZeroGPUTransformersExtractor


class AutoExtractor:
    """Use Transformers when CUDA is visible, otherwise fall back to CPU llama.cpp."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id
        self._selected: Extractor | None = None

    def extract(self, file_path: str, max_pages: int = 3):
        backend = self._backend()
        try:
            return backend.extract(file_path, max_pages=max_pages)
        except Exception as exc:
            if not isinstance(backend, ZeroGPUTransformersExtractor) or not _fallback_enabled():
                raise

            print(
                "[Blood Test Explainer] CUDA Transformers backend failed; "
                f"falling back to CPU llama.cpp. Inner error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            self._selected = LlamaCppGPUExtractor()
            return self._selected.extract(file_path, max_pages=max_pages)

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


def _fallback_enabled() -> bool:
    return os.getenv("AUTO_FALLBACK_TO_LLAMACPP", "1").strip().lower() not in {"0", "false", "no"}

    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False
