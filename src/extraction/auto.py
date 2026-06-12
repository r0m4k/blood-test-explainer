"""Adaptive extraction backend selection."""

from __future__ import annotations

import os

from src.extraction.base import Extractor
from src.extraction.llamacpp_gpu import LlamaCppGPUExtractor
from src.extraction.zerogpu_transformers import ZeroGPUTransformersExtractor


class AutoExtractor:
    """Use Transformers on ZeroGPU/CUDA, otherwise use CPU llama.cpp."""

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
            target = runtime_target()
            print(f"[Blood Test Explainer] auto extractor selected {target}", flush=True)
            if target == "transformers":
                self._selected = ZeroGPUTransformersExtractor(model_id=self.model_id)
            else:
                self._selected = LlamaCppGPUExtractor()
        return self._selected


def runtime_target() -> str:
    """Return `transformers` for ZeroGPU/CUDA and `llamacpp` for CPU-only runtime."""
    if zerogpu_runtime_requested() or cuda_available():
        return "transformers"
    return "llamacpp"


def zerogpu_runtime_requested() -> bool:
    """Detect HF ZeroGPU from explicit Space/runtime environment flags.

    ZeroGPU exposes CUDA only inside a `@spaces.GPU` worker, so checking
    `torch.cuda.is_available()` in normal Gradio app code is not enough.
    """
    boolean_flags = ("ZERO_GPU", "SPACES_ZERO_GPU", "HF_ZERO_GPU", "BTE_ZERO_GPU")
    for name in boolean_flags:
        value = os.getenv(name, "").strip().lower()
        if value in {"1", "true", "yes", "on", "zerogpu", "zero-gpu"}:
            return True

    hardware_flags = ("ACCELERATOR", "BTE_RUNTIME", "BTE_HARDWARE", "SPACE_HARDWARE", "HF_SPACE_HARDWARE")
    for name in hardware_flags:
        value = os.getenv(name, "").strip().lower()
        if "zero" in value and "gpu" in value:
            return True
        if value.startswith("zero-"):
            return True

    return False


def cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _fallback_enabled() -> bool:
    return os.getenv("AUTO_FALLBACK_TO_LLAMACPP", "0").strip().lower() in {"1", "true", "yes", "on"}
