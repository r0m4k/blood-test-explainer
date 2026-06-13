"""Backend selection.

`EXTRACTOR_BACKEND` env:
  - `transformers` (default): local OpenBMB MiniCPM-V through Transformers.
  - `auto`: same as `transformers`.
  - `zerogpu` / `zero-gpu`: alias for `transformers`.
  - `llamacpp-gpu` / `llama-champion`: llama.cpp GGUF badge path. Add `LLAMACPP_VISION=1`
    for PDF/image vision via mmproj.
  - `local` / `server`: local llama-server backend for local development.
  - `llamacpp`: in-process llama-cpp-python backend for local development.

The hosted OpenBMB HTTP API is disabled in this project.
"""

from __future__ import annotations

import os

from src.extraction.base import Extractor
from src.extraction.auto import AutoExtractor
from src.extraction.llamacpp_gpu import LlamaCppGPUExtractor
from src.extraction.local_minicpmv import LocalMiniCPMVExtractor
from src.extraction.local_server import LocalServerExtractor

_DEFAULT_BACKEND = "transformers"
_DISABLED_BACKENDS = {"api", "openbmb", "hosted"}


def build_extractor(model: str | None = None) -> Extractor:
    backend = os.getenv("EXTRACTOR_BACKEND", _DEFAULT_BACKEND).strip().lower()

    if backend in _DISABLED_BACKENDS:
        raise ValueError(
            "The hosted OpenBMB API backend is disabled. "
            "Use EXTRACTOR_BACKEND=transformers for local MiniCPM-V extraction."
        )

    if backend in ("auto", "zerogpu", "zero-gpu", "transformers"):
        return AutoExtractor(model_id=model)
    if backend in ("llamacpp-gpu", "gpu-llamacpp", "llama-champion"):
        from src.extraction.llamacpp_vision import llamacpp_vision_enabled

        if llamacpp_vision_enabled():
            print(
                "[Blood Test Explainer] using llama.cpp vision extractor "
                f"(repo={os.getenv('LLAMACPP_GGUF_REPO', 'openbmb/MiniCPM-V-4.6-gguf')})",
                flush=True,
            )
        return LlamaCppGPUExtractor()
    if backend in ("local", "server", "local-server"):
        return LocalServerExtractor()
    if backend == "llamacpp":
        return LocalMiniCPMVExtractor()
    raise ValueError(f"Unknown EXTRACTOR_BACKEND: {backend}")
