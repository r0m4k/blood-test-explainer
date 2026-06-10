"""Backend selection.

`EXTRACTOR_BACKEND` env:
  - `auto` / `zerogpu` (default): HF ZeroGPU + official OpenBMB Transformers model.
  - `api`: hosted OpenBMB endpoint (dev fallback only).
  - `local` / `server`: local llama-server backend for local development.
  - `llamacpp`: in-process llama-cpp-python backend for local development.
"""

from __future__ import annotations

import os

from src.extraction.base import Extractor
from src.extraction.local_minicpmv import LocalMiniCPMVExtractor
from src.extraction.local_server import LocalServerExtractor
from src.extraction.zerogpu_transformers import ZeroGPUTransformersExtractor
from src.openbmb_client import OpenBMBExtractor


def build_extractor(
    api_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> Extractor:
    backend = os.getenv("EXTRACTOR_BACKEND", "auto").strip().lower()

    if backend in ("auto", "zerogpu", "zero-gpu", "transformers"):
        return ZeroGPUTransformersExtractor(model_id=model)
    if backend == "api":
        return OpenBMBExtractor(api_url=api_url, model=model, api_key=api_key)
    if backend in ("local", "server", "local-server"):
        return LocalServerExtractor()
    if backend == "llamacpp":
        return LocalMiniCPMVExtractor()
    raise ValueError(f"Unknown EXTRACTOR_BACKEND: {backend}")


def _llamacpp_available() -> bool:
    if not (os.getenv("LOCAL_MODEL_PATH") and os.getenv("LOCAL_MMPROJ_PATH")):
        return False
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return True


def _in_process_local_configured() -> bool:
    return bool(os.getenv("LOCAL_MODEL_PATH") and os.getenv("LOCAL_MMPROJ_PATH"))
