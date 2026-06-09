"""Backend selection.

`EXTRACTOR_BACKEND` env:
  - `auto` (default): local if a GGUF is configured + importable, otherwise the API backend.
    This keeps the app working today (API) and flips to fully-offline the moment the fine-tuned
    GGUF is bundled — no code change.
  - `local`: force the offline MiniCPM-V backend (errors if not configured).
  - `api`:   force the hosted OpenBMB endpoint (dev fallback only).
"""

from __future__ import annotations

import os

from src.extraction.base import Extractor
from src.extraction.local_minicpmv import LocalMiniCPMVExtractor
from src.openbmb_client import OpenBMBExtractor


def build_extractor(
    api_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> Extractor:
    backend = os.getenv("EXTRACTOR_BACKEND", "auto").strip().lower()

    if backend == "api":
        return OpenBMBExtractor(api_url=api_url, model=model, api_key=api_key)

    if backend == "local":
        return LocalMiniCPMVExtractor()

    # auto
    if _local_available():
        try:
            return LocalMiniCPMVExtractor()
        except Exception:
            pass  # fall through to API
    return OpenBMBExtractor(api_url=api_url, model=model, api_key=api_key)


def _local_available() -> bool:
    if not (os.getenv("LOCAL_MODEL_PATH") and os.getenv("LOCAL_MMPROJ_PATH")):
        return False
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return True
