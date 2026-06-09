"""Backend selection.

`EXTRACTOR_BACKEND` env:
  - `auto` (default): local-server if LLAMA_SERVER_URL is set; else the in-process llama.cpp
    backend if a GGUF + llama-cpp-python are available; else the hosted API.
  - `local` / `server`: the offline **llama-server** backend (the path that works for MiniCPM-V
    4.6). Run `llama-server -m model.gguf --mmproj mmproj.gguf --port 8080` alongside the app.
  - `llamacpp`: in-process llama-cpp-python (only once it supports the model build).
  - `api`: the hosted OpenBMB endpoint (dev fallback only).
"""

from __future__ import annotations

import os

from src.extraction.base import Extractor
from src.extraction.local_minicpmv import LocalMiniCPMVExtractor
from src.extraction.local_server import LocalServerExtractor
from src.openbmb_client import OpenBMBExtractor


def build_extractor(
    api_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> Extractor:
    backend = os.getenv("EXTRACTOR_BACKEND", "auto").strip().lower()

    if backend == "api":
        return OpenBMBExtractor(api_url=api_url, model=model, api_key=api_key)
    if backend in ("local", "server", "local-server"):
        return LocalServerExtractor()
    if backend == "llamacpp":
        return LocalMiniCPMVExtractor()

    # auto
    if os.getenv("LLAMA_SERVER_URL"):
        return LocalServerExtractor()
    if _llamacpp_available():
        try:
            return LocalMiniCPMVExtractor()
        except Exception:
            pass
    return OpenBMBExtractor(api_url=api_url, model=model, api_key=api_key)


def _llamacpp_available() -> bool:
    if not (os.getenv("LOCAL_MODEL_PATH") and os.getenv("LOCAL_MMPROJ_PATH")):
        return False
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return True
