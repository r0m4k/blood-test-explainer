"""Extraction backends behind one interface.

`build_extractor()` returns the right backend for the environment:
  - **transformers** (default): local OpenBMB MiniCPM-V through Transformers.
  - **auto**: same as transformers.
  - **llamacpp-gpu** / **llama-champion**: force GGUF through llama.cpp.
  - **local**: local llama-server / llama.cpp backends for local experimentation.

The hosted OpenBMB HTTP API is disabled.
"""

from src.extraction.base import Extractor, ExtractionResult
from src.extraction.factory import build_extractor

__all__ = ["Extractor", "ExtractionResult", "build_extractor"]
