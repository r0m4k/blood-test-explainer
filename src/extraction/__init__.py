"""Extraction backends behind one interface.

`build_extractor()` returns the right backend for the environment:
  - **zerogpu** / **auto**: official OpenBMB MiniCPM-V through Transformers on HF ZeroGPU.
  - **local**: local llama-server / llama.cpp backends for local experimentation.
  - **api**: the original OpenBMB hosted endpoint, kept as a dev fallback only.

Default is `auto`, which resolves to the ZeroGPU Transformers backend for the Space.
"""

from src.extraction.base import Extractor, ExtractionResult
from src.extraction.factory import build_extractor

__all__ = ["Extractor", "ExtractionResult", "build_extractor"]
