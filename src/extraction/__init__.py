"""Extraction backends behind one interface.

`build_extractor()` returns the right backend for the environment:
  - **auto**: Transformers when CUDA is visible; CPU llama.cpp otherwise.
  - **zerogpu** / **transformers**: force official OpenBMB MiniCPM-V through Transformers.
  - **llamacpp-gpu** / **llama-champion**: force GGUF through llama.cpp.
  - **local**: local llama-server / llama.cpp backends for local experimentation.
  - **api**: the original OpenBMB hosted endpoint, kept as a dev fallback only.

Default is `auto`.
"""

from src.extraction.base import Extractor, ExtractionResult
from src.extraction.factory import build_extractor

__all__ = ["Extractor", "ExtractionResult", "build_extractor"]
