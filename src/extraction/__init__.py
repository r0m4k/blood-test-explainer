"""Extraction backends behind one interface.

`build_extractor()` returns the right backend for the environment:
  - unset / **auto** (default): CPU Basic Spaces use llama.cpp + base GGUF; other runtimes use Transformers.
  - **transformers**: force OpenBMB MiniCPM-V through Transformers.
  - **llamacpp-gpu** / **llama-champion**: force GGUF through llama.cpp. Set `LLAMACPP_VISION=1`
    to run the same PDF/image vision pipeline as Transformers (requires mmproj).
  - **local**: local llama-server / llama.cpp backends for local experimentation.

The hosted OpenBMB HTTP API is disabled.
"""

from src.extraction.base import Extractor, ExtractionResult
from src.extraction.factory import build_extractor

__all__ = ["Extractor", "ExtractionResult", "build_extractor"]
