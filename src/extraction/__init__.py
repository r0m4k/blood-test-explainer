"""Extraction backends behind one interface.

`build_extractor()` returns the right backend for the environment:
  - **local** (off-grid): fine-tuned MiniCPM-V as GGUF under llama.cpp, fully on-device.
  - **api**: the original OpenBMB hosted endpoint (kept as a dev fallback only).

Default is `auto`: use local when a GGUF is configured + llama.cpp is importable, else api.
"""

from src.extraction.base import Extractor, ExtractionResult
from src.extraction.factory import build_extractor

__all__ = ["Extractor", "ExtractionResult", "build_extractor"]
