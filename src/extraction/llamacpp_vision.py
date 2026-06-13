"""Shared llama.cpp vision helpers (GGUF + mmproj + MiniCPM chat handler)."""

from __future__ import annotations

import os
from functools import lru_cache

DEFAULT_MMPROJ_FILE = "mmproj-model-f16.gguf"
DEFAULT_CHAT_HANDLER = "MiniCPMv26ChatHandler"


def llamacpp_vision_enabled() -> bool:
    return os.getenv("LLAMACPP_VISION", "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=8)
def download_hf_file(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename)


@lru_cache(maxsize=4)
def load_vision_llama(
    model_path: str,
    mmproj_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    handler_name: str,
):
    """Load a MiniCPM-V GGUF with its vision projector."""
    try:
        from llama_cpp import Llama
        from llama_cpp import llama_chat_format
    except ImportError as exc:  # pragma: no cover - optional heavy dep
        raise ImportError(
            "llama-cpp-python is not installed. Install it (see requirements.txt) to use the "
            "llama.cpp vision backend."
        ) from exc

    handler_cls = getattr(llama_chat_format, handler_name, None)
    if handler_cls is None:
        raise RuntimeError(
            f"Chat handler '{handler_name}' not found in llama_cpp.llama_chat_format. "
            "Set LLAMACPP_CHAT_HANDLER to the handler matching your MiniCPM-V build."
        )
    chat_handler = handler_cls(clip_model_path=mmproj_path)
    return Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
