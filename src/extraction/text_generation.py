"""Shared text-generation helpers for chat (mirrors EXTRACTOR_BACKEND)."""

from __future__ import annotations

import os

from src.extraction.llamacpp_gpu import DEFAULT_GGUF_REPO, DEFAULT_MODEL_FILE
from src.local_env import load_local_env

load_local_env()


def generate_text_chat(messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
    """Run a text-only chat completion using the configured extraction backend family."""
    backend = os.getenv("EXTRACTOR_BACKEND", "transformers").strip().lower()
    token_limit = max_tokens or int(os.getenv("CHAT_MAX_TOKENS", "1024"))

    if backend in {"api", "openbmb", "hosted"}:
        raise RuntimeError(
            "Chat via the hosted OpenBMB API is disabled. Use EXTRACTOR_BACKEND=transformers."
        )
    if backend in {"auto", "zerogpu", "zero-gpu", "transformers"}:
        return _transformers_chat(messages, token_limit)
    if backend in {"llamacpp-gpu", "gpu-llamacpp", "llama-champion", "llamacpp"}:
        return _llamacpp_chat(messages, token_limit)

    raise RuntimeError(f"Chat is not configured for EXTRACTOR_BACKEND={backend!r}.")


def _transformers_chat(messages: list[dict[str, str]], max_tokens: int) -> str:
    from src.extraction.zerogpu_transformers import _run_zerogpu_generation
    from src.model_paths import resolve_transformers_model_source

    model_source = resolve_transformers_model_source(os.getenv("ZEROGPU_MODEL_ID"))
    downsample_mode = (os.getenv("ZEROGPU_DOWNSAMPLE_MODE") or "16x").strip()
    structured = [{"role": m["role"], "content": m["content"]} for m in messages]
    return _run_zerogpu_generation(
        messages=structured,
        model_source=model_source,
        max_new_tokens=max_tokens,
        downsample_mode=downsample_mode,
    )


def _llamacpp_chat(messages: list[dict[str, str]], max_tokens: int) -> str:
    from src.extraction.llamacpp_gpu import _run_llamacpp_chat

    repo = os.getenv("LLAMACPP_GGUF_REPO", DEFAULT_GGUF_REPO).strip()
    model_file = os.getenv("LLAMACPP_MODEL_FILE", DEFAULT_MODEL_FILE).strip()
    n_ctx = int(os.getenv("LLAMACPP_N_CTX", "8192"))
    n_gpu_layers = int(os.getenv("LLAMACPP_N_GPU_LAYERS", "0"))
    return _run_llamacpp_chat(
        messages=messages,
        repo=repo,
        model_file=model_file,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
    )
