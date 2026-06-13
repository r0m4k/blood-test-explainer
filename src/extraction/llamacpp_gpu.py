"""ZeroGPU extraction via the **llama.cpp** runtime (earns the 🦙 Llama Champion badge).

Text-only by default. Set LLAMACPP_VISION=1 to run the same PDF/image vision pipeline as
Transformers (GGUF + mmproj through llama-cpp-python).

Config (env):
  EXTRACTOR_BACKEND=llamacpp-gpu
  LLAMACPP_VISION=1                 enable vision (PDF/image uploads)
  LLAMACPP_GGUF_REPO                HF repo with GGUF weights
  LLAMACPP_MODEL_FILE               GGUF filename
  LLAMACPP_MMPROJ_FILE              mmproj filename (required when vision is on)
  LLAMACPP_CHAT_HANDLER             default MiniCPMv26ChatHandler
  LLAMACPP_MAX_TOKENS               default 3072
  LLAMACPP_N_CTX                    default 8192
  LLAMACPP_N_GPU_LAYERS             default 0 for CPU wheels; -1 on CUDA builds
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

from src.document_processing import document_intake_metadata, document_to_payload_parts
from src.extraction.llamacpp_vision import (
    DEFAULT_CHAT_HANDLER,
    DEFAULT_MMPROJ_FILE,
    download_hf_file,
    llamacpp_vision_enabled,
    load_vision_llama,
)
from src.openbmb_client import (
    EXTRACTION_PROMPT,
    ExtractionResult,
    _normalize_notes,
    _normalize_patient,
    _normalize_tests,
    _parse_json_response,
    summarize_document_parts,
)

DEFAULT_GGUF_REPO = "openbmb/MiniCPM-V-4.6-gguf"
DEFAULT_MODEL_FILE = "MiniCPM-V-4_6-Q4_K_M.gguf"

try:
    import spaces
except ImportError:  # Local dev without the HF Spaces package.
    class _SpacesFallback:
        @staticmethod
        def GPU(*_args: Any, **_kwargs: Any):
            def decorator(func):
                return func

            return decorator

    spaces = _SpacesFallback()  # type: ignore[assignment]


class LlamaCppGPUExtractor:
    """Extractor that runs the GGUF through llama.cpp on ZeroGPU."""

    def __init__(self) -> None:
        self.repo = os.getenv("LLAMACPP_GGUF_REPO", DEFAULT_GGUF_REPO).strip()
        self.model_file = os.getenv("LLAMACPP_MODEL_FILE", DEFAULT_MODEL_FILE).strip()
        self.mmproj_file = os.getenv("LLAMACPP_MMPROJ_FILE", DEFAULT_MMPROJ_FILE).strip()
        self.chat_handler = os.getenv("LLAMACPP_CHAT_HANDLER", DEFAULT_CHAT_HANDLER).strip()
        self.max_tokens = int(os.getenv("LLAMACPP_MAX_TOKENS", "3072"))
        self.n_ctx = int(os.getenv("LLAMACPP_N_CTX", "8192"))
        self.n_gpu_layers = int(os.getenv("LLAMACPP_N_GPU_LAYERS", "0"))
        self.vision_enabled = llamacpp_vision_enabled()
        if self.vision_enabled and not self.mmproj_file:
            raise ValueError("LLAMACPP_VISION=1 requires LLAMACPP_MMPROJ_FILE.")

    def extract(self, file_path: str, max_pages: int = 3) -> ExtractionResult:
        parts = document_to_payload_parts(file_path, max_pages=max_pages)
        started = time.perf_counter()
        if self.vision_enabled:
            raw = _run_llamacpp_vision_generation(
                parts=parts,
                repo=self.repo,
                model_file=self.model_file,
                mmproj_file=self.mmproj_file,
                chat_handler=self.chat_handler,
                max_tokens=self.max_tokens,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
            )
            backend = "llamacpp-gpu-vision"
            composed_prompt = None
        else:
            prompt_text = _compose_prompt(parts)
            raw = _run_llamacpp_generation(
                prompt_text=prompt_text,
                repo=self.repo,
                model_file=self.model_file,
                max_tokens=self.max_tokens,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
            )
            backend = "llamacpp-gpu"
            composed_prompt = prompt_text

        duration_ms = int((time.perf_counter() - started) * 1000)
        parsed = _parse_json_response(raw)
        summary = {
            "backend": backend,
            "repo": self.repo,
            "model": self.model_file,
            "vision_enabled": self.vision_enabled,
            "document_parts": len(parts),
            "max_pages": max_pages,
            "extraction_prompt": EXTRACTION_PROMPT,
            "user_message_preview": summarize_document_parts(parts),
            **document_intake_metadata(file_path, parts),
            "return_code": 0,
            "duration_ms": duration_ms,
        }
        if self.vision_enabled:
            summary["mmproj"] = self.mmproj_file
            summary["chat_handler"] = self.chat_handler
        else:
            summary["composed_prompt"] = composed_prompt

        return ExtractionResult(
            patient=_normalize_patient(parsed.get("patient", {})),
            tests=_normalize_tests(parsed.get("tests", [])),
            notes=_normalize_notes(parsed.get("notes", [])),
            raw_response=raw,
            request_summary=summary,
        )


@lru_cache(maxsize=1)
def _load_text(model_path: str, n_ctx: int, n_gpu_layers: int):
    from llama_cpp import Llama

    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


def _vision_messages(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": EXTRACTION_PROMPT}, *parts]}]


def _raise_generation_error(exc: Exception, *, vision: bool) -> RuntimeError:
    message = f"{type(exc).__name__}: {exc}"
    if "llama_decode returned -1" in message:
        detail = (
            "llama.cpp ran out of room while decoding the vision prompt. "
            if vision
            else "llama.cpp ran out of room while decoding the PDF text prompt. "
        )
        raise RuntimeError(
            detail
            + "Try increasing LLAMACPP_N_CTX, lowering the number of PDF pages, or trimming the "
            "input before sending it to the model."
        ) from exc
    model_label = "vision GGUF + mmproj" if vision else "text-only GGUF"
    raise RuntimeError(
        f"The llama.cpp backend could not complete extraction with the {model_label} model. "
        f"Inner error: {message}"
    ) from exc


@spaces.GPU(duration=600)
def _run_llamacpp_vision_generation(
    parts: list[dict[str, Any]],
    repo: str,
    model_file: str,
    mmproj_file: str,
    chat_handler: str,
    max_tokens: int,
    n_ctx: int,
    n_gpu_layers: int,
) -> str:
    try:
        model_path = download_hf_file(repo, model_file)
        mmproj_path = download_hf_file(repo, mmproj_file)
    except Exception as exc:
        raise RuntimeError(
            "llama.cpp download failed while preparing the vision GGUF assets: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        llm = load_vision_llama(model_path, mmproj_path, n_ctx, n_gpu_layers, chat_handler)
    except Exception as exc:
        raise RuntimeError(
            "The llama.cpp backend could not load the vision GGUF + mmproj model. "
            "This usually means the downloaded model build is incompatible with the installed "
            "llama-cpp-python wheel or the model files are incomplete. "
            f"Inner error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        response = llm.create_chat_completion(
            messages=_vision_messages(parts),
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"].get("content") or "{}"
    except Exception as exc:
        raise _raise_generation_error(exc, vision=True) from exc


@spaces.GPU(duration=600)
def _run_llamacpp_generation(
    prompt_text: str,
    repo: str,
    model_file: str,
    max_tokens: int,
    n_ctx: int,
    n_gpu_layers: int,
) -> str:
    try:
        model_path = download_hf_file(repo, model_file)
    except Exception as exc:
        raise RuntimeError(
            "llama.cpp download failed while preparing the GGUF model: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        llm = _load_text(model_path, n_ctx, n_gpu_layers)
    except Exception as exc:
        raise RuntimeError(
            "The llama.cpp backend could not load the text-only GGUF model. "
            "This usually means the downloaded model build is incompatible with the installed "
            "llama-cpp-python wheel or the model file is incomplete. "
            f"Inner error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt_text}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"].get("content") or "{}"
    except Exception as exc:
        raise _raise_generation_error(exc, vision=False) from exc


@spaces.GPU(duration=120)
def _run_llamacpp_chat(
    messages: list[dict[str, str]],
    repo: str,
    model_file: str,
    max_tokens: int,
    n_ctx: int,
    n_gpu_layers: int,
    *,
    vision_enabled: bool = False,
    mmproj_file: str = DEFAULT_MMPROJ_FILE,
    chat_handler: str = DEFAULT_CHAT_HANDLER,
) -> str:
    try:
        model_path = download_hf_file(repo, model_file)
        if vision_enabled:
            mmproj_path = download_hf_file(repo, mmproj_file)
    except Exception as exc:
        raise RuntimeError(
            "llama.cpp download failed while preparing the GGUF model: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        if vision_enabled:
            llm = load_vision_llama(model_path, mmproj_path, n_ctx, n_gpu_layers, chat_handler)
        else:
            llm = _load_text(model_path, n_ctx, n_gpu_layers)
    except Exception as exc:
        label = "vision GGUF model for chat" if vision_enabled else "text-only GGUF model for chat"
        raise RuntimeError(
            f"The llama.cpp backend could not load the {label}. "
            f"Inner error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return str(response["choices"][0]["message"].get("content") or "").strip()
    except Exception as exc:
        raise RuntimeError(
            "llama.cpp chat generation failed. "
            f"Inner error: {type(exc).__name__}: {exc}"
        ) from exc


def _compose_prompt(parts: list[dict[str, Any]]) -> str:
    text_parts: list[str] = [EXTRACTION_PROMPT]
    image_count = 0
    for part in parts:
        if part.get("type") == "text":
            text = str(part.get("text", "")).strip()
            if text:
                text_parts.append(text)
        elif part.get("type") == "image_url":
            image_count += 1

    if image_count and len(text_parts) == 1:
        raise RuntimeError(
            "The llama.cpp text backend cannot analyze image-based documents. "
            "Set LLAMACPP_VISION=1 with EXTRACTOR_BACKEND=llamacpp-gpu, or use "
            "EXTRACTOR_BACKEND=transformers for local vision extraction."
        )

    return "\n\n".join(text_parts)
