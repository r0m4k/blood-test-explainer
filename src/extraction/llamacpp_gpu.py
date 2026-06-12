"""ZeroGPU extraction via the **llama.cpp** runtime (earns the 🦙 Llama Champion badge).

Runs a quantized llama.cpp model against PDF-extracted text on the ZeroGPU GPU.
The prompt is now text-only, which avoids loading any image encoder or mmproj:
  - 🦙 Llama Champion  (runs through llama.cpp)
  - 🔌 Off the Grid    (model in the Space, no external inference API)
  - quantized GGUF on GPU/CPU (fast, low VRAM)

Requires a llama-cpp-python wheel with standard text chat support.

Config (env):
  LLAMACPP_GGUF_REPO    HF repo with the text-only GGUF   (default: set to your fine-tuned repo)
  LLAMACPP_MODEL_FILE   default text-only GGUF filename
  LLAMACPP_MAX_TOKENS   default 3072
  LLAMACPP_N_CTX        default 8192
  LLAMACPP_N_GPU_LAYERS default 0 for CPU wheels; set to -1 only on CUDA builds
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from src.document_processing import document_to_payload_parts
from src.openbmb_client import (
    EXTRACTION_PROMPT,
    ExtractionResult,
    _normalize_notes,
    _normalize_patient,
    _normalize_tests,
    _parse_json_response,
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
        self.max_tokens = int(os.getenv("LLAMACPP_MAX_TOKENS", "3072"))
        self.n_ctx = int(os.getenv("LLAMACPP_N_CTX", "8192"))
        self.n_gpu_layers = int(os.getenv("LLAMACPP_N_GPU_LAYERS", "0"))

    def extract(self, file_path: str, max_pages: int = 3) -> ExtractionResult:
        parts = document_to_payload_parts(file_path, max_pages=max_pages)
        prompt_text = _compose_prompt(parts)
        raw = _run_llamacpp_generation(
            prompt_text=prompt_text,
            repo=self.repo,
            model_file=self.model_file,
            max_tokens=self.max_tokens,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
        )
        parsed = _parse_json_response(raw)
        return ExtractionResult(
            patient=_normalize_patient(parsed.get("patient", {})),
            tests=_normalize_tests(parsed.get("tests", [])),
            notes=_normalize_notes(parsed.get("notes", [])),
            raw_response=raw,
            request_summary={
                "backend": "llamacpp-gpu",
                "repo": self.repo,
                "document_parts": len(parts),
                "max_pages": max_pages,
            },
        )


@lru_cache(maxsize=1)
def _download(repo: str, model_file: str) -> str:
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=repo, filename=model_file)
    return model_path


@lru_cache(maxsize=1)
def _load(model_path: str, n_ctx: int, n_gpu_layers: int):
    from llama_cpp import Llama
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


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
        model_path = _download(repo, model_file)
    except Exception as exc:
        raise RuntimeError(
            "llama.cpp download failed while preparing the GGUF model: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        llm = _load(model_path, n_ctx, n_gpu_layers)
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
        message = f"{type(exc).__name__}: {exc}"
        if "llama_decode returned -1" in message:
            raise RuntimeError(
                "llama.cpp ran out of room while decoding the PDF text prompt. "
                "Try increasing LLAMACPP_N_CTX, lowering the number of PDF pages, or trimming the "
                "input text before sending it to the model."
            ) from exc
        raise RuntimeError(
            "llama.cpp generation failed while extracting the document. "
            f"Inner error: {message}"
        ) from exc


def _compose_prompt(parts: list[dict[str, Any]]) -> str:
    text_parts: list[str] = [EXTRACTION_PROMPT]
    for part in parts:
        if part.get("type") == "text":
            text = str(part.get("text", "")).strip()
            if text:
                text_parts.append(text)
    return "\n\n".join(text_parts)
