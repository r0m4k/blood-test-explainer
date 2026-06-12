"""ZeroGPU extraction via the **llama.cpp** runtime (earns the 🦙 Llama Champion badge).

Runs the quantized MiniCPM-V GGUF through llama.cpp (llama-cpp-python) on the ZeroGPU GPU:
the model + vision projector are downloaded from the Hub, all layers are offloaded to CUDA
(`n_gpu_layers=-1`), and generation is wrapped in `@spaces.GPU`. This gives us, in one path:
  - 🦙 Llama Champion  (runs through llama.cpp)
  - 🔌 Off the Grid    (model in the Space, no external inference API)
  - quantized GGUF on GPU (fast, low VRAM)

Requires a llama-cpp-python CUDA wheel with the MiniCPM-V chat handler available.

Config (env):
  LLAMACPP_GGUF_REPO    HF repo with the GGUF + mmproj   (default: openbmb/MiniCPM-V-4.6-gguf;
                        point at your fine-tuned GGUF repo once trained)
  LLAMACPP_MODEL_FILE   default MiniCPM-V-4_6-Q4_K_M.gguf
  LLAMACPP_MMPROJ_FILE  default mmproj-model-f16.gguf
  LLAMACPP_CHAT_HANDLER llama_cpp chat-handler class (default MiniCPMv26ChatHandler; confirm the
                        4.6 handler shipped with your llama-cpp-python build)
  LLAMACPP_MAX_TOKENS   default 3072
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
DEFAULT_MMPROJ_FILE = "mmproj-model-f16.gguf"

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
        self.max_tokens = int(os.getenv("LLAMACPP_MAX_TOKENS", "3072"))
        self.chat_handler = os.getenv("LLAMACPP_CHAT_HANDLER", "MiniCPMv26ChatHandler").strip()

    def extract(self, file_path: str, max_pages: int = 3) -> ExtractionResult:
        parts = document_to_payload_parts(file_path, max_pages=max_pages)
        raw = _run_llamacpp_generation(
            parts=parts,
            repo=self.repo,
            model_file=self.model_file,
            mmproj_file=self.mmproj_file,
            chat_handler=self.chat_handler,
            max_tokens=self.max_tokens,
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
def _download(repo: str, model_file: str, mmproj_file: str) -> tuple[str, str]:
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=repo, filename=model_file)
    mmproj_path = hf_hub_download(repo_id=repo, filename=mmproj_file)
    return model_path, mmproj_path


@lru_cache(maxsize=1)
def _load(model_path: str, mmproj_path: str, chat_handler_name: str):
    from llama_cpp import Llama, llama_chat_format

    handler_cls = getattr(llama_chat_format, chat_handler_name, None)
    if handler_cls is None:
        raise RuntimeError(
            f"Chat handler '{chat_handler_name}' not in llama_cpp.llama_chat_format. "
            "Set LLAMACPP_CHAT_HANDLER to the handler matching your MiniCPM-V build."
        )
    handler = handler_cls(clip_model_path=mmproj_path)
    return Llama(
        model_path=model_path,
        chat_handler=handler,
        n_ctx=4096,
        n_gpu_layers=-1,   # offload everything to the ZeroGPU CUDA device
        verbose=False,
    )


@spaces.GPU(duration=120)
def _run_llamacpp_generation(
    parts: list[dict[str, Any]],
    repo: str,
    model_file: str,
    mmproj_file: str,
    chat_handler: str,
    max_tokens: int,
) -> str:
    try:
        model_path, mmproj_path = _download(repo, model_file, mmproj_file)
    except Exception as exc:
        raise RuntimeError(
            "llama.cpp download failed while preparing the GGUF/mmproj pair: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        llm = _load(model_path, mmproj_path, chat_handler)
    except Exception as exc:
        raise RuntimeError(
            "The llama.cpp backend could not load the MiniCPM-V GGUF/mmproj pair. "
            "This usually means the downloaded model build is incompatible with the installed "
            "llama-cpp-python wheel, the chat handler name does not match the wheel, or one of "
            "the model files is incomplete. "
            f"Inner error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": [{"type": "text", "text": EXTRACTION_PROMPT}, *parts]}],
            response_format={"type": "json_object"},
            temperature=0.0,
            reasoning_budget=0,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"].get("content") or "{}"
    except Exception as exc:
        raise RuntimeError(
            "llama.cpp generation failed while extracting the document. "
            f"Inner error: {type(exc).__name__}: {exc}"
        ) from exc
