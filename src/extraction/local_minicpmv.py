"""Local, offline extraction with fine-tuned MiniCPM-V under llama.cpp.

This is the off-grid backend: the (fine-tuned) MiniCPM-V vision model as a quantized GGUF,
plus its multimodal projector (mmproj), run entirely on-device via llama-cpp-python. The same
PDF/image → data-URL pipeline used by the API backend feeds the model here, and the output is
GBNF-constrained to our extraction schema so it is always valid JSON.

No network calls. Earns: off-grid (local model), fine-tune (LoRA → merged GGUF), quantization
(Q4_K_M GGUF). The GGUF + mmproj files come from the fine-tune pipeline (see train/ + scripts/).

Configuration (env):
    LOCAL_MODEL_PATH    path to the (quantized) MiniCPM-V GGUF        [required for local]
    LOCAL_MMPROJ_PATH   path to the mmproj GGUF (vision projector)    [required for local]
    LOCAL_N_CTX         context window (default 4096)
    LOCAL_N_GPU_LAYERS  GPU offload layers (0 = pure CPU; >0 on ZeroGPU/CUDA)
    LOCAL_CHAT_HANDLER  llama_cpp chat-handler class name (default: MiniCPMv26ChatHandler)

⚠️ The exact chat-handler class is version-dependent. MiniCPM-V 2.6 uses
`MiniCPMv26ChatHandler`; confirm the handler shipped with your llama-cpp-python build for the
4.6 checkpoint and override via LOCAL_CHAT_HANDLER if needed. Verify on real hardware.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

from src.document_processing import document_to_payload_parts
from src.grammar import extraction_grammar
from src.openbmb_client import (
    EXTRACTION_PROMPT,
    ExtractionResult,
    _normalize_notes,
    _normalize_tests,
)


class LocalMiniCPMVExtractor:
    """Offline MiniCPM-V extractor (llama.cpp). Implements the `Extractor` protocol."""

    def __init__(
        self,
        model_path: str | None = None,
        mmproj_path: str | None = None,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        chat_handler_name: str | None = None,
    ) -> None:
        self.model_path = model_path or os.getenv("LOCAL_MODEL_PATH")
        self.mmproj_path = mmproj_path or os.getenv("LOCAL_MMPROJ_PATH")
        self.n_ctx = n_ctx if n_ctx is not None else int(os.getenv("LOCAL_N_CTX", "4096"))
        self.n_gpu_layers = (
            n_gpu_layers if n_gpu_layers is not None else int(os.getenv("LOCAL_N_GPU_LAYERS", "0"))
        )
        self.chat_handler_name = (
            chat_handler_name or os.getenv("LOCAL_CHAT_HANDLER", "MiniCPMv26ChatHandler")
        )
        if not self.model_path or not self.mmproj_path:
            raise RuntimeError(
                "Local backend needs LOCAL_MODEL_PATH and LOCAL_MMPROJ_PATH (the fine-tuned "
                "MiniCPM-V GGUF + mmproj). Run the fine-tune + GGUF pipeline first, or set "
                "EXTRACTOR_BACKEND=api to use the hosted endpoint."
            )
        # Fail fast if the model files are missing.
        for path in (self.model_path, self.mmproj_path):
            if not os.path.exists(path):
                raise RuntimeError(f"Model file not found: {path}")

    @property
    def is_configured(self) -> bool:
        return bool(self.model_path and self.mmproj_path)

    def extract(self, file_path: str, max_pages: int = 3) -> ExtractionResult:
        llm = _load_model(
            self.model_path, self.mmproj_path, self.n_ctx, self.n_gpu_layers, self.chat_handler_name
        )
        parts = document_to_payload_parts(file_path, max_pages=max_pages)

        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": [{"type": "text", "text": EXTRACTION_PROMPT}, *parts]}],
            grammar=_grammar(),
            temperature=0.0,
            max_tokens=2048,
        )
        raw = response["choices"][0]["message"]["content"] or "{}"
        # GBNF guarantees valid JSON, but never trust a single parse.
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}

        return ExtractionResult(
            tests=_normalize_tests(parsed.get("tests", [])),
            notes=_normalize_notes(parsed.get("notes", [])),
            raw_response=raw,
            request_summary={
                "backend": "local-minicpmv",
                "model_path": os.path.basename(self.model_path),
                "document_parts": len(parts),
                "max_pages": max_pages,
            },
        )


@lru_cache(maxsize=1)
def _grammar():
    from llama_cpp import LlamaGrammar  # lazy

    return LlamaGrammar.from_string(extraction_grammar())


@lru_cache(maxsize=2)
def _load_model(model_path: str, mmproj_path: str, n_ctx: int, n_gpu_layers: int, handler_name: str):
    """Load the GGUF + vision projector once and cache it (cold start is expensive)."""
    try:
        from llama_cpp import Llama
        from llama_cpp import llama_chat_format
    except ImportError as exc:  # pragma: no cover - optional heavy dep
        raise ImportError(
            "llama-cpp-python is not installed. Install it (see requirements.txt) to use the "
            "local backend, or set EXTRACTOR_BACKEND=api."
        ) from exc

    handler_cls = getattr(llama_chat_format, handler_name, None)
    if handler_cls is None:
        raise RuntimeError(
            f"Chat handler '{handler_name}' not found in llama_cpp.llama_chat_format. "
            "Set LOCAL_CHAT_HANDLER to the handler matching your MiniCPM-V build."
        )
    chat_handler = handler_cls(clip_model_path=mmproj_path)
    return Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
