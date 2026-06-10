"""ZeroGPU extraction backend using the official MiniCPM-V Transformers path."""

from __future__ import annotations

import os
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

DEFAULT_ZEROGPU_MODEL = "openbmb/MiniCPM-V-4.6"


class ZeroGPUTransformersExtractor:
    """Extractor backed by HF ZeroGPU and the OpenBMB Transformers implementation."""

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = 2048,
        downsample_mode: str = "16x",
    ) -> None:
        self.model_id = (model_id or os.getenv("ZEROGPU_MODEL_ID") or DEFAULT_ZEROGPU_MODEL).strip()
        self.max_new_tokens = int(os.getenv("ZEROGPU_MAX_NEW_TOKENS", str(max_new_tokens)))
        self.downsample_mode = (os.getenv("ZEROGPU_DOWNSAMPLE_MODE") or downsample_mode).strip()

    def extract(self, file_path: str, max_pages: int = 3) -> ExtractionResult:
        parts = document_to_payload_parts(file_path, max_pages=max_pages)
        messages = [
            {
                "role": "user",
                "content": [
                    *_to_transformers_content(parts),
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ]
        raw = _run_zerogpu_generation(
            messages=messages,
            model_id=self.model_id,
            max_new_tokens=self.max_new_tokens,
            downsample_mode=self.downsample_mode,
        )
        parsed = _parse_json_response(raw)
        return ExtractionResult(
            patient=_normalize_patient(parsed.get("patient", {})),
            tests=_normalize_tests(parsed.get("tests", [])),
            notes=_normalize_notes(parsed.get("notes", [])),
            raw_response=raw,
            request_summary={
                "backend": "zerogpu-transformers",
                "model": self.model_id,
                "document_parts": len(parts),
                "max_pages": max_pages,
                "downsample_mode": self.downsample_mode,
            },
        )


def _to_transformers_content(parts: list[dict[str, Any]]) -> list[dict[str, str]]:
    content: list[dict[str, str]] = []
    text_chunks: list[str] = []
    for part in parts:
        if part.get("type") == "image_url":
            image_url = part.get("image_url") or {}
            url = image_url.get("url")
            if url:
                content.append({"type": "image", "url": str(url)})
        elif part.get("type") == "text":
            text = str(part.get("text") or "").strip()
            if text:
                text_chunks.append(text)

    if text_chunks:
        content.append({"type": "text", "text": "\n\n".join(text_chunks)})
    return content


def _load_model(model_id: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)

    # 4-bit (NF4) quantization on GPU: earns the quantization badge and roughly quarters the
    # GPU memory footprint (helps stay within ZeroGPU limits). Set ZEROGPU_QUANTIZE=0 to fall
    # back to bf16 full precision if bitsandbytes ever misbehaves on the runtime.
    use_4bit = os.getenv("ZEROGPU_QUANTIZE", "1") != "0" and torch.cuda.is_available()
    load_kwargs: dict[str, Any] = {"device_map": "auto"}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else "auto"

    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return processor, model


_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _get_model(model_id: str) -> tuple[Any, Any]:
    if model_id not in _MODEL_CACHE:
        _MODEL_CACHE[model_id] = _load_model(model_id)
    return _MODEL_CACHE[model_id]


try:
    import spaces
except ImportError:  # Local development without the HF Spaces package.
    class _SpacesFallback:
        @staticmethod
        def GPU(*_args: Any, **_kwargs: Any):
            def decorator(func):
                return func

            return decorator

    spaces = _SpacesFallback()  # type: ignore[assignment]


@spaces.GPU(duration=120)
def _run_zerogpu_generation(
    messages: list[dict[str, Any]],
    model_id: str,
    max_new_tokens: int,
    downsample_mode: str,
) -> str:
    import torch

    processor, model = _get_model(model_id)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        downsample_mode=downsample_mode,
        max_slice_nums=9,
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            downsample_mode=downsample_mode,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return str(output_text[0]).strip() if output_text else ""
