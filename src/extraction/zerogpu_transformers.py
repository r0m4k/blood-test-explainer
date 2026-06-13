"""ZeroGPU extraction backend using the fine-tuned MiniCPM-V Transformers path."""

from __future__ import annotations

import os
from typing import Any

from src.document_processing import document_intake_metadata, document_to_payload_parts
from src.openbmb_client import (
    EXTRACTION_PROMPT,
    ExtractionResult,
    _normalize_notes,
    _normalize_patient,
    _normalize_tests,
    _parse_json_response,
    summarize_document_parts,
)

from src.model_paths import TransformersModelSource, resolve_transformers_model_source


class ZeroGPUTransformersExtractor:
    """Extractor backed by local or Hub MiniCPM-V Transformers weights."""

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = 2048,
        downsample_mode: str = "16x",
    ) -> None:
        self.model_source = resolve_transformers_model_source(model_id)
        self.model_id = self.model_source.model_id
        self.max_new_tokens = int(os.getenv("ZEROGPU_MAX_NEW_TOKENS", str(max_new_tokens)))
        self.downsample_mode = (os.getenv("ZEROGPU_DOWNSAMPLE_MODE") or downsample_mode).strip()

    def extract(self, file_path: str, max_pages: int = 3) -> ExtractionResult:
        parts = document_to_payload_parts(file_path, max_pages=max_pages)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    *_to_transformers_content(parts),
                ],
            }
        ]
        raw = _run_zerogpu_generation(
            messages=messages,
            model_source=self.model_source,
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
                "backend": "transformers",
                "model": self.model_id,
                "model_origin": self.model_source.origin,
                "model_local_only": self.model_source.local_files_only,
                "document_parts": len(parts),
                "max_pages": max_pages,
                "downsample_mode": self.downsample_mode,
                "extraction_prompt": EXTRACTION_PROMPT,
                "user_message_preview": summarize_document_parts(parts),
                **document_intake_metadata(file_path, parts),
                "messages_preview": _messages_preview(messages),
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


def _messages_preview(messages: list[dict[str, Any]]) -> str:
    """Serialize message structure without embedding image data URLs."""
    preview: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            preview.append({"role": message.get("role"), "content": _truncate_preview(content)})
            continue
        if not isinstance(content, list):
            continue
        items: list[dict[str, str]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                items.append({"type": "image", "url": "[image omitted]"})
            elif item.get("type") == "text":
                items.append({"type": "text", "text": _truncate_preview(str(item.get("text") or ""))})
            elif item.get("type") == "image_url":
                items.append({"type": "image_url", "url": "[image omitted]"})
        preview.append({"role": message.get("role"), "content": items})
    import json

    return json.dumps(preview, indent=2)


def _truncate_preview(text: str, limit: int = 1200) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _load_model(source: TransformersModelSource):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from src.model_paths import hub_cache_dir

    pretrained_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": source.local_files_only,
    }
    if not source.local_files_only:
        pretrained_kwargs["cache_dir"] = str(hub_cache_dir())

    processor = AutoProcessor.from_pretrained(source.model_id, **pretrained_kwargs)

    use_4bit = os.getenv("ZEROGPU_QUANTIZE", "1") != "0" and torch.cuda.is_available()
    load_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True, "local_files_only": source.local_files_only}
    if not source.local_files_only:
        load_kwargs["cache_dir"] = str(hub_cache_dir())
    if use_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForImageTextToText.from_pretrained(source.model_id, **load_kwargs)
    model.eval()
    return processor, model


_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _cache_key(source: TransformersModelSource) -> str:
    return f"{source.model_id}|local={int(source.local_files_only)}|origin={source.origin}"


def _get_model(source: TransformersModelSource) -> tuple[Any, Any]:
    from src.model_paths import hub_cache_dir

    key = _cache_key(source)
    if key not in _MODEL_CACHE:
        if source.local_files_only:
            print(f"[Blood Test Explainer] loading local Transformers model from {source.model_id}", flush=True)
        else:
            print(
                f"[Blood Test Explainer] downloading Transformers model {source.model_id} "
                f"(cache: {hub_cache_dir()}) and loading into memory",
                flush=True,
            )
        _MODEL_CACHE[key] = _load_model(source)
    return _MODEL_CACHE[key]


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


@spaces.GPU(duration=180)
def _run_zerogpu_generation(
    messages: list[dict[str, Any]],
    model_source: TransformersModelSource,
    max_new_tokens: int,
    downsample_mode: str,
) -> str:
    import torch

    try:
        processor, model = _get_model(model_source)
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
    except Exception as exc:
        raise RuntimeError(
            "MiniCPM-V Transformers generation failed. "
            f"Inner error: {type(exc).__name__}: {exc}"
        ) from exc
