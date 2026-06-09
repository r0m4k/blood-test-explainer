"""Offline extraction via a local llama.cpp server (llama-server).

This is the off-grid backend that actually works for MiniCPM-V 4.6. The pip `llama-cpp-python`
bundles an llama.cpp too old to load 4.6, but the current `llama-server` (brew / release build)
runs it fine. We POST to a llama-server on localhost with the document image plus:
  - our **GBNF grammar**, so the output is always the `{tests, notes}` schema, and
  - `enable_thinking: false`, so the model doesn't spend its whole token budget on a `<think>`
    ramble (the cause of the "could not be converted into a report" failure).

localhost = the model running on this machine, so it is still fully off-grid (no external call).

Run the server next to the app:
    llama-server -m model.gguf --mmproj mmproj.gguf --port 8080

Config (env):
    LLAMA_SERVER_URL    default http://127.0.0.1:8080/v1/chat/completions
    LLAMA_SERVER_MODEL  default "minicpm-v"
    LLAMA_SERVER_GRAMMAR set to "1" to send the GBNF grammar (OFF by default: the current
                        llama-server build rejects our grammar, and `enable_thinking:false`
                        plus the tolerant parser already yield clean {tests,notes} output)
"""

from __future__ import annotations

import os

import requests

from src.document_processing import document_to_payload_parts
from src.grammar import extraction_grammar
from src.openbmb_client import (
    EXTRACTION_PROMPT,
    ExtractionResult,
    _normalize_notes,
    _normalize_tests,
    _parse_json_response,
)

DEFAULT_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"


class LocalServerExtractor:
    """Implements the `Extractor` protocol against a local llama-server."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 180,
    ) -> None:
        self.url = (url or os.getenv("LLAMA_SERVER_URL") or DEFAULT_SERVER_URL).strip()
        self.model = (model or os.getenv("LLAMA_SERVER_MODEL") or "minicpm-v").strip()
        self.timeout_seconds = timeout_seconds
        self.use_grammar = os.getenv("LLAMA_SERVER_GRAMMAR", "0") == "1"

    def extract(self, file_path: str, max_pages: int = 3) -> ExtractionResult:
        parts = document_to_payload_parts(file_path, max_pages=max_pages)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": EXTRACTION_PROMPT}, *parts]}
            ],
            "temperature": 0,
            "max_tokens": 2048,
            # Stop the model from emitting a <think> reasoning block (it otherwise burns the
            # whole token budget before producing JSON). Unknown fields are ignored by the server.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.use_grammar:
            # Grammar-constrained decoding: output can only be our {tests, notes} schema.
            payload["grammar"] = extraction_grammar()

        response = requests.post(
            self.url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        raw = _message_content(response.json())
        parsed = _parse_json_response(raw)
        return ExtractionResult(
            tests=_normalize_tests(parsed.get("tests", [])),
            notes=_normalize_notes(parsed.get("notes", [])),
            raw_response=raw,
            request_summary={
                "backend": "local-server",
                "url": self.url,
                "document_parts": len(parts),
                "max_pages": max_pages,
                "grammar": self.use_grammar,
            },
        )


def _message_content(payload: dict) -> str:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("llama-server response did not include choices[0].message.") from error
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content.strip()
