import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.llamacpp_gpu import LlamaCppGPUExtractor, _compose_prompt
from src.extraction.llamacpp_vision import llamacpp_vision_enabled
from src.openbmb_client import EXTRACTION_PROMPT


def test_llamacpp_vision_enabled(monkeypatch):
    monkeypatch.delenv("LLAMACPP_VISION", raising=False)
    assert llamacpp_vision_enabled() is False

    monkeypatch.setenv("LLAMACPP_VISION", "1")
    assert llamacpp_vision_enabled() is True

    monkeypatch.setenv("LLAMACPP_VISION", "yes")
    assert llamacpp_vision_enabled() is True


def test_compose_prompt_rejects_image_only_without_vision():
    parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
    with pytest.raises(RuntimeError, match="LLAMACPP_VISION=1"):
        _compose_prompt(parts)


def test_compose_prompt_keeps_text_files():
    parts = [{"type": "text", "text": "Hemoglobin 11.2 g/dL"}]
    prompt = _compose_prompt(parts)
    assert EXTRACTION_PROMPT in prompt
    assert "Hemoglobin 11.2 g/dL" in prompt


def test_extract_uses_vision_generation_when_enabled(monkeypatch):
    monkeypatch.setenv("LLAMACPP_VISION", "1")
    monkeypatch.delenv("BTE_SPACE_HARDWARE", raising=False)
    image_parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]

    with patch(
        "src.extraction.llamacpp_gpu.document_to_payload_parts",
        return_value=image_parts,
    ), patch(
        "src.extraction.llamacpp_gpu._run_llamacpp_vision_generation",
        return_value='{"patient":{},"tests":[],"notes":[]}',
    ) as vision_run, patch(
        "src.extraction.llamacpp_gpu._run_llamacpp_generation",
    ) as text_run:
        result = LlamaCppGPUExtractor().extract("/tmp/report.pdf")

    vision_run.assert_called_once()
    text_run.assert_not_called()
    assert result.request_summary["backend"] == "llamacpp-gpu-vision"
    assert result.request_summary["vision_enabled"] is True
    assert result.request_summary["mmproj"]
    assert result.request_summary["spaces_gpu"] is True


def test_cpu_basic_uses_plain_cpu_vision_runner(monkeypatch):
    monkeypatch.setenv("BTE_SPACE_HARDWARE", "cpu-basic")
    monkeypatch.setenv("LLAMACPP_VISION", "1")
    image_parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]

    with patch(
        "src.extraction.llamacpp_gpu.document_to_payload_parts",
        return_value=image_parts,
    ), patch(
        "src.extraction.llamacpp_gpu._run_llamacpp_vision_generation_cpu",
        return_value='{"patient":{},"tests":[],"notes":[]}',
    ) as cpu_run, patch(
        "src.extraction.llamacpp_gpu._run_llamacpp_vision_generation",
    ) as gpu_run:
        result = LlamaCppGPUExtractor().extract("/tmp/report.pdf")

    cpu_run.assert_called_once()
    gpu_run.assert_not_called()
    assert result.request_summary["backend"] == "llamacpp-cpu-vision"
    assert result.request_summary["spaces_gpu"] is False


def test_extract_uses_text_generation_when_vision_disabled(monkeypatch):
    monkeypatch.delenv("LLAMACPP_VISION", raising=False)
    monkeypatch.delenv("BTE_SPACE_HARDWARE", raising=False)
    text_parts = [{"type": "text", "text": "WBC 6.5"}]

    with patch(
        "src.extraction.llamacpp_gpu.document_to_payload_parts",
        return_value=text_parts,
    ), patch(
        "src.extraction.llamacpp_gpu._run_llamacpp_generation",
        return_value='{"patient":{},"tests":[],"notes":[]}',
    ) as text_run, patch(
        "src.extraction.llamacpp_gpu._run_llamacpp_vision_generation",
    ) as vision_run:
        result = LlamaCppGPUExtractor().extract("/tmp/report.txt")

    text_run.assert_called_once()
    vision_run.assert_not_called()
    assert result.request_summary["backend"] == "llamacpp-gpu"
    assert result.request_summary["vision_enabled"] is False


def test_vision_extractor_requires_mmproj_file(monkeypatch):
    monkeypatch.setenv("LLAMACPP_VISION", "1")
    monkeypatch.setenv("LLAMACPP_MMPROJ_FILE", " ")
    with pytest.raises(ValueError, match="LLAMACPP_MMPROJ_FILE"):
        LlamaCppGPUExtractor()
