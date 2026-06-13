import sys
import os
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.factory import build_extractor


def test_auto_uses_llamacpp_on_cpu_basic_space(monkeypatch):
    monkeypatch.delenv("EXTRACTOR_BACKEND", raising=False)
    monkeypatch.delenv("LLAMACPP_VISION", raising=False)
    monkeypatch.setenv("BTE_SPACE_HARDWARE", "cpu-basic")

    with patch("src.extraction.factory.LlamaCppGPUExtractor") as llama_cls:
        extractor = build_extractor()

    llama_cls.assert_called_once_with()
    assert extractor is llama_cls.return_value
    assert os.environ["LLAMACPP_VISION"] == "1"


def test_auto_uses_transformers_when_hardware_is_not_cpu_basic(monkeypatch):
    monkeypatch.delenv("EXTRACTOR_BACKEND", raising=False)
    monkeypatch.setenv("BTE_SPACE_HARDWARE", "zero-a10g")

    with patch("src.extraction.factory.AutoExtractor") as auto_cls:
        extractor = build_extractor(model="demo/model")

    auto_cls.assert_called_once_with(model_id="demo/model")
    assert extractor is auto_cls.return_value
