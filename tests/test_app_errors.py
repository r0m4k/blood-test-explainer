import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import _format_extraction_error  # noqa: E402


def test_llamacpp_load_failure_message_is_specific():
    message = _format_extraction_error(ValueError("Failed to load model from file: /tmp/model.gguf"))
    assert "llama.cpp backend could not load the GGUF model" in message
    assert "background worker problem" in message
