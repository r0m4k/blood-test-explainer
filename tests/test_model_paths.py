import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_paths import (
    BASE_HF_REPO,
    DEFAULT_HF_REPO,
    is_transformers_model_dir,
    resolve_transformers_model_source,
)


def test_resolve_uses_hub_download_when_no_local_weights(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        empty_models = Path(tmp) / "models"
        empty_models.mkdir()
        monkeypatch.setenv("BTE_MODELS_DIR", str(empty_models))
        monkeypatch.setenv("HF_HOME", str(Path(tmp) / "hf"))
        monkeypatch.setattr(
            "src.model_paths.latest_complete_snapshot",
            lambda repo_id, hub_cache: None,
        )

        source = resolve_transformers_model_source(DEFAULT_HF_REPO)
        assert source.local_files_only is False
        assert source.origin == "hub-download"
        assert source.model_id == DEFAULT_HF_REPO


def test_resolve_defaults_to_finetuned_repo(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        empty_models = Path(tmp) / "models"
        empty_models.mkdir()
        monkeypatch.setenv("BTE_MODELS_DIR", str(empty_models))
        monkeypatch.setenv("HF_HOME", str(Path(tmp) / "hf"))
        monkeypatch.delenv("ZEROGPU_MODEL_ID", raising=False)
        monkeypatch.setattr(
            "src.model_paths.latest_complete_snapshot",
            lambda repo_id, hub_cache: None,
        )

        source = resolve_transformers_model_source()
        assert source.model_id == DEFAULT_HF_REPO
        assert source.origin == "hub-download"


def test_resolve_base_repo_for_eval_baseline(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        empty_models = Path(tmp) / "models"
        empty_models.mkdir()
        monkeypatch.setenv("BTE_MODELS_DIR", str(empty_models))
        monkeypatch.setenv("HF_HOME", str(Path(tmp) / "hf"))
        monkeypatch.setattr(
            "src.model_paths.latest_complete_snapshot",
            lambda repo_id, hub_cache: None,
        )

        source = resolve_transformers_model_source(BASE_HF_REPO)
        assert source.model_id == BASE_HF_REPO


def test_resolve_uses_local_dir_when_complete():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "MiniCPM-V-4.6"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"test")

        source = resolve_transformers_model_source(str(model_dir))
        assert source.local_files_only is True
        assert source.origin == "local-dir"
        assert Path(source.model_id) == model_dir.resolve()


def test_is_transformers_model_dir_requires_weights():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "partial"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        assert is_transformers_model_dir(model_dir) is False


if __name__ == "__main__":
    test_resolve_uses_hub_download_when_no_local_weights()
    test_resolve_defaults_to_finetuned_repo()
    test_resolve_base_repo_for_eval_baseline()
    test_resolve_uses_local_dir_when_complete()
    test_is_transformers_model_dir_requires_weights()
    print("test_model_paths: ok")
