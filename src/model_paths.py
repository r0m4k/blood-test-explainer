"""Resolve MiniCPM-V Transformers weights: local disk first, Hub download fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Fine-tuned MiniCPM-V 4.6 for lab extraction (MedReason SFT).
DEFAULT_HF_REPO = "build-small-hackathon/blood-test-minicpmv-4_6-medreason"
# Upstream OpenBMB base — eval baselines and training scripts.
BASE_HF_REPO = "openbmb/MiniCPM-V-4.6"


@dataclass(frozen=True)
class TransformersModelSource:
    model_id: str
    local_files_only: bool
    origin: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def models_dir() -> Path:
    raw = os.getenv("BTE_MODELS_DIR", "models").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def hub_cache_dir() -> Path:
    cache = models_dir() / ".cache" / "huggingface" / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def apply_local_model_defaults() -> None:
    """Send Hugging Face downloads to the project models/ cache by default."""
    models = models_dir()
    os.environ.setdefault("BTE_MODELS_DIR", str(models))
    hf_home = models / ".cache" / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache_dir()))


def resolve_transformers_model_source(model_id: str | None = None) -> TransformersModelSource:
    """Use a complete local checkpoint when present; otherwise download and load from Hub."""
    configured = (model_id or os.getenv("ZEROGPU_MODEL_ID") or DEFAULT_HF_REPO).strip()
    repo_id = DEFAULT_HF_REPO if configured.startswith(".") or configured.startswith("/") else configured

    explicit = Path(configured).expanduser()
    if explicit.is_dir() and is_transformers_model_dir(explicit):
        return TransformersModelSource(str(explicit.resolve()), True, "local-dir")

    for candidate in (
        models_dir() / "MiniCPM-V-4.6",
        models_dir() / "openbmb" / "MiniCPM-V-4.6",
        models_dir() / repo_id.split("/", 1)[-1],
    ):
        if is_transformers_model_dir(candidate):
            return TransformersModelSource(str(candidate.resolve()), True, "local-dir")

    for cache_root in (hub_cache_dir(), Path.home() / ".cache" / "huggingface" / "hub"):
        snapshot = latest_complete_snapshot(repo_id, cache_root)
        if snapshot:
            label = "local-cache" if cache_root == hub_cache_dir() else "local-cache-global"
            return TransformersModelSource(str(snapshot), True, label)

    return TransformersModelSource(repo_id, False, "hub-download")


def is_transformers_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    if list(path.glob("*.safetensors")) or list(path.glob("model*.bin")):
        return True
    return any(path.glob("model*.safetensors.index.json"))


def latest_complete_snapshot(repo_id: str, hub_cache: Path) -> Path | None:
    if not hub_cache.is_dir():
        return None
    repo_dir = hub_cache / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not repo_dir.is_dir():
        return None
    snapshots = sorted(
        (p for p in repo_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for snapshot in snapshots:
        if is_transformers_model_dir(snapshot):
            return snapshot
    return None
