from __future__ import annotations

import os
from pathlib import Path


def load_local_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        _apply_model_defaults()
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value

    _apply_model_defaults()


def _apply_model_defaults() -> None:
    from src.model_paths import apply_local_model_defaults

    apply_local_model_defaults()
