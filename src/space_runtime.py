"""Helpers for hardware-aware Hugging Face Space behavior."""

from __future__ import annotations

import os
from functools import lru_cache


_HARDWARE_ENV_KEYS = (
    "BTE_SPACE_HARDWARE",
    "SPACE_HARDWARE",
    "HF_SPACE_HARDWARE",
)


def is_huggingface_space() -> bool:
    return bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST"))


def configured_space_hardware() -> str | None:
    for key in _HARDWARE_ENV_KEYS:
        value = os.getenv(key, "").strip().lower()
        if value:
            return value
    return _hub_space_hardware()


def is_cpu_basic_space() -> bool:
    return configured_space_hardware() == "cpu-basic"


@lru_cache(maxsize=8)
def _hub_space_hardware() -> str | None:
    repo_id = os.getenv("SPACE_ID", "").strip()
    if not repo_id:
        return None
    try:
        from huggingface_hub import HfApi

        info = HfApi().space_info(repo_id=repo_id)
        runtime = getattr(info, "runtime", None)
        hardware = getattr(runtime, "hardware", None)
        if hardware:
            return str(hardware).strip().lower()
    except Exception:
        return None
    return None
