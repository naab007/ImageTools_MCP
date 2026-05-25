"""Server-wide model path configuration.

Centralises the filesystem paths used by all model-loading modules so they
can be overridden in one place rather than hardcoded across ``qwen.py``,
``sd.py``, ``face_swap.py``, ``sam.py``, etc.

The defaults match the ComfyUI convention on the author's machine. Override
via:
- The ``configure_paths`` MCP tool (runtime, per-session)
- Environment variables (``IMAGETOOLS_INSIGHTFACE_ROOT``, etc.)
- Editing this file (permanent)

The config is a plain dict — no file on disk, no persistence across server
restarts. If you need persistence, set the env vars in your shell profile
or in the MCP server registration's ``env`` block.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_DEFAULTS: dict[str, str] = {
    "insightface_root": r"B:\-AI-Stuff-\ComfyUI\models\insightface",
    "facerestore_dir": r"B:\-AI-Stuff-\ComfyUI\models\facerestore_models",
    "saved_faces_dir": r"B:\-AI-Stuff-\faces",
    "comfyui_unet_dir": r"B:\-AI-Stuff-\ComfyUI\models\unet",
    "hf_cache_dir": "",  # empty = HuggingFace default (~/.cache/huggingface)
    "scratch_dir": r"C:\Users\Naabin\AppData\Local\Temp\imagetools_scratch",
}

import os as _os

_config: dict[str, str] = {}
for k, v in _DEFAULTS.items():
    env_key = f"IMAGETOOLS_{k.upper()}"
    _config[k] = _os.environ.get(env_key, v)


def get(key: str) -> str:
    """Return the current value for ``key``. Raises ``KeyError`` if unknown."""
    if key not in _config:
        raise KeyError(
            f"unknown config key {key!r}. Known: {sorted(_config)}"
        )
    return _config[key]


def get_path(key: str) -> Path:
    """Like ``get`` but returns a ``Path``. Empty string → ``Path('.')``."""
    return Path(get(key)) if get(key) else Path(".")


def set(key: str, value: str) -> None:
    """Override a config value for this session."""
    if key not in _config:
        raise KeyError(
            f"unknown config key {key!r}. Known: {sorted(_config)}"
        )
    _config[key] = str(value)


def get_all() -> dict[str, str]:
    """Return a copy of the full config dict."""
    return dict(_config)
