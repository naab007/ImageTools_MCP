"""GGUF helper utilities — detection, download, and diffusers config.

GGUF support in ImageTools_MCP lets you run heavy diffusion models
(Qwen-Image-Edit, FLUX, SD3) at fp4/fp5/fp8 precision instead of bfloat16,
dropping VRAM from ~12 GB to ~3-6 GB at a small quality cost.

Workflow:

    # one-time: pull a quantized transformer file
    path = download_gguf("city96/Qwen-Image-Edit-gguf",
                        "qwen-image-edit-Q4_K_S.gguf")

    # then pass that path when loading the pipeline
    qwen_load(model="Qwen/Qwen-Image-Edit", gguf_path=path)
    qwen_edit_image(canvas_id="photo", prompt="add a hat")

The base pipeline (text encoder, VAE, scheduler) still comes from the HF
repo — GGUF only quantizes the heavy transformer/UNet.

Diffusers' GGUFQuantizationConfig handles dequantization on the fly during
inference. Compute dtype controls intermediate precision (bfloat16 default
on modern GPUs; float16 for older hardware).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Common GGUF quantization levels in order of memory savings (Q2 smallest,
# Q8 largest, BF16/F16 unquantized). Documented here so callers know what
# tradeoff filenames imply.
QUANT_LEVELS = [
    "Q2_K",      # ~2.5 bpw, lowest quality, smallest file
    "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",   # ~4 bpw, popular sweet spot
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K",
    "Q8_0",      # ~8 bpw, lossless-ish
    "F16", "BF16",  # unquantized
]


def is_gguf_path(path: str | Path | None) -> bool:
    """True iff ``path`` looks like a GGUF file (extension match — we don't
    open the file to check magic bytes because callers may pass a path to a
    download target that doesn't exist yet)."""
    if not path:
        return False
    return str(path).lower().endswith(".gguf")


def detect_quant_level(path: str | Path) -> str | None:
    """Sniff the quantization level from a GGUF filename, e.g.
    ``foo-Q4_K_S.gguf`` → ``"Q4_K_S"``. Returns None if no known level matches."""
    name = Path(path).stem.upper()
    for q in sorted(QUANT_LEVELS, key=len, reverse=True):
        if q in name:
            return q
    return None


def _check_available() -> None:
    """Raise if the GGUF stack isn't installed."""
    missing: list[str] = []
    try:
        import diffusers  # noqa: F401
    except ImportError:
        missing.append("diffusers>=0.31")
    try:
        import gguf  # noqa: F401
    except ImportError:
        missing.append("gguf>=0.10")
    if missing:
        raise RuntimeError(
            "GGUF support is unavailable. Install the optional extra: "
            f"`pip install image-tools-mcp[gguf]`. Missing: {missing}"
        )


def gguf_quantization_config(compute_dtype: str = "bfloat16"):
    """Return a configured ``GGUFQuantizationConfig`` ready to pass to
    diffusers' ``from_single_file``. ``compute_dtype`` controls the dtype
    that dequantized values use during inference.

    Use ``bfloat16`` on Ampere+ (RTX 30/40, A100), ``float16`` for older
    cards that lack bfloat16 support.
    """
    _check_available()
    import torch
    from diffusers import GGUFQuantizationConfig

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16":     torch.bfloat16,
        "float16":  torch.float16,
        "fp16":     torch.float16,
        "float32":  torch.float32,
        "fp32":     torch.float32,
    }
    dtype = dtype_map.get(compute_dtype.lower())
    if dtype is None:
        raise ValueError(
            f"unknown compute_dtype {compute_dtype!r}; use "
            f"{sorted(dtype_map)}"
        )
    return GGUFQuantizationConfig(compute_dtype=dtype)


def download_gguf(repo_id: str, filename: str,
                  dest_dir: str | Path | None = None) -> str:
    """Pull a GGUF file from HuggingFace into a local cache. Returns the
    absolute path. Skips the download if the file already exists.

    Examples:
        download_gguf("city96/Qwen-Image-Edit-gguf", "qwen-image-edit-Q4_K_S.gguf")
        download_gguf("city96/FLUX.1-dev-gguf",       "flux1-dev-Q4_K_S.gguf")
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to fetch GGUF files. "
            "`pip install image-tools-mcp[gguf]`"
        ) from e

    if not filename.lower().endswith(".gguf"):
        raise ValueError(f"filename should end in .gguf, got {filename!r}")
    cache_dir = str(dest_dir) if dest_dir else None
    path = hf_hub_download(
        repo_id=repo_id, filename=filename, cache_dir=cache_dir,
    )
    return str(Path(path).resolve())
