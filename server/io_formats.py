"""Format-aware image load/save.

Pillow covers the common formats out of the box. We extend it with:

- ``pillow-heif`` for HEIC/HEIF/AVIF (registers itself as Pillow plugin).
- ``cairosvg`` for SVG, which Pillow cannot rasterize on its own.
- ``rawpy`` for camera RAW (CR2/NEF/ARW/DNG/...) — optional ``[raw]`` extra.

Format detection is by file extension on save and by magic bytes on load
(Pillow's default). Extensions are normalized to lowercase without leading dot.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from PIL import Image, ExifTags

# Register HEIF/AVIF plugin. Safe to import multiple times.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    try:
        pillow_heif.register_avif_opener()
    except Exception:
        # Older pillow-heif versions don't expose register_avif_opener; HEIF
        # opener handles AVIF too. Swallow.
        pass
    _HEIF_OK = True
except Exception as e:
    _HEIF_OK = False
    _HEIF_ERR = f"{type(e).__name__}: {e}"


def _ext(path: str | os.PathLike[str]) -> str:
    return Path(path).suffix.lower().lstrip(".")


# Extensions that need special-case handling rather than Pillow's default
# ``Image.open`` path.
_RAW_EXTS = {
    "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2", "dng",
    "raf", "rw2", "orf", "pef", "ptx", "x3f", "3fr", "mef",
    "mos", "kdc", "dcr", "erf", "iiq",
}

# SVG is text; goes through cairosvg. The user can pass either path or content.
_SVG_EXTS = {"svg", "svgz"}


def _load_svg(path: str, *, target_width: int | None = None,
              target_height: int | None = None) -> Image.Image:
    """Rasterize an SVG file to a Pillow Image via cairosvg."""
    import cairosvg  # type: ignore

    png_bytes = cairosvg.svg2png(
        url=str(Path(path).resolve()),
        output_width=target_width,
        output_height=target_height,
    )
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def _load_raw(path: str) -> Image.Image:
    """Demosaic a camera RAW file via rawpy.

    Returns an RGB Image at the camera's native resolution using rawpy's
    default postprocessing. For finer control (white balance, exposure), the
    user should convert externally; this is a "give me a viewable image"
    convenience path.
    """
    try:
        import rawpy  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "rawpy is not installed. Install with `pip install image-tools-mcp[raw]` "
            "to handle camera RAW formats."
        ) from e
    import numpy as np

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
    return Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")


def load_image(path: str, *, svg_width: int | None = None,
               svg_height: int | None = None) -> Image.Image:
    """Open ``path`` and return a Pillow Image.

    SVG and camera RAW are dispatched to specialized readers; everything else
    goes through ``PIL.Image.open`` which auto-detects via magic bytes.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"no such file: {path}")
    ext = _ext(path)
    if ext in _SVG_EXTS:
        return _load_svg(path, target_width=svg_width, target_height=svg_height)
    if ext in _RAW_EXTS:
        return _load_raw(path)
    img = Image.open(path)
    img.load()
    return img


# ---- save ------------------------------------------------------------------

_PIL_FORMAT_BY_EXT = {
    "jpg": "JPEG", "jpeg": "JPEG", "jpe": "JPEG", "jfif": "JPEG",
    "png": "PNG",
    "gif": "GIF",
    "bmp": "BMP", "dib": "BMP",
    "tif": "TIFF", "tiff": "TIFF",
    "webp": "WEBP",
    "ico": "ICO",
    "tga": "TGA",
    "ppm": "PPM", "pgm": "PPM", "pbm": "PPM", "pnm": "PPM",
    "pcx": "PCX",
    "dds": "DDS",
    "heic": "HEIF", "heif": "HEIF",
    "avif": "AVIF",
}


def _coerce_for_format(img: Image.Image, fmt: str) -> Image.Image:
    """JPEG and a few others reject alpha or palette modes — flatten or convert
    rather than letting Pillow raise on save."""
    fmt = fmt.upper()
    if fmt == "JPEG":
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            mask = img.split()[-1]
            bg.paste(img.convert("RGB"), mask=mask)
            return bg
        if img.mode != "RGB" and img.mode != "L" and img.mode != "CMYK":
            return img.convert("RGB")
    if fmt == "GIF" and img.mode not in ("P", "L", "RGB", "RGBA"):
        return img.convert("RGBA")
    if fmt == "ICO" and img.mode != "RGBA":
        return img.convert("RGBA")
    return img


def save_image(img: Image.Image, path: str, *, format: str | None = None,
               quality: int | None = None, optimize: bool = True) -> dict[str, Any]:
    """Write ``img`` to ``path``. Format inferred from extension if not given.

    Returns a small summary dict suitable as a tool result.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ext = _ext(path)

    if ext in _SVG_EXTS:
        raise ValueError(
            "Writing SVG is not supported. SVG is vector; rasterize on load only."
        )
    if ext in _RAW_EXTS:
        raise ValueError(
            "Writing camera RAW is not supported. RAW formats encode sensor data, "
            "not a developed image. Save as PNG/TIFF instead."
        )

    fmt = (format or _PIL_FORMAT_BY_EXT.get(ext) or ext).upper()
    out = _coerce_for_format(img, fmt)

    save_kwargs: dict[str, Any] = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs["quality"] = quality if quality is not None else 92
        save_kwargs["optimize"] = optimize
        save_kwargs["progressive"] = True
    elif fmt == "WEBP":
        save_kwargs["quality"] = quality if quality is not None else 90
        save_kwargs["method"] = 6
    elif fmt == "PNG":
        save_kwargs["optimize"] = optimize
    elif fmt in ("HEIF", "AVIF"):
        if not _HEIF_OK:
            raise RuntimeError(f"pillow-heif unavailable: {_HEIF_ERR}")
        save_kwargs["quality"] = quality if quality is not None else 80

    out.save(path, **save_kwargs)
    return {
        "path": str(Path(path).resolve()),
        "format": fmt,
        "width": out.width,
        "height": out.height,
        "mode": out.mode,
        "size_bytes": Path(path).stat().st_size,
    }


# ---- info ------------------------------------------------------------------

def _exif_dict(img: Image.Image) -> dict[str, Any]:
    try:
        raw = img.getexif()
    except Exception:
        return {}
    if not raw:
        return {}
    decoded: dict[str, Any] = {}
    for tag_id, value in raw.items():
        name = ExifTags.TAGS.get(tag_id, str(tag_id))
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="replace")
            except Exception:
                value = repr(value)
        decoded[name] = value
    return decoded


def image_info(path: str) -> dict[str, Any]:
    img = load_image(path)
    return {
        "path": str(Path(path).resolve()),
        "format": img.format or _ext(path).upper(),
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "has_alpha": img.mode in ("RGBA", "LA", "PA") or "A" in img.getbands(),
        "n_frames": getattr(img, "n_frames", 1),
        "is_animated": getattr(img, "is_animated", False),
        "exif": _exif_dict(img),
        "size_bytes": Path(path).stat().st_size,
    }


def supported_formats() -> dict[str, Any]:
    return {
        "load": sorted(set(_PIL_FORMAT_BY_EXT) | _SVG_EXTS | _RAW_EXTS),
        "save": sorted(_PIL_FORMAT_BY_EXT),
        "heif_avif_ok": _HEIF_OK,
        "raw_ok": _has_module("rawpy"),
        "svg_ok": _has_module("cairosvg"),
    }


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False
