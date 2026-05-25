"""Format conversions that don't fit the simple load→save shape.

Six conversion classes are covered here:

- **Animation frames** — extract per-frame stills from animated GIF/WebP/APNG,
  or build an animation from a list of frame files. Output format is inferred
  from the destination extension.

- **ICO** — Windows icon files carry multiple resolutions in one container.
  ``build_ico`` packs several PNGs (or one image rendered at each requested
  size) into a single .ico; ``split_ico`` extracts every embedded size.

- **PDF** — ``pdf_to_images`` rasterizes each page via pypdfium2 (pure-Python
  wheel, no Poppler/GhostScript dependency); ``images_to_pdf`` combines a
  list of images into a multi-page PDF using Pillow's built-in PDF writer.

- **Color mode** — explicit conversions to RGB/RGBA/L/LA/P/CMYK/1 with
  dither control. Useful for "force palette for GIF", "1-bit for fax",
  "CMYK for print pipelines", etc.

- **Metadata** — strip everything (EXIF, ICC profile, XMP, GPS) or transplant
  the EXIF block from one file onto another (e.g. preserve camera info
  through a destructive edit).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from . import io_formats


# ---------------------------------------------------------------- animation

_ANIMATED_EXTS = {"gif", "webp", "apng", "png"}


def extract_frames(src_path: str, dst_dir: str, *,
                   format: str = "png",
                   filename_template: str = "frame_{i:04d}") -> dict[str, Any]:
    """Save every frame of an animated image to ``dst_dir`` as separate files.

    Falls through gracefully for single-frame inputs: writes one file. The
    ``filename_template`` uses ``{i}`` (0-based frame index); the extension
    comes from ``format``.
    """
    img = io_formats.load_image(src_path)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    fmt = format.lower().lstrip(".")
    n_frames = getattr(img, "n_frames", 1)
    written: list[str] = []
    for i in range(n_frames):
        try:
            img.seek(i)
        except EOFError:
            break
        out = dst / (filename_template.format(i=i) + "." + fmt)
        # ``.copy()`` because seek mutates ``img`` in place.
        info = io_formats.save_image(img.copy(), str(out))
        written.append(info["path"])
    return {"frames": len(written), "files": written}


def build_animation(frame_paths: Sequence[str], dst_path: str, *,
                    fps: float = 10.0, loop: int = 0,
                    quality: int | None = None,
                    optimize: bool = True) -> dict[str, Any]:
    """Build an animated GIF / WebP / APNG from a list of frame files.

    Output format is inferred from ``dst_path``'s extension. All frames are
    resized to the first frame's dimensions if they differ — most encoders
    refuse mismatched sizes otherwise.

    ``fps`` controls inter-frame delay; ``loop=0`` means loop forever, any
    positive value caps the loop count (GIF only).
    """
    if not frame_paths:
        raise ValueError("build_animation: frame_paths is empty")
    ext = Path(dst_path).suffix.lower().lstrip(".")
    if ext not in _ANIMATED_EXTS:
        raise ValueError(
            f"output must be one of {sorted(_ANIMATED_EXTS)} (got .{ext})"
        )

    frames = [io_formats.load_image(p) for p in frame_paths]
    first = frames[0]
    target_size = first.size
    normalized = []
    for f in frames:
        if f.size != target_size:
            f = f.resize(target_size, Image.LANCZOS)
        normalized.append(f)

    duration_ms = int(round(1000.0 / max(0.1, float(fps))))
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

    save_kwargs: dict[str, Any] = {
        "save_all": True,
        "append_images": normalized[1:],
        "duration": duration_ms,
        "loop": int(loop),
        "optimize": optimize,
    }

    fmt = ext.upper()
    if ext == "gif":
        # Encode palette for size; GIF needs P or L mode frames.
        normalized = [f.convert("RGBA").convert("P", palette=Image.ADAPTIVE)
                      for f in normalized]
        save_kwargs["append_images"] = normalized[1:]
        save_kwargs["disposal"] = 2  # restore-to-background between frames
    elif ext == "webp":
        save_kwargs["quality"] = quality if quality is not None else 80
        save_kwargs["method"] = 6
        save_kwargs.pop("optimize", None)
    elif ext in ("apng", "png"):
        # Pillow's PNG plugin writes APNG when ``save_all=True``.
        fmt = "PNG"
        save_kwargs.pop("optimize", None)

    normalized[0].save(dst_path, format=fmt, **save_kwargs)
    return {
        "path": str(Path(dst_path).resolve()),
        "format": fmt,
        "frames": len(normalized),
        "fps": float(fps),
        "duration_ms_per_frame": duration_ms,
        "size_bytes": Path(dst_path).stat().st_size,
    }


# ---------------------------------------------------------------- ICO

_ICO_DEFAULT_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                      (128, 128), (256, 256)]


def build_ico(src_paths: Sequence[str], dst_path: str, *,
              sizes: Sequence[int] | None = None) -> dict[str, Any]:
    """Pack one or more images into a multi-resolution Windows .ico file.

    Two modes:
    - Multiple source paths: each becomes one size in the ICO. The encoder
      reads each at its own dimensions.
    - Single source path + ``sizes``: render the same image at every requested
      square size. ``sizes`` is a list of ints, e.g. ``[16, 32, 48, 256]``.
      If omitted, defaults to Windows' standard set.
    """
    if not src_paths:
        raise ValueError("build_ico: src_paths is empty")

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

    if len(src_paths) == 1:
        # Single image → multiple square sizes via Pillow's ICO encoder.
        base = io_formats.load_image(src_paths[0]).convert("RGBA")
        size_pairs: list[tuple[int, int]] = (
            [(s, s) for s in sizes] if sizes else _ICO_DEFAULT_SIZES
        )
        base.save(dst_path, format="ICO", sizes=size_pairs)
        return {
            "path": str(Path(dst_path).resolve()),
            "format": "ICO",
            "sizes": [list(s) for s in size_pairs],
            "size_bytes": Path(dst_path).stat().st_size,
        }

    # Multiple sources: stack as append_images, letting each provide its own
    # dimensions. Pillow's ICO writer keys off each image's size.
    images = [io_formats.load_image(p).convert("RGBA") for p in src_paths]
    base = images[0]
    base.save(dst_path, format="ICO", append_images=images[1:])
    return {
        "path": str(Path(dst_path).resolve()),
        "format": "ICO",
        "sizes": [[im.width, im.height] for im in images],
        "size_bytes": Path(dst_path).stat().st_size,
    }


def split_ico(src_path: str, dst_dir: str, *,
              format: str = "png") -> dict[str, Any]:
    """Extract every embedded resolution from a .ico into individual files.

    Files are named ``icon_{w}x{h}.{ext}``. Pillow's ICO plugin exposes each
    embedded size via ``.ico.frame(i)`` or ``.size`` after ``.seek``.
    """
    ico = Image.open(src_path)
    if ico.format != "ICO":
        raise ValueError(f"{src_path} is not an ICO file (detected {ico.format})")
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    ext = format.lower().lstrip(".")

    # Pillow's ICO plugin keeps the list of frames on ``.ico.sizes()``.
    sizes = sorted(ico.ico.sizes())
    written: list[str] = []
    for w, h in sizes:
        frame = ico.ico.getimage((w, h))
        out = dst / f"icon_{w}x{h}.{ext}"
        info = io_formats.save_image(frame, str(out))
        written.append(info["path"])
    return {"count": len(written), "files": written, "sizes": [list(s) for s in sizes]}


# ---------------------------------------------------------------- PDF

def pdf_to_images(src_path: str, dst_dir: str, *,
                  dpi: int = 200, format: str = "png",
                  pages: str | None = None) -> dict[str, Any]:
    """Rasterize each page of a PDF to an image via pypdfium2.

    ``dpi`` controls resolution (300+ for print-quality, 96-150 for screen).
    ``pages`` accepts a 1-based range like ``"1-3,7,10-12"``; omit for all.
    Files are named ``page_{i:04d}.{ext}`` (1-based).
    """
    import pypdfium2 as pdfium  # type: ignore

    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    ext = format.lower().lstrip(".")

    pdf = pdfium.PdfDocument(src_path)
    try:
        n = len(pdf)
        selected = _parse_page_range(pages, n) if pages else list(range(1, n + 1))
        scale = dpi / 72.0  # pdfium renders at 72 dpi by default

        written: list[str] = []
        for page_no in selected:
            page = pdf[page_no - 1]
            pil = page.render(scale=scale).to_pil()
            out = dst / f"page_{page_no:04d}.{ext}"
            info = io_formats.save_image(pil, str(out))
            written.append(info["path"])
        return {"pages_rendered": len(written), "total_pages": n,
                "dpi": dpi, "files": written}
    finally:
        pdf.close()


def _parse_page_range(spec: str, total: int) -> list[int]:
    """Parse "1-3,7,10-12" → [1,2,3,7,10,11,12]. Clamps to ``total``."""
    pages: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            lo = max(1, int(a))
            hi = min(total, int(b))
            for p in range(lo, hi + 1):
                pages.add(p)
        else:
            p = int(chunk)
            if 1 <= p <= total:
                pages.add(p)
    return sorted(pages)


def images_to_pdf(src_paths: Sequence[str], dst_path: str, *,
                  quality: int = 92) -> dict[str, Any]:
    """Combine images into a multi-page PDF, one image per page.

    Uses Pillow's built-in PDF writer — no external dep. Alpha is flattened
    onto white (PDFs can't carry alpha through Pillow's writer).
    """
    if not src_paths:
        raise ValueError("images_to_pdf: src_paths is empty")
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

    pages = []
    for p in src_paths:
        img = io_formats.load_image(p)
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        pages.append(img)

    pages[0].save(dst_path, format="PDF", save_all=True,
                  append_images=pages[1:], resolution=300.0, quality=quality)
    return {
        "path": str(Path(dst_path).resolve()),
        "pages": len(pages),
        "size_bytes": Path(dst_path).stat().st_size,
    }


# ---------------------------------------------------------------- color mode

_MODES = {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "I", "F"}


def convert_mode(src_path: str, dst_path: str, mode: str, *,
                 dither: bool = True, palette_size: int = 256,
                 quality: int | None = None) -> dict[str, Any]:
    """Convert color mode and save.

    Modes:
    - ``1``    1-bit black/white (Floyd-Steinberg if ``dither=True``).
    - ``L``    8-bit grayscale.
    - ``LA``   grayscale + alpha.
    - ``P``    palette (indexed). Use ``palette_size`` (max 256).
    - ``RGB``  24-bit color.
    - ``RGBA`` 32-bit color + alpha.
    - ``CMYK`` 4-channel print color (no alpha).
    - ``I``    32-bit integer pixels (scientific).
    - ``F``    32-bit float pixels (HDR-ish).
    """
    if mode not in _MODES:
        raise ValueError(
            f"unknown mode {mode!r}. Choose from {sorted(_MODES)}."
        )
    img = io_formats.load_image(src_path)
    d = Image.FLOYDSTEINBERG if dither else Image.NONE

    if mode == "P":
        out = img.convert("P", palette=Image.ADAPTIVE,
                          colors=max(2, min(256, int(palette_size))),
                          dither=d)
    elif mode == "1":
        out = img.convert("1", dither=d)
    else:
        out = img.convert(mode)

    return io_formats.save_image(out, dst_path, quality=quality)


# ---------------------------------------------------------------- metadata

def strip_metadata(src_path: str, dst_path: str, *,
                   quality: int | None = None) -> dict[str, Any]:
    """Re-encode ``src_path`` to ``dst_path`` dropping EXIF, ICC profile,
    XMP, and any other metadata. Pixel data is preserved.

    Trick: ``paste`` into a fresh Image so the .info dict / EXIF / ICC don't
    ride along (they live on the source object, not in pixel data)."""
    img = io_formats.load_image(src_path)
    clean = Image.new(img.mode, img.size)
    clean.paste(img)
    return io_formats.save_image(clean, dst_path, quality=quality)


def copy_metadata(meta_src: str, image_src: str, dst_path: str, *,
                  quality: int | None = None) -> dict[str, Any]:
    """Transplant the EXIF block from ``meta_src`` onto ``image_src`` and save.

    Useful after a destructive edit (rotation, crop) where you want to retain
    the camera's capture data. Only JPEG / TIFF / WebP / HEIF can carry EXIF.
    """
    meta_img = io_formats.load_image(meta_src)
    image = io_formats.load_image(image_src)
    exif = meta_img.getexif()
    if not exif:
        raise ValueError(f"no EXIF data in {meta_src}")

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    fmt = io_formats._PIL_FORMAT_BY_EXT.get(
        Path(dst_path).suffix.lower().lstrip("."), None
    )
    save_kwargs: dict[str, Any] = {"exif": exif.tobytes()}
    if quality is not None:
        save_kwargs["quality"] = quality

    coerced = io_formats._coerce_for_format(image, fmt or "JPEG")
    if fmt:
        save_kwargs["format"] = fmt
    coerced.save(dst_path, **save_kwargs)
    return {
        "path": str(Path(dst_path).resolve()),
        "format": fmt or "JPEG",
        "exif_tags_copied": len(exif),
        "size_bytes": Path(dst_path).stat().st_size,
    }
