"""Cross-cutting image utilities — watermark, QR, hash, compare, histogram,
colour replace, rounded corners, letterbox, white balance, annotate, glitch.

These don't fit cleanly into one of the dedicated modules but are common
enough that workflows would want them. Slim: no model loads, no heavy deps
beyond what we already have (PIL, numpy, opencv, optional qrcode).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ---- watermark ------------------------------------------------------------

_WM_POSITIONS = ("top_left", "top_right", "bottom_left", "bottom_right",
                 "center", "top", "bottom", "left", "right")


def _resolve_position(canvas_size: tuple[int, int],
                      wm_size: tuple[int, int],
                      position: str, padding: int) -> tuple[int, int]:
    """Return ``(x, y)`` for placing a wm of ``wm_size`` on a canvas of
    ``canvas_size`` at named ``position`` with ``padding`` from the edge."""
    cw, ch = canvas_size
    ww, wh = wm_size
    p = max(0, int(padding))
    cx = (cw - ww) // 2
    cy = (ch - wh) // 2
    placements = {
        "top_left":     (p, p),
        "top_right":    (cw - ww - p, p),
        "bottom_left":  (p, ch - wh - p),
        "bottom_right": (cw - ww - p, ch - wh - p),
        "center":       (cx, cy),
        "top":          (cx, p),
        "bottom":       (cx, ch - wh - p),
        "left":         (p, cy),
        "right":        (cw - ww - p, cy),
    }
    if position not in placements:
        raise ValueError(
            f"unknown position {position!r}. Choose from {_WM_POSITIONS}"
        )
    return placements[position]


def add_watermark(base: Image.Image, *,
                  text: str | None = None,
                  watermark_image: Image.Image | None = None,
                  position: str = "bottom_right",
                  padding: int = 20,
                  opacity: float = 0.5,
                  scale: float = 1.0,
                  text_color: tuple = (255, 255, 255, 255),
                  text_size: int = 24,
                  font_path: str | None = None,
                  ) -> Image.Image:
    """Add a text or image watermark to ``base``. Exactly one of ``text``
    or ``watermark_image`` must be provided.

    - ``position`` ∈ ``top_left | top_right | bottom_left | bottom_right |
      center | top | bottom | left | right``
    - ``padding`` is the edge inset in pixels
    - ``opacity`` 0-1 scales the watermark's alpha
    - ``scale`` resizes an image watermark; ignored for text
    """
    if (text is None) == (watermark_image is None):
        raise ValueError(
            "Provide exactly one of: text=... or watermark_image=..."
        )
    out = base.convert("RGBA")
    if text is not None:
        try:
            font = ImageFont.truetype(font_path or "arial.ttf", int(text_size))
        except Exception:
            font = ImageFont.load_default()
        # Measure text → create a transparent overlay sized for it.
        tmp = Image.new("RGBA", (1, 1))
        bx, by, ex, ey = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
        tw, th = (ex - bx), (ey - by)
        wm = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        d = ImageDraw.Draw(wm)
        # Shift by negative bbox origin so the glyphs align to (0, 0).
        d.text((-bx, -by), text, fill=text_color, font=font)
    else:
        wm = watermark_image.convert("RGBA")
        if abs(scale - 1.0) > 1e-3:
            wm = wm.resize(
                (max(1, int(wm.width * scale)),
                 max(1, int(wm.height * scale))),
                Image.LANCZOS,
            )

    if abs(opacity - 1.0) > 1e-3:
        a = wm.split()[-1].point(lambda v: int(v * max(0.0, min(1.0, opacity))))
        wm.putalpha(a)

    x, y = _resolve_position(out.size, wm.size, position, padding)
    out.alpha_composite(wm, dest=(x, y))
    return out


# ---- QR code --------------------------------------------------------------

def make_qr_code(text: str, *,
                 size: int = 256,
                 error_correction: str = "M",
                 fill_color: str = "#000000",
                 back_color: str = "#FFFFFF",
                 border: int = 4) -> Image.Image:
    """Generate a QR code as a PIL image of size ``size``×``size``.

    - ``error_correction`` ∈ ``L | M | Q | H`` (7 / 15 / 25 / 30 % recoverable
      data). Higher = denser code but tolerates more damage / overlay.
    - ``border`` is the quiet-zone width in modules (4 recommended by spec).
    """
    try:
        import qrcode
        from qrcode.constants import (
            ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H,
        )
    except ImportError as e:
        raise RuntimeError(
            "make_qr_code needs the `qrcode` package. `pip install qrcode[pil]`."
        ) from e
    ec_map = {
        "L": ERROR_CORRECT_L, "M": ERROR_CORRECT_M,
        "Q": ERROR_CORRECT_Q, "H": ERROR_CORRECT_H,
    }
    ec = ec_map.get(error_correction.upper())
    if ec is None:
        raise ValueError(f"error_correction must be L/M/Q/H, got {error_correction!r}")
    qr = qrcode.QRCode(
        version=None, error_correction=ec,
        box_size=10, border=max(0, int(border)),
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color=fill_color, back_color=back_color)
    img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((int(size), int(size)), Image.NEAREST)
    return img


# ---- perceptual hashes ----------------------------------------------------

def _ahash_dhash_phash(image: Image.Image, *, kind: str,
                      size: int = 8) -> str:
    """Compact pHash / aHash / dHash implementation. Returns a hex string."""
    gs = image.convert("L")
    if kind == "phash":
        # Larger DCT input → take top-left 8×8 of the DCT.
        n = 32
        small = gs.resize((n, n), Image.LANCZOS)
        arr = np.asarray(small, dtype=np.float32)
        # 2D DCT via two 1D passes (separable).
        from scipy.fft import dct  # may not be installed
        dct2 = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")
        sub = dct2[:size, :size]
        med = np.median(sub[1:].ravel())  # exclude DC term
        bits = (sub > med).flatten()
    elif kind == "ahash":
        small = gs.resize((size, size), Image.LANCZOS)
        arr = np.asarray(small, dtype=np.float32)
        bits = (arr > arr.mean()).flatten()
    elif kind == "dhash":
        small = gs.resize((size + 1, size), Image.LANCZOS)
        arr = np.asarray(small, dtype=np.float32)
        bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    else:
        raise ValueError(f"unknown hash kind {kind!r}")
    # Pack to hex.
    val = 0
    for b in bits:
        val = (val << 1) | int(bool(b))
    return f"{val:0{(size * size + 3) // 4}x}"


def perceptual_hash(image: Image.Image, *, kind: str = "phash",
                    size: int = 8) -> dict[str, Any]:
    """Compute a perceptual hash of ``image`` for duplicate / similarity
    detection. ``kind`` ∈ ``phash`` (DCT-based, most robust), ``ahash``
    (average), ``dhash`` (difference, very fast). Returns the hash as hex
    plus the bit length."""
    if kind == "phash":
        try:
            from scipy.fft import dct  # noqa: F401
        except ImportError:
            # Fall back to dhash if scipy isn't around — pHash needs DCT.
            kind = "dhash"
    h = _ahash_dhash_phash(image, kind=kind, size=size)
    return {"kind": kind, "hash": h, "bits": size * size}


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Hamming distance between two same-length hex hashes — number of bits
    that differ. < 5 typically means visually identical (8×8 hash); 5-10
    means similar; > 12 means different."""
    if len(hash_a) != len(hash_b):
        raise ValueError("hashes must be the same length")
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


# ---- image comparison ----------------------------------------------------

def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over the image (gaussian-windowed). Inputs are uint8 (H,W)
    or (H,W,C); the index is computed per-channel and averaged."""
    if a.ndim == 2:
        a = a[..., None]
        b = b[..., None]
    # Single global Gaussian window — fast approximation.
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    af = a.astype(np.float64); bf = b.astype(np.float64)
    mu_a = af.mean(axis=(0, 1), keepdims=True)
    mu_b = bf.mean(axis=(0, 1), keepdims=True)
    sigma_a = af.var(axis=(0, 1), keepdims=True)
    sigma_b = bf.var(axis=(0, 1), keepdims=True)
    sigma_ab = ((af - mu_a) * (bf - mu_b)).mean(axis=(0, 1), keepdims=True)
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)
    return float(np.mean(num / den))


def compare_images(image_a: Image.Image, image_b: Image.Image, *,
                   metrics: list[str] | None = None) -> dict[str, Any]:
    """Compute similarity metrics between two images. Both are converted to
    RGB and the smaller is resized to match the larger.

    ``metrics`` may include ``mse``, ``rmse``, ``psnr``, ``ssim``,
    ``diff_pct`` (percent of pixels differing by > 5 grey levels). Default
    runs all of them.
    """
    metrics = metrics or ["mse", "rmse", "psnr", "ssim", "diff_pct"]
    a = image_a.convert("RGB")
    b = image_b.convert("RGB")
    if a.size != b.size:
        target = max(a.size, b.size)
        a = a.resize(target, Image.BILINEAR)
        b = b.resize(target, Image.BILINEAR)
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    diff = arr_a.astype(np.int32) - arr_b.astype(np.int32)
    out: dict[str, Any] = {"width": a.size[0], "height": a.size[1]}
    if "mse" in metrics or "rmse" in metrics or "psnr" in metrics:
        mse = float((diff ** 2).mean())
        if "mse" in metrics:
            out["mse"] = round(mse, 4)
        if "rmse" in metrics:
            out["rmse"] = round(math.sqrt(mse), 4)
        if "psnr" in metrics:
            out["psnr"] = float("inf") if mse < 1e-6 else round(
                20 * math.log10(255.0 / math.sqrt(mse)), 4,
            )
    if "ssim" in metrics:
        out["ssim"] = round(_ssim(arr_a, arr_b), 4)
    if "diff_pct" in metrics:
        mag = np.max(np.abs(diff), axis=-1)
        out["diff_pct"] = round(float((mag > 5).mean()) * 100.0, 3)
    return out


def diff_image(image_a: Image.Image, image_b: Image.Image, *,
               highlight_color: tuple[int, int, int] = (255, 0, 255),
               alpha: float = 0.6) -> Image.Image:
    """Visualise the difference between two images — pixels that differ
    are tinted in ``highlight_color`` at ``alpha`` opacity over the first
    image. Useful for QA / regression screenshots."""
    a = image_a.convert("RGB")
    b = image_b.convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size, Image.BILINEAR)
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    mag = np.max(np.abs(arr_a - arr_b), axis=-1)
    mask = (mag > 3) * float(alpha)  # < 3 levels = noise
    weight = mask[..., None]
    tint = np.array(highlight_color, dtype=np.float32)[None, None, :]
    out = arr_a * (1.0 - weight) + tint * weight
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


# ---- histogram -----------------------------------------------------------

def histogram(image: Image.Image, *, bins: int = 256,
              channels: list[str] | None = None) -> dict[str, Any]:
    """Return histogram data — counts per bin per channel. ``channels`` ∈
    ``r``, ``g``, ``b``, ``a``, ``l`` (luminance). Default: ``[r, g, b, l]``."""
    channels = [c.lower() for c in (channels or ["r", "g", "b", "l"])]
    src = image.convert("RGBA") if image.mode == "RGBA" else image.convert("RGB")
    arr = np.asarray(src)
    out = {}
    for ch in channels:
        if ch in ("r", "g", "b"):
            if arr.shape[-1] < 3:
                continue
            idx = {"r": 0, "g": 1, "b": 2}[ch]
            h, _ = np.histogram(arr[..., idx], bins=int(bins), range=(0, 256))
        elif ch == "a":
            if src.mode != "RGBA":
                continue
            h, _ = np.histogram(arr[..., 3], bins=int(bins), range=(0, 256))
        elif ch == "l":
            l = src.convert("L")
            h, _ = np.histogram(np.asarray(l), bins=int(bins), range=(0, 256))
        else:
            continue
        out[ch] = h.tolist()
    return {"bins": int(bins), "channels": out, "total_pixels": int(arr.shape[0] * arr.shape[1])}


# ---- colour replace ------------------------------------------------------

def color_replace(image: Image.Image, *,
                  from_color: tuple[int, int, int],
                  to_color: tuple[int, int, int, int],
                  tolerance: int = 16,
                  feather: float = 0.0) -> Image.Image:
    """Replace pixels close to ``from_color`` with ``to_color`` (which may
    include alpha — pass ``(r, g, b, a)``).

    ``tolerance`` is the max per-channel difference in RGB. ``feather``
    > 0 softens the transition (Gaussian-blurs the selection mask)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    target = np.array(from_color[:3], dtype=np.int16)
    mag = np.max(np.abs(rgb - target), axis=-1)
    match = (mag <= int(tolerance)).astype(np.float32)
    if feather > 0:
        mask_pil = Image.fromarray((match * 255).astype(np.uint8), mode="L")
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=float(feather)))
        match = np.asarray(mask_pil, dtype=np.float32) / 255.0
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32)
    tint = np.array(list(to_color[:3]) + [255 if len(to_color) < 4 else to_color[3]],
                    dtype=np.float32)[None, None, :]
    w = match[..., None]
    out = arr * (1.0 - w) + tint * w
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGBA")


# ---- rounded corners -----------------------------------------------------

def rounded_corners(image: Image.Image, *, radius: int = 20,
                    corners: str = "all") -> Image.Image:
    """Round the corners by ``radius`` pixels, producing an RGBA image
    where the corner pixels become transparent.

    ``corners`` ∈ ``all | top | bottom | left | right`` to round only
    a subset (or by individual corner via ``top_left | top_right |
    bottom_left | bottom_right``).
    """
    if radius < 1:
        return image.convert("RGBA")
    rgba = image.convert("RGBA")
    w, h = rgba.size
    r = min(int(radius), min(w, h) // 2)
    mask = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(mask)
    # Stamp transparent quarter-circles at each requested corner.
    sets = {
        "all": ("top_left", "top_right", "bottom_left", "bottom_right"),
        "top": ("top_left", "top_right"),
        "bottom": ("bottom_left", "bottom_right"),
        "left": ("top_left", "bottom_left"),
        "right": ("top_right", "bottom_right"),
        "top_left": ("top_left",),
        "top_right": ("top_right",),
        "bottom_left": ("bottom_left",),
        "bottom_right": ("bottom_right",),
    }
    targets = sets.get(corners)
    if targets is None:
        raise ValueError(f"unknown corners {corners!r}. Choose from {list(sets)}")
    for c in targets:
        if c == "top_left":
            d.rectangle((0, 0, r, r), fill=0)
            d.ellipse((0, 0, 2 * r, 2 * r), fill=255)
        elif c == "top_right":
            d.rectangle((w - r, 0, w, r), fill=0)
            d.ellipse((w - 2 * r, 0, w, 2 * r), fill=255)
        elif c == "bottom_left":
            d.rectangle((0, h - r, r, h), fill=0)
            d.ellipse((0, h - 2 * r, 2 * r, h), fill=255)
        elif c == "bottom_right":
            d.rectangle((w - r, h - r, w, h), fill=0)
            d.ellipse((w - 2 * r, h - 2 * r, w, h), fill=255)
    # Combine the round-corner mask with the source alpha.
    src_alpha = rgba.split()[-1]
    from PIL import ImageChops
    new_alpha = ImageChops.multiply(src_alpha, mask)
    r_, g_, b_, _ = rgba.split()
    return Image.merge("RGBA", (r_, g_, b_, new_alpha))


# ---- letterbox -----------------------------------------------------------

def letterbox(image: Image.Image, *, aspect: float,
              fill_color: tuple = (0, 0, 0, 255)) -> Image.Image:
    """Extend the canvas to match ``aspect = width / height``, adding bars
    (filled with ``fill_color``) on the short axis. Opposite of
    ``smart_crop_to_aspect`` — preserves all of the source content."""
    if aspect <= 0:
        raise ValueError(f"aspect must be > 0, got {aspect}")
    w, h = image.size
    cur = w / h
    if abs(cur - aspect) < 1e-3:
        return image.copy()
    if cur < aspect:
        # Source is too tall → add bars left+right.
        new_w = int(round(h * aspect))
        new_h = h
    else:
        new_w = w
        new_h = int(round(w / aspect))
    out = Image.new(image.mode if image.mode in ("RGB", "RGBA") else "RGBA",
                    (new_w, new_h), fill_color)
    out.paste(image, ((new_w - w) // 2, (new_h - h) // 2))
    return out


# ---- white balance -------------------------------------------------------

def white_balance(image: Image.Image, *, method: str = "gray_world",
                  strength: float = 1.0) -> Image.Image:
    """Auto white balance / colour-cast removal.

    Methods:
      - ``gray_world``: assumes scene's average colour is neutral grey,
        scales each channel to that. Best for general use.
      - ``white_patch``: assumes the brightest pixels should be white.
        Better for scenes with a known bright reference.
      - ``simplest_cb`` (Limare's "simplest colour balance"): stretches
        each channel's histogram to cover 1..99 % quantiles. Most
        aggressive — handles strong casts well.

    ``strength`` 0-1 blends the corrected result with the original.
    """
    has_alpha = image.mode == "RGBA"
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    if method == "gray_world":
        means = arr.mean(axis=(0, 1))
        target = means.mean()
        scale = np.where(means > 1e-9, target / means, 1.0)
        corrected = arr * scale[None, None, :]
    elif method == "white_patch":
        maxes = arr.max(axis=(0, 1))
        scale = np.where(maxes > 1e-9, 1.0 / maxes, 1.0)
        corrected = arr * scale[None, None, :]
    elif method == "simplest_cb":
        corrected = arr.copy()
        for ch in range(3):
            lo = np.quantile(arr[..., ch], 0.01)
            hi = np.quantile(arr[..., ch], 0.99)
            if hi > lo:
                corrected[..., ch] = np.clip((arr[..., ch] - lo) / (hi - lo), 0, 1)
    else:
        raise ValueError(
            f"unknown method {method!r}. Choose from "
            "gray_world, white_patch, simplest_cb"
        )
    s = max(0.0, min(1.0, float(strength)))
    blended = np.clip(arr * (1.0 - s) + corrected * s, 0.0, 1.0)
    out_arr = (blended * 255.0).astype(np.uint8)
    out = Image.fromarray(out_arr, mode="RGB")
    if has_alpha:
        return Image.merge("RGBA", (*out.split(), image.split()[-1]))
    return out


# ---- annotate ------------------------------------------------------------

def annotate(image: Image.Image, items: list[dict], *,
             font_size: int = 14, default_color: str = "#FF00FF",
             ) -> Image.Image:
    """Draw annotations on a copy of ``image``. ``items`` is a list of
    dicts, one per annotation. Each item has a ``kind`` and per-kind args:

      - ``rect``: ``{kind: "rect", bbox: [x1,y1,x2,y2], label?, color?,
        width?}``
      - ``arrow``: ``{kind: "arrow", x1, y1, x2, y2, color?, width?,
        head_size?}``
      - ``label``: ``{kind: "label", x, y, text, color?, bg?}`` (text
        with optional background pill)
      - ``circle``: ``{kind: "circle", x, y, radius, color?, width?}``

    Useful for AI-output overlays, annotated demos, debug screenshots."""
    out = image.convert("RGBA").copy()
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", int(font_size))
    except Exception:
        font = ImageFont.load_default()
    for item in items:
        kind = item.get("kind", "rect")
        color = item.get("color", default_color)
        if isinstance(color, (list, tuple)):
            color = tuple(int(c) for c in color)
        if kind == "rect":
            x1, y1, x2, y2 = item["bbox"]
            d.rectangle([x1, y1, x2, y2], outline=color,
                        width=int(item.get("width", 2)))
            label = item.get("label")
            if label:
                tx, ty = int(x1) + 3, int(y1) + 3
                bx, by, ex, ey = d.textbbox((tx, ty), label, font=font)
                d.rectangle([bx - 2, by - 2, ex + 2, ey + 2],
                            fill=(0, 0, 0, 200))
                d.text((tx, ty), label, fill=color, font=font)
        elif kind == "arrow":
            x1, y1, x2, y2 = (item["x1"], item["y1"], item["x2"], item["y2"])
            w = int(item.get("width", 2))
            head = float(item.get("head_size", 12))
            d.line([x1, y1, x2, y2], fill=color, width=w)
            # Arrow head — two short lines at the tip.
            ang = math.atan2(y2 - y1, x2 - x1)
            h_ang = math.radians(25)
            hx1 = x2 - head * math.cos(ang - h_ang)
            hy1 = y2 - head * math.sin(ang - h_ang)
            hx2 = x2 - head * math.cos(ang + h_ang)
            hy2 = y2 - head * math.sin(ang + h_ang)
            d.line([x2, y2, hx1, hy1], fill=color, width=w)
            d.line([x2, y2, hx2, hy2], fill=color, width=w)
        elif kind == "label":
            x, y, text = item["x"], item["y"], item.get("text", "")
            bg = item.get("bg", (0, 0, 0, 200))
            if isinstance(bg, (list, tuple)):
                bg = tuple(int(c) for c in bg)
            bx, by, ex, ey = d.textbbox((x, y), text, font=font)
            d.rectangle([bx - 3, by - 3, ex + 3, ey + 3], fill=bg)
            d.text((x, y), text, fill=color, font=font)
        elif kind == "circle":
            x, y, r = item["x"], item["y"], item["radius"]
            d.ellipse([x - r, y - r, x + r, y + r], outline=color,
                      width=int(item.get("width", 2)))
        else:
            raise ValueError(f"unknown annotate kind {kind!r}")
    return out


# ---- glitch effect -------------------------------------------------------

def glitch_effect(image: Image.Image, *, intensity: float = 0.5,
                  seed: int | None = None) -> Image.Image:
    """Quick glitch effect: random horizontal row shifts + channel
    misalignment (red/blue shift). ``intensity`` 0-1 scales the effect."""
    rng = np.random.default_rng(seed)
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    h, w = rgb.shape[:2]
    # 1. RGB channel shift: move R left, B right by ~intensity * 8 px.
    sx = int(round(intensity * 8))
    if sx:
        r = np.roll(rgb[..., 0], -sx, axis=1)
        b = np.roll(rgb[..., 2], sx, axis=1)
        rgb = np.stack([r, rgb[..., 1], b], axis=-1)
    # 2. Random row shifts on a few stripes.
    n_stripes = int(round(intensity * 12))
    for _ in range(n_stripes):
        y0 = rng.integers(0, h)
        thickness = rng.integers(1, max(2, int(intensity * 15)))
        y1 = min(h, int(y0) + int(thickness))
        shift = int(rng.integers(-int(intensity * 60), int(intensity * 60) + 1))
        rgb[y0:y1] = np.roll(rgb[y0:y1], shift, axis=1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
