"""Photoshop Image > Adjustments — numpy-based color/tone correction.

Every function takes a Pillow Image, returns a new Image. Alpha is preserved.
Math runs in 0..1 float space then clips to 0..255 at the end. A consistent
``_to_rgba_float`` / ``_back_to_pil`` pair keeps the boilerplate quiet.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _to_rgba_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0


def _back_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0) * 255.0 + 0.5
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


# ---------------------------------------------------------------- HSL

def _rgb_to_hsv_arr(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB → HSV. ``rgb`` is HxWx3 in 0..1; returns HxWx3 in 0..1."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    v = mx
    delta = mx - mn
    s = np.where(mx == 0, 0.0, delta / np.maximum(mx, 1e-9))
    # Hue
    h = np.zeros_like(mx)
    mask = delta > 1e-9
    rc = (mx - r) / np.maximum(delta, 1e-9)
    gc = (mx - g) / np.maximum(delta, 1e-9)
    bc = (mx - b) / np.maximum(delta, 1e-9)
    h_red = (bc - gc)
    h_grn = 2.0 + (rc - bc)
    h_blu = 4.0 + (gc - rc)
    h = np.where(r == mx, h_red, np.where(g == mx, h_grn, h_blu))
    h = (h / 6.0) % 1.0
    h = np.where(mask, h, 0.0)
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb_arr(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6
    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def hue_saturation_lightness(img: Image.Image, *, hue: float = 0.0,
                             saturation: float = 0.0,
                             lightness: float = 0.0) -> Image.Image:
    """Photoshop Hue/Saturation. All three inputs are in -100..+100 percent.

    - ``hue``: rotate hue around the wheel (±180° at ±100).
    - ``saturation``: -100 fully desaturates, +100 doubles saturation.
    - ``lightness``: -100 to black, +100 to white (clipped).
    """
    arr = _to_rgba_float(img)
    rgb = arr[..., :3]
    hsv = _rgb_to_hsv_arr(rgb)
    hsv[..., 0] = (hsv[..., 0] + (float(hue) / 200.0)) % 1.0  # -100..100 → -0.5..0.5
    sat_mult = 1.0 + float(saturation) / 100.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_mult, 0.0, 1.0)
    new_rgb = _hsv_to_rgb_arr(hsv)
    # Lightness as additive shift (Photoshop's behavior is closer to this than
    # multiplicative when scaling toward black/white).
    light = float(lightness) / 100.0
    if light > 0:
        new_rgb = new_rgb + (1.0 - new_rgb) * light
    elif light < 0:
        new_rgb = new_rgb + new_rgb * light
    arr[..., :3] = new_rgb
    return _back_to_pil(arr)


# ---------------------------------------------------------------- Levels

def levels(img: Image.Image, *, in_black: int = 0, in_white: int = 255,
           gamma: float = 1.0, out_black: int = 0,
           out_white: int = 255) -> Image.Image:
    """Photoshop Levels. Input black/white clip the histogram, gamma reshapes
    midtones (γ>1 darkens, γ<1 brightens — same as PS), output black/white
    rescale the result to a narrower range."""
    arr = _to_rgba_float(img)
    rgb = arr[..., :3]
    ib = in_black / 255.0
    iw = in_white / 255.0
    span = max(iw - ib, 1e-9)
    rgb = np.clip((rgb - ib) / span, 0.0, 1.0)
    if gamma != 1.0:
        rgb = np.power(rgb, 1.0 / max(gamma, 1e-6))
    ob = out_black / 255.0
    ow = out_white / 255.0
    rgb = rgb * (ow - ob) + ob
    arr[..., :3] = rgb
    return _back_to_pil(arr)


# ---------------------------------------------------------------- Curves

def curves(img: Image.Image, *,
           rgb_curve: list[tuple[float, float]] | None = None,
           r_curve: list[tuple[float, float]] | None = None,
           g_curve: list[tuple[float, float]] | None = None,
           b_curve: list[tuple[float, float]] | None = None) -> Image.Image:
    """Photoshop Curves. Each ``*_curve`` is a list of (input, output) control
    points in 0..255; values between control points are linearly interpolated
    (``np.interp``) to build a 256-entry LUT. ``rgb_curve`` is applied to all
    three channels; per-channel curves stack on top.

    Identity curve: ``[(0, 0), (255, 255)]``."""
    arr = _to_rgba_float(img)
    rgb = arr[..., :3]

    if rgb_curve:
        lut = _build_curve_lut(rgb_curve)
        rgb = lut[(rgb * 255 + 0.5).astype(np.int32)]
    for ch, cv in enumerate([r_curve, g_curve, b_curve]):
        if cv:
            lut = _build_curve_lut(cv)
            rgb[..., ch] = lut[(rgb[..., ch] * 255 + 0.5).astype(np.int32)]

    arr[..., :3] = rgb
    return _back_to_pil(arr)


def _build_curve_lut(points: list[tuple[float, float]]) -> np.ndarray:
    """Monotone-interpolated 256-entry LUT in 0..1 from (x,y) ∈ 0..255 points."""
    pts = sorted(points, key=lambda p: p[0])
    if pts[0][0] > 0:
        pts = [(0.0, pts[0][1])] + pts
    if pts[-1][0] < 255:
        pts = pts + [(255.0, pts[-1][1])]
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32) / 255.0
    out = np.interp(np.arange(256, dtype=np.float32), xs, ys)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------- Color Balance

def color_balance(img: Image.Image, *,
                  cyan_red: float = 0.0, magenta_green: float = 0.0,
                  yellow_blue: float = 0.0,
                  tonal_range: str = "midtones") -> Image.Image:
    """Photoshop Color Balance. Sliders are -100..+100. ``tonal_range`` is
    ``shadows``, ``midtones``, or ``highlights`` — the shift is weighted by
    a Gaussian-ish bell over luminance centered on that range."""
    arr = _to_rgba_float(img)
    rgb = arr[..., :3].copy()

    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if tonal_range == "shadows":
        # bell centered around 0.25
        weight = np.exp(-((lum - 0.25) ** 2) / (2 * 0.2 ** 2))
    elif tonal_range == "highlights":
        weight = np.exp(-((lum - 0.75) ** 2) / (2 * 0.2 ** 2))
    else:
        weight = np.exp(-((lum - 0.5) ** 2) / (2 * 0.25 ** 2))

    rgb[..., 0] += weight * (cyan_red / 100.0) * 0.5
    rgb[..., 1] += weight * (magenta_green / 100.0) * 0.5
    rgb[..., 2] += weight * (yellow_blue / 100.0) * 0.5
    arr[..., :3] = rgb
    return _back_to_pil(arr)


# ---------------------------------------------------------------- simple LUTs

def threshold(img: Image.Image, level: int = 128) -> Image.Image:
    """1-bit threshold on luminance — pixels above ``level`` go to white,
    below to black. Alpha preserved."""
    arr = _to_rgba_float(img)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mask = (lum * 255 >= level).astype(np.float32)
    arr[..., 0] = mask
    arr[..., 1] = mask
    arr[..., 2] = mask
    return _back_to_pil(arr)


def vibrance(img: Image.Image, amount: float = 0.0) -> Image.Image:
    """Boost saturation of less-saturated pixels more than already-saturated
    ones. ``amount`` ∈ -100..+100. Preserves skin tones better than a flat
    saturation boost."""
    arr = _to_rgba_float(img)
    hsv = _rgb_to_hsv_arr(arr[..., :3])
    boost = float(amount) / 100.0
    # Weight: less saturated pixels get more boost (1 - sat)
    delta = boost * (1.0 - hsv[..., 1])
    hsv[..., 1] = np.clip(hsv[..., 1] + delta, 0.0, 1.0)
    arr[..., :3] = _hsv_to_rgb_arr(hsv)
    return _back_to_pil(arr)


def channel_mixer(img: Image.Image, *,
                  r_mix: tuple[float, float, float] = (1.0, 0.0, 0.0),
                  g_mix: tuple[float, float, float] = (0.0, 1.0, 0.0),
                  b_mix: tuple[float, float, float] = (0.0, 0.0, 1.0),
                  constant: tuple[float, float, float] = (0.0, 0.0, 0.0)
                  ) -> Image.Image:
    """Photoshop Channel Mixer. Each output channel is a weighted sum of
    the three input channels plus a constant. Defaults are identity."""
    arr = _to_rgba_float(img)
    r = arr[..., 0]; g = arr[..., 1]; b = arr[..., 2]
    new_r = r_mix[0] * r + r_mix[1] * g + r_mix[2] * b + constant[0]
    new_g = g_mix[0] * r + g_mix[1] * g + g_mix[2] * b + constant[1]
    new_b = b_mix[0] * r + b_mix[1] * g + b_mix[2] * b + constant[2]
    arr[..., 0] = new_r
    arr[..., 1] = new_g
    arr[..., 2] = new_b
    return _back_to_pil(arr)


# ---------------------------------------------------------------- auto

def _opaque_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels with alpha > 0. Used by auto-tone functions
    so the histogram isn't dominated by fully-transparent regions
    (a fresh layer's transparent pixels are RGBA=(0,0,0,0) and would otherwise
    skew quantiles toward black)."""
    return arr[..., 3] > 1e-6


def auto_levels(img: Image.Image, *, clip: float = 0.01) -> Image.Image:
    """Per-channel histogram stretch. ``clip`` is the fraction of pixels to
    clip at each end (0.01 = 1%). Transparent pixels are excluded from the
    histogram so a layer with empty regions isn't dragged toward black.

    Degenerate input (all opaque pixels identical, or no opaque pixels at all)
    returns unchanged — there's no histogram to stretch."""
    arr = _to_rgba_float(img)
    opaque = _opaque_mask(arr)
    if not opaque.any():
        return _back_to_pil(arr)
    for ch in range(3):
        vals = arr[..., ch][opaque]
        lo = float(np.quantile(vals, clip))
        hi = float(np.quantile(vals, 1.0 - clip))
        span = hi - lo
        if span < 1e-6:
            continue  # degenerate channel — leave it alone
        arr[..., ch] = np.clip((arr[..., ch] - lo) / span, 0.0, 1.0)
    return _back_to_pil(arr)


def auto_contrast(img: Image.Image, *, clip: float = 0.01) -> Image.Image:
    """Like auto_levels but stretches based on overall luminance, preserving
    color balance. Transparent pixels are excluded from the histogram."""
    arr = _to_rgba_float(img)
    opaque = _opaque_mask(arr)
    if not opaque.any():
        return _back_to_pil(arr)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    lo = float(np.quantile(lum[opaque], clip))
    hi = float(np.quantile(lum[opaque], 1.0 - clip))
    span = hi - lo
    if span < 1e-6:
        return _back_to_pil(arr)
    for ch in range(3):
        arr[..., ch] = np.clip((arr[..., ch] - lo) / span, 0.0, 1.0)
    return _back_to_pil(arr)


def equalize(img: Image.Image) -> Image.Image:
    """Histogram equalize on luminance. Preserves hue/saturation roughly by
    rescaling RGB proportionally to the luminance shift. Transparent pixels
    are excluded from the histogram."""
    arr = _to_rgba_float(img)
    opaque = _opaque_mask(arr)
    if not opaque.any():
        return _back_to_pil(arr)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    # Build CDF from opaque pixels only, then apply it to the whole image.
    hist, _ = np.histogram((lum[opaque] * 255).astype(np.int32),
                           bins=256, range=(0, 256))
    cdf = hist.cumsum().astype(np.float32)
    cdf = cdf / max(cdf[-1], 1e-9)
    new_lum = cdf[(lum * 255).clip(0, 255).astype(np.int32)]
    ratio = np.where(lum > 1e-9, new_lum / np.maximum(lum, 1e-9), 1.0)
    for ch in range(3):
        arr[..., ch] = np.clip(arr[..., ch] * ratio, 0.0, 1.0)
    return _back_to_pil(arr)


def unsharp_mask(img: Image.Image, *, radius: float = 2.0,
                 amount: float = 1.5, threshold: int = 0) -> Image.Image:
    """Photoshop-style Unsharp Mask: subtract a blurred copy from the
    original to amplify edges.

    ``radius`` controls blur extent (PS calls it "radius", same units —
    pixels). ``amount`` is the strength multiplier (1.0 ≈ PS 100%; values
    around 0.5-2.0 are typical). ``threshold`` 0-255 suppresses sharpening
    where the difference is below this many gray levels — useful for
    leaving skin smooth while sharpening edges.
    """
    from PIL import ImageFilter
    has_alpha = img.mode == "RGBA"
    rgb = img.convert("RGBA" if has_alpha else "RGB")
    base = np.asarray(rgb, dtype=np.float32)
    blurred = np.asarray(
        rgb.filter(ImageFilter.GaussianBlur(radius=float(radius))),
        dtype=np.float32,
    )
    diff = base - blurred
    if threshold > 0:
        # Per-pixel gating on luminance-equivalent magnitude.
        mag = np.abs(diff[..., :3]).max(axis=-1, keepdims=True)
        diff = diff * (mag > float(threshold))
    out = base + diff * float(amount)
    out[..., :3] = np.clip(out[..., :3], 0, 255)
    if has_alpha:
        out[..., 3] = base[..., 3]
    return Image.fromarray(out.astype(np.uint8), mode="RGBA" if has_alpha else "RGB")


def high_pass(img: Image.Image, *, radius: float = 4.0) -> Image.Image:
    """High-pass filter: image - blur(image), shifted to mid-gray. Common
    in frequency-separation retouching workflows — use ``soft_light`` or
    ``linear_light`` blend mode of the result over the original for an
    edge-only sharpening pass.

    ``radius`` controls the blur extent that defines the "low frequency"
    being removed."""
    from PIL import ImageFilter
    rgb = img.convert("RGB")
    base = np.asarray(rgb, dtype=np.float32)
    blurred = np.asarray(
        rgb.filter(ImageFilter.GaussianBlur(radius=float(radius))),
        dtype=np.float32,
    )
    out = (base - blurred) + 128.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def add_vignette(img: Image.Image, *, strength: float = 0.6,
                 radius: float = 1.0, falloff: float = 2.0,
                 color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Classic photo vignette — darken (or tint) the corners radially.

    - ``strength`` 0-1: intensity at the corners (0 = no change, 1 = full
      colour at edges).
    - ``radius`` 0-2: where the unaffected region ends, in units of the
      shorter half-axis. 1.0 means the vignette starts halfway to the
      edge; higher = smaller affected area.
    - ``falloff`` >= 1: steepness of the gradient between unaffected and
      max-affected zones (2 = smooth quadratic, 4 = sharp).
    - ``color``: RGB of the corner tint. Default (0, 0, 0) = darken.
    """
    has_alpha = img.mode == "RGBA"
    base = img.convert("RGBA" if has_alpha else "RGB")
    arr = np.asarray(base, dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    half = min(w, h) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(half, 1.0)
    t = np.clip((r - float(radius)) / max(2.0 - float(radius), 1e-3), 0.0, 1.0)
    weight = (t ** float(falloff)) * float(strength)
    weight = weight[..., None]
    tint = np.array(list(color) + ([255] if has_alpha else []),
                    dtype=np.float32)[None, None, :]
    out = arr * (1.0 - weight) + tint * weight
    if has_alpha:
        out[..., 3] = arr[..., 3]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8),
                           mode="RGBA" if has_alpha else "RGB")


def add_noise(img: Image.Image, *, amount: float = 0.1,
              kind: str = "gaussian", seed: int | None = None) -> Image.Image:
    """Add noise — film-grain effect or pre-process step.

    - ``kind='gaussian'``: zero-mean noise with std = ``amount * 255``.
      ``amount`` typical 0.02-0.2.
    - ``kind='uniform'``: ±(``amount * 255``).
    - ``kind='salt_pepper'``: random pixels flipped to pure black/white.
      ``amount`` is the proportion of pixels affected (e.g. 0.02 = 2 %).
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        n = rng.normal(0.0, max(amount, 0.0) * 255.0, size=arr[..., :3].shape)
        arr[..., :3] = np.clip(arr[..., :3] + n, 0, 255)
    elif kind == "uniform":
        n = (rng.random(arr[..., :3].shape) - 0.5) * 2.0 * max(amount, 0.0) * 255.0
        arr[..., :3] = np.clip(arr[..., :3] + n, 0, 255)
    elif kind == "salt_pepper":
        prob = max(0.0, min(1.0, float(amount)))
        flat = rng.random(arr.shape[:2])
        salt = flat < prob / 2.0
        pepper = flat > 1.0 - prob / 2.0
        arr[salt, :3] = 255.0
        arr[pepper, :3] = 0.0
    else:
        raise ValueError(
            f"unknown noise kind {kind!r}. Choose from "
            "gaussian, uniform, salt_pepper"
        )
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def bilateral_filter(img: Image.Image, *, diameter: int = 9,
                     sigma_color: float = 75.0,
                     sigma_space: float = 75.0) -> Image.Image:
    """Edge-preserving smoothing — denoise without blurring edges.
    Implemented via OpenCV's ``cv2.bilateralFilter``.

    - ``diameter`` is the neighbourhood pixel diameter (5-15 typical).
    - ``sigma_color`` controls how dissimilar pixels stop contributing
      (higher = more aggressive smoothing across colour edges).
    - ``sigma_space`` is the Gaussian falloff in coordinate space.
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "bilateral_filter needs opencv-python. "
            "`pip install opencv-python` (or install [seg]/[face]/[yolo]/[sd])."
        ) from e
    rgb = np.asarray(img.convert("RGB"))
    out = cv2.bilateralFilter(
        rgb, int(diameter), float(sigma_color), float(sigma_space),
    )
    return Image.fromarray(out, mode="RGB")
