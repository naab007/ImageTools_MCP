"""Palette extraction + colour quantisation.

- ``extract_palette(image, n_colors, method)`` — find the N dominant
  colours via k-means / median-cut / mode-by-bin. Returns list of dicts
  with ``hex``, ``rgb``, and ``ratio`` (fraction of pixels in that bin).
- ``quantize_to_palette(image, n_colors, method, dither)`` — reduce the
  image to N colours, returning a new image (palette mode is upgradeable
  to RGB on save). Different from ``posterize`` which clips bit-depth
  per channel and produces banding.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _flat_rgb(image: Image.Image, max_samples: int = 50_000) -> np.ndarray:
    """RGB pixel array (N, 3) uint8. Sub-samples for speed when the image
    has more than ``max_samples`` pixels — the result is still
    representative for clustering."""
    rgb = np.asarray(image.convert("RGB"))
    flat = rgb.reshape(-1, 3)
    if flat.shape[0] > max_samples:
        idx = np.random.default_rng(42).choice(flat.shape[0],
                                                max_samples, replace=False)
        flat = flat[idx]
    return flat


def _kmeans(samples: np.ndarray, k: int, max_iter: int = 30) -> np.ndarray:
    """Tiny in-pure-numpy k-means. Returns ``k`` centroids as float32 (k, 3).
    Good enough for palette extraction — no sklearn dependency."""
    rng = np.random.default_rng(42)
    # k-means++ init for stability on tricky distributions.
    centroids = np.empty((k, 3), dtype=np.float32)
    centroids[0] = samples[rng.integers(0, samples.shape[0])]
    for i in range(1, k):
        d = np.min(
            np.sum((samples[:, None, :].astype(np.float32) - centroids[:i])
                   ** 2, axis=-1), axis=1,
        )
        probs = d / max(d.sum(), 1e-9)
        idx = rng.choice(samples.shape[0], p=probs)
        centroids[i] = samples[idx]
    s = samples.astype(np.float32)
    for _ in range(max_iter):
        # Assign
        d2 = np.sum((s[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        labels = d2.argmin(axis=1)
        # Update — guard empty clusters by keeping the prior centroid.
        new = centroids.copy()
        for j in range(k):
            mask = labels == j
            if mask.any():
                new[j] = s[mask].mean(axis=0)
        if np.allclose(new, centroids, atol=0.5):
            break
        centroids = new
    return centroids


def extract_palette(image: Image.Image, *,
                    n_colors: int = 8,
                    method: str = "kmeans") -> list[dict]:
    """Return ``n_colors`` dominant colours as a list of
    ``{"hex": "#rrggbb", "rgb": [r, g, b], "ratio": 0-1}`` entries, sorted
    by frequency (most common first).

    ``method``:
      - ``kmeans`` — pure-NumPy k-means (default; perceptually decent).
      - ``median_cut`` — PIL's built-in quantiser (very fast, OK quality).
      - ``mode`` — bin pixels into 6³ = 216 cells, pick the top ``n_colors``
        by occupancy. Crude but extremely fast.
    """
    n = max(2, min(64, int(n_colors)))
    samples = _flat_rgb(image)
    if method == "median_cut":
        # Pillow's quantizer is median-cut by default; force RGB output.
        q = image.convert("RGB").quantize(colors=n, method=Image.Quantize.MEDIANCUT)
        pal = q.getpalette()[:n * 3]
        counts = np.bincount(np.asarray(q).ravel(), minlength=n)[:n]
        order = np.argsort(-counts)
        out = []
        total = max(counts.sum(), 1)
        for i in order:
            r, g, b = pal[i * 3], pal[i * 3 + 1], pal[i * 3 + 2]
            out.append({
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "rgb": [int(r), int(g), int(b)],
                "ratio": round(float(counts[i] / total), 4),
            })
        return out
    if method == "mode":
        # Bin into 6³ rgb cells, pick top-N by count.
        binned = (samples // 43).astype(np.int32)  # 6 bins per channel
        keys = binned[:, 0] * 36 + binned[:, 1] * 6 + binned[:, 2]
        uniq, counts = np.unique(keys, return_counts=True)
        order = np.argsort(-counts)[:n]
        total = max(counts.sum(), 1)
        out = []
        for idx in order:
            k = uniq[idx]
            b = (k % 6) * 43 + 21
            g = ((k // 6) % 6) * 43 + 21
            r = (k // 36) * 43 + 21
            out.append({
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "rgb": [int(r), int(g), int(b)],
                "ratio": round(float(counts[idx] / total), 4),
            })
        return out
    # default — kmeans
    centroids = _kmeans(samples, n)
    # Compute final assignments + counts.
    d2 = np.sum(
        (samples[:, None, :].astype(np.float32) - centroids[None, :, :])
        ** 2, axis=-1,
    )
    labels = d2.argmin(axis=1)
    counts = np.bincount(labels, minlength=n)
    order = np.argsort(-counts)
    total = max(counts.sum(), 1)
    out = []
    for i in order:
        r, g, b = (int(round(c)) for c in centroids[i])
        out.append({
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "rgb": [int(r), int(g), int(b)],
            "ratio": round(float(counts[i] / total), 4),
        })
    return out


def quantize_to_palette(image: Image.Image, *,
                       n_colors: int = 16,
                       method: str = "median_cut",
                       dither: bool = True) -> Image.Image:
    """Reduce ``image`` to ``n_colors`` total colours. Returns an RGB image
    (the palette-mode intermediate is converted back so downstream tools
    can keep working).

    ``method`` ∈ ``median_cut`` (default; Pillow's classic), ``maxcoverage``
    (Pillow's alternative — better with skewed colour distributions),
    ``fastoctree``. ``dither=True`` applies Floyd-Steinberg dithering for
    smoother gradients."""
    method_map = {
        "median_cut": Image.Quantize.MEDIANCUT,
        "maxcoverage": Image.Quantize.MAXCOVERAGE,
        "fastoctree": Image.Quantize.FASTOCTREE,
    }
    q = method_map.get(method.lower())
    if q is None:
        raise ValueError(
            f"unknown method {method!r}. Choose from {list(method_map)}"
        )
    src = image.convert("RGB")
    dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    pal = src.quantize(colors=max(2, min(256, int(n_colors))),
                       method=q, dither=dither_mode)
    return pal.convert("RGB")
