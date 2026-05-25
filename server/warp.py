"""Photoshop-style warping / distortion tools.

Five primitives:
1. ``warp_perspective`` — 4-corner quad warp (PS Edit > Transform > Distort
   / Perspective Warp).
2. ``warp_mesh`` — NxN bezier-grid warp via thin-plate spline (PS Edit >
   Transform > Warp).
3. ``liquify`` — local brush-based deformation (push / twirl / pucker /
   bloat). PS Filter > Liquify.
4. ``distort`` — whole-image procedural filters: ``spherize``, ``pinch``,
   ``twirl``, ``wave``, ``ripple``, ``polar_to_rect``, ``rect_to_polar``.
   PS Filter > Distort.
5. ``displace_by_map`` — use a grayscale image as a displacement field
   (R/luminance → x shift, G → y shift). PS Filter > Distort > Displace.

All operations preserve the source canvas size; out-of-bounds samples are
filled transparent (RGBA) or with a user-supplied colour.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def _check_cv2():
    try:
        import cv2  # noqa: F401
        return cv2
    except ImportError as e:
        raise RuntimeError(
            "Warping needs OpenCV. `pip install opencv-python` "
            "(or install the [seg] / [face] / [yolo] / [sd] extra)."
        ) from e


def _parse_quad(points: list, label: str) -> np.ndarray:
    """Validate a 4-point list and return a float32 (4, 2) array.
    Accepts ``[[x, y], ...]`` and ``[(x, y), ...]``."""
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise ValueError(f"{label} must be a list of 4 [x, y] points")
    return np.asarray([[float(p[0]), float(p[1])] for p in points],
                      dtype=np.float32)


def _to_rgba_np(image: Image.Image) -> np.ndarray:
    """PIL → numpy in RGBA uint8 with predictable channel order."""
    return np.asarray(image.convert("RGBA"))


def _from_rgba_np(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA")


# ---- 1. perspective warp --------------------------------------------------

def warp_perspective(image: Image.Image,
                     src_corners: list, dst_corners: list, *,
                     output_size: tuple[int, int] | None = None,
                     ) -> Image.Image:
    """4-corner quad warp. ``src_corners`` and ``dst_corners`` are each a
    list of four ``[x, y]`` points in **the same order** — typically
    top-left, top-right, bottom-right, bottom-left.

    Output is ``output_size`` (defaults to source size). Out-of-bounds
    samples are transparent.

    Useful for: straightening a photographed document, correcting a
    perspective-skewed sign, mapping artwork onto a flat surface in another
    image, etc.
    """
    cv2 = _check_cv2()
    src = _parse_quad(src_corners, "src_corners")
    dst = _parse_quad(dst_corners, "dst_corners")
    H = cv2.getPerspectiveTransform(src, dst)
    arr = _to_rgba_np(image)
    h, w = arr.shape[:2]
    out_size = output_size or (w, h)
    warped = cv2.warpPerspective(
        arr, H, (int(out_size[0]), int(out_size[1])),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return _from_rgba_np(warped)


# ---- 2. mesh warp (thin-plate spline) -------------------------------------

def _tps_kernel(r2: np.ndarray) -> np.ndarray:
    """Thin-plate-spline radial kernel: U(r) = r² log(r). ``r2`` is squared
    distance. Returns 0 where r=0 (otherwise NaN from log(0))."""
    out = np.zeros_like(r2)
    mask = r2 > 1e-12
    # 0.5 * r² log(r²) is the same value (and avoids a sqrt).
    out[mask] = 0.5 * r2[mask] * np.log(r2[mask])
    return out


def _tps_fit(src: np.ndarray, dst: np.ndarray,
             lam: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 2-D thin-plate spline mapping ``src`` (N, 2) points to ``dst``
    (N, 2) targets. Returns ``(W, A)`` where ``W`` is (N, 2) (kernel weights)
    and ``A`` is (3, 2) (affine part). ``lam`` adds Tikhonov regularisation
    to the kernel diagonal — > 0 relaxes the exact-interpolation constraint
    for a smoother fit.
    """
    n = src.shape[0]
    diff = src[:, None, :] - src[None, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    K = _tps_kernel(r2)
    if lam > 0:
        K = K + np.eye(n) * float(lam)
    P = np.concatenate([np.ones((n, 1), dtype=np.float64), src], axis=1)  # (N, 3)
    # Build the (N+3) × (N+3) block system:
    #   [K  P]
    #   [P' 0]
    L = np.zeros((n + 3, n + 3), dtype=np.float64)
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T
    rhs = np.concatenate([dst, np.zeros((3, 2), dtype=np.float64)], axis=0)
    # Solve. Use lstsq for numerical robustness — solve() can NaN on near-
    # singular cases (e.g. colinear control points).
    sol, *_ = np.linalg.lstsq(L, rhs, rcond=None)
    return sol[:n], sol[n:]  # (N, 2), (3, 2)


def _tps_apply(W: np.ndarray, A: np.ndarray,
               src: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Evaluate the fitted TPS at each query point. Returns (M, 2) mapped
    coordinates."""
    # Pairwise squared distances between query (M, 2) and src (N, 2).
    diff = query[:, None, :] - src[None, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    K = _tps_kernel(r2)
    Pq = np.concatenate([np.ones((query.shape[0], 1)), query], axis=1)
    return K @ W + Pq @ A


def warp_mesh(image: Image.Image,
              src_grid: list, dst_grid: list, *,
              regularization: float = 0.0,
              ) -> Image.Image:
    """Warp ``image`` so the points listed in ``src_grid`` move to the
    matching points in ``dst_grid``. Both lists must be the same length
    (>= 3, more = finer control; 3x3 = 9 or 5x5 = 25 mirrors Photoshop's
    default).

    Implemented via a pure-NumPy thin-plate spline — smooth deformation
    that interpolates exactly through the control points (unless
    ``regularization`` > 0 is set, which trades fidelity for smoothness).

    Each grid entry is ``[x, y]`` in pixel coordinates. Points outside the
    image are accepted (they constrain the deformation near the edges).
    """
    cv2 = _check_cv2()
    if len(src_grid) != len(dst_grid) or len(src_grid) < 3:
        raise ValueError(
            f"src_grid and dst_grid must be the same length and >= 3; got "
            f"{len(src_grid)} and {len(dst_grid)}"
        )
    arr = _to_rgba_np(image)
    h, w = arr.shape[:2]
    src = np.asarray(
        [[float(p[0]), float(p[1])] for p in src_grid], dtype=np.float64)
    dst = np.asarray(
        [[float(p[0]), float(p[1])] for p in dst_grid], dtype=np.float64)

    # For an inverse-warp remap we need a function f such that, for every
    # OUTPUT pixel (which is a destination location q), we know where in
    # the source to sample. So we fit TPS dst -> src.
    W, A = _tps_fit(dst, src, lam=float(regularization))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    query = np.column_stack([xx.ravel(), yy.ravel()])  # (H*W, 2) in (x, y)
    mapped = _tps_apply(W, A, dst, query)  # (H*W, 2) in (x, y)
    map_x = mapped[:, 0].reshape(h, w).astype(np.float32)
    map_y = mapped[:, 1].reshape(h, w).astype(np.float32)

    warped = cv2.remap(
        arr, map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return _from_rgba_np(warped)


# ---- 3. liquify brushes ---------------------------------------------------

_LIQUIFY_MODES = ("push", "twirl_cw", "twirl_ccw", "pucker", "bloat")


def _stamp_displacement(
    dx_field: np.ndarray, dy_field: np.ndarray,
    cx: float, cy: float, radius: float, strength: float,
    mode: str, push_vec: tuple[float, float] | None = None,
) -> None:
    """Accumulate one brush stamp into the per-pixel displacement field
    ``(dx_field, dy_field)``. Mutates the arrays in place.

    Strength is in [-1.5, 1.5] — Photoshop allows over-100% which produces
    cartoony deformation.
    """
    h, w = dx_field.shape
    r = int(np.ceil(radius))
    x0 = max(0, int(cx) - r)
    y0 = max(0, int(cy) - r)
    x1 = min(w, int(cx) + r + 1)
    y1 = min(h, int(cy) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    rx = xx - cx
    ry = yy - cy
    dist = np.sqrt(rx * rx + ry * ry)
    inside = dist < radius
    # Falloff: smoothstep from centre (1.0) to edge (0.0). Squared distance
    # in normalised space — Photoshop uses something similar.
    t = np.clip(1.0 - dist / max(radius, 1e-3), 0.0, 1.0)
    falloff = t * t * (3.0 - 2.0 * t)
    falloff = falloff * inside  # zero outside brush

    if mode == "push":
        if push_vec is None:
            return
        ux, uy = push_vec
        s = float(strength) * falloff
        dx_field[y0:y1, x0:x1] += ux * s
        dy_field[y0:y1, x0:x1] += uy * s
    elif mode in ("twirl_cw", "twirl_ccw"):
        sign = 1.0 if mode == "twirl_cw" else -1.0
        angle = sign * float(strength) * falloff * np.pi
        cos_a = np.cos(angle); sin_a = np.sin(angle)
        new_rx = cos_a * rx - sin_a * ry
        new_ry = sin_a * rx + cos_a * ry
        # Displacement = (new_position - current_position) inside the brush.
        dx_field[y0:y1, x0:x1] += (new_rx - rx)
        dy_field[y0:y1, x0:x1] += (new_ry - ry)
    elif mode == "pucker":
        # Pull each pixel toward the centre.
        s = float(strength) * falloff
        dx_field[y0:y1, x0:x1] += -rx * s
        dy_field[y0:y1, x0:x1] += -ry * s
    elif mode == "bloat":
        s = float(strength) * falloff
        dx_field[y0:y1, x0:x1] += rx * s
        dy_field[y0:y1, x0:x1] += ry * s
    else:
        raise ValueError(
            f"unknown liquify mode {mode!r}. Supported: {_LIQUIFY_MODES}"
        )


def liquify(image: Image.Image, mode: str,
            points: list, *,
            radius: float = 50.0, strength: float = 0.5,
            push_vector: tuple[float, float] | None = None,
            ) -> Image.Image:
    """Local brush deformation (PS Filter > Liquify).

    Modes:
      - ``push`` — drag pixels along ``push_vector`` ``(dx, dy)``. Use this
        when the caller knows the stroke direction (e.g. cursor delta).
      - ``twirl_cw`` / ``twirl_ccw`` — rotate pixels inside the brush around
        each point.
      - ``pucker`` — pull pixels toward each point (shrink feature).
      - ``bloat`` — push pixels away from each point (enlarge feature).

    ``points`` is a list of ``[x, y]`` brush-centre positions.
    ``radius`` is the brush radius in pixels (Photoshop calls this "brush
    size" and gives a diameter; halve it if you're translating).
    ``strength`` 0-1 nominal; >1 over-pushes.
    """
    cv2 = _check_cv2()
    if mode not in _LIQUIFY_MODES:
        raise ValueError(
            f"unknown liquify mode {mode!r}. Supported: {_LIQUIFY_MODES}"
        )
    arr = _to_rgba_np(image)
    h, w = arr.shape[:2]
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    for p in points:
        _stamp_displacement(
            dx, dy, float(p[0]), float(p[1]),
            radius, strength, mode, push_vec=push_vector,
        )
    map_x = (np.arange(w, dtype=np.float32)[None, :] + dx).astype(np.float32)
    map_y = (np.arange(h, dtype=np.float32)[:, None] + dy).astype(np.float32)
    warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return _from_rgba_np(warped)


# ---- 4. distort filters ---------------------------------------------------

_DISTORT_MODES = (
    "spherize", "pinch", "twirl", "wave", "ripple",
    "polar_to_rect", "rect_to_polar",
)


def distort(image: Image.Image, mode: str, **params: Any) -> Image.Image:
    """Procedural distortion filters (PS Filter > Distort family).

    Modes + params:
      - ``spherize(amount: -1.0..1.0)`` — bulge (positive) or pinch
        (negative) like a fisheye lens applied to the centre. ``center_x``
        and ``center_y`` (optional) pick the centre; defaults to the image
        centre.
      - ``pinch(amount: -1.0..1.0)`` — same as spherize with inverted sign
        for compatibility with PS naming.
      - ``twirl(angle_deg: float)`` — rotate pixels by an angle proportional
        to distance from centre. Positive = clockwise. ``center_x``,
        ``center_y``, ``radius`` (default = half the shorter edge) tune
        the effect.
      - ``wave(amplitude_x: float = 0, amplitude_y: float = 0, period_x:
        float = 100, period_y: float = 100)`` — sinusoidal displacement.
      - ``ripple(amplitude: float = 10, period: float = 50, center_x,
        center_y)`` — radial sinusoidal displacement (pond ripple).
      - ``polar_to_rect()`` / ``rect_to_polar()`` — polar↔rectangular
        coordinate remap. Optional ``center_x`` / ``center_y``.

    All filters preserve image size; pixels remapped from outside the
    source replicate the edge.
    """
    cv2 = _check_cv2()
    if mode not in _DISTORT_MODES:
        raise ValueError(
            f"unknown distort mode {mode!r}. Supported: {_DISTORT_MODES}"
        )
    arr = _to_rgba_np(image)
    h, w = arr.shape[:2]
    cx_default = (w - 1) / 2.0
    cy_default = (h - 1) / 2.0
    cx = float(params.get("center_x", cx_default))
    cy = float(params.get("center_y", cy_default))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    if mode in ("spherize", "pinch"):
        amount = float(params.get("amount", 0.5))
        if mode == "pinch":
            amount = -amount
        r_max = min(w, h) / 2.0
        rx = xx - cx
        ry = yy - cy
        r = np.sqrt(rx * rx + ry * ry)
        r_norm = np.clip(r / r_max, 0, 1)
        # Smoothly scale displacement near the edge.
        # New r = r * (1 + amount * sin(pi * r_norm)) keeps centre and edge
        # fixed while distorting the middle — matches PS spherize feel.
        scale = 1.0 + amount * np.sin(np.pi * r_norm)
        new_r = r * scale
        # Avoid divide-by-zero at the centre.
        unit = np.where(r > 1e-3, 1.0 / r, 0.0)
        map_x = (cx + rx * new_r * unit).astype(np.float32)
        map_y = (cy + ry * new_r * unit).astype(np.float32)
    elif mode == "twirl":
        angle_deg = float(params.get("angle_deg", 90))
        radius = float(params.get("radius", min(w, h) / 2.0))
        rx = xx - cx
        ry = yy - cy
        r = np.sqrt(rx * rx + ry * ry)
        # Angle falls linearly from full at centre to 0 at radius.
        angle = np.radians(angle_deg) * np.clip(1.0 - r / max(radius, 1.0), 0, 1)
        cos_a = np.cos(angle); sin_a = np.sin(angle)
        new_rx = cos_a * rx + sin_a * ry
        new_ry = -sin_a * rx + cos_a * ry
        map_x = (cx + new_rx).astype(np.float32)
        map_y = (cy + new_ry).astype(np.float32)
    elif mode == "wave":
        ax = float(params.get("amplitude_x", 0))
        ay = float(params.get("amplitude_y", 10))
        px = float(params.get("period_x", 100))
        py = float(params.get("period_y", 100))
        if px <= 0 or py <= 0:
            raise ValueError("wave periods must be > 0")
        map_x = (xx + ax * np.sin(2 * np.pi * yy / px)).astype(np.float32)
        map_y = (yy + ay * np.sin(2 * np.pi * xx / py)).astype(np.float32)
    elif mode == "ripple":
        amp = float(params.get("amplitude", 10))
        period = float(params.get("period", 50))
        rx = xx - cx
        ry = yy - cy
        r = np.sqrt(rx * rx + ry * ry)
        # Push along radial direction by sin(2π r / period) * amplitude.
        unit_x = np.where(r > 1e-3, rx / r, 0.0)
        unit_y = np.where(r > 1e-3, ry / r, 0.0)
        wave = amp * np.sin(2 * np.pi * r / max(period, 1e-3))
        map_x = (xx + unit_x * wave).astype(np.float32)
        map_y = (yy + unit_y * wave).astype(np.float32)
    elif mode == "rect_to_polar":
        # For each output pixel (x, y), interpret (x, y) as (theta, radius)
        # and sample the corresponding cartesian source pixel.
        theta = (xx / max(w - 1, 1)) * 2.0 * np.pi
        r_max = min(w, h) / 2.0
        radius = (yy / max(h - 1, 1)) * r_max
        map_x = (cx + radius * np.cos(theta)).astype(np.float32)
        map_y = (cy + radius * np.sin(theta)).astype(np.float32)
    elif mode == "polar_to_rect":
        # For each output pixel, treat it as (theta, radius) relative to
        # (cx, cy) and convert.
        rx = xx - cx
        ry = yy - cy
        r = np.sqrt(rx * rx + ry * ry)
        theta = np.arctan2(ry, rx) % (2 * np.pi)
        r_max = min(w, h) / 2.0
        map_x = ((theta / (2.0 * np.pi)) * (w - 1)).astype(np.float32)
        map_y = ((r / max(r_max, 1.0)) * (h - 1)).astype(np.float32)
    else:
        raise ValueError(f"unhandled mode {mode!r}")  # unreachable

    warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return _from_rgba_np(warped)


# ---- 5. displacement map --------------------------------------------------

def displace_by_map(image: Image.Image,
                    displacement_image: Image.Image, *,
                    scale_x: float = 10.0,
                    scale_y: float | None = None,
                    channel_mode: str = "luminance",
                    ) -> Image.Image:
    """Displace pixels of ``image`` according to a displacement map (PS
    Filter > Distort > Displace).

    ``channel_mode``:
      - ``luminance`` (default): single grayscale channel → both x and y
        displacement (so any greyscale map works). Values ≷ 128 push
        positive/negative.
      - ``rg``: red channel → x, green channel → y. Lets you make custom
        maps with per-axis control.

    ``scale_x`` / ``scale_y`` (``scale_y`` defaults to ``scale_x``) scale
    the displacement in pixels. PS default is 10 pixels.

    The displacement map is resized to the source image size if needed.
    """
    cv2 = _check_cv2()
    arr = _to_rgba_np(image)
    h, w = arr.shape[:2]

    disp = displacement_image
    if disp.size != (w, h):
        disp = disp.resize((w, h), Image.BILINEAR)

    if scale_y is None:
        scale_y = float(scale_x)

    if channel_mode == "rg":
        dmap = np.asarray(disp.convert("RGB"), dtype=np.float32)
        # 128 = no displacement; map to [-1, 1] then scale.
        dx = (dmap[..., 0] - 128.0) / 127.0 * float(scale_x)
        dy = (dmap[..., 1] - 128.0) / 127.0 * float(scale_y)
    elif channel_mode == "luminance":
        dmap = np.asarray(disp.convert("L"), dtype=np.float32)
        d = (dmap - 128.0) / 127.0
        dx = d * float(scale_x)
        dy = d * float(scale_y)
    else:
        raise ValueError(
            f"unknown channel_mode {channel_mode!r}. Supported: rg, luminance"
        )

    map_x = (np.arange(w, dtype=np.float32)[None, :] + dx).astype(np.float32)
    map_y = (np.arange(h, dtype=np.float32)[:, None] + dy).astype(np.float32)
    warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return _from_rgba_np(warped)
