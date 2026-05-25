"""PSD read + write via psd-tools.

PSD files preserve a layer stack with names, opacity, visibility, blend modes,
masks, and per-layer offsets — the things our :class:`Layer` model carries.

Loading walks the PSD's top-level layer list bottom-to-top, converting each
to a :class:`Layer` via ``topil()``. Group layers (folders) are recursively
flattened to a single composite layer for now; round-tripping groups would
require a separate ``LayerGroup`` model.

Saving creates a fresh PSDImage of the canvas size and adds one pixel layer
per Layer in stack order (bottom to top — psd-tools handles z-ordering).
Masks aren't yet written through because psd-tools' write API doesn't expose
a clean "add mask to layer" call as of 1.17. Masks are baked into the
layer's alpha on save with a one-line note in the response.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .layers import Layer


# Mapping our blend_mode strings to the psd-tools BlendMode enum value.
# We use the enum's bytestring at write time so the import here is light.
_BLEND_TO_PSD = {
    "normal": "norm",
    "multiply": "mul ",
    "screen": "scrn",
    "overlay": "over",
    "darken": "dark",
    "lighten": "lite",
    "color_dodge": "div ",
    "color_burn": "idiv",
    "hard_light": "hLit",
    "soft_light": "sLit",
    "difference": "diff",
    "exclusion": "smud",
    # "add" in our model maps to Photoshop's "Linear Dodge (Add)".
    "add": "lddg",
    "subtract": "fsub",
}
_PSD_TO_BLEND = {v: k for k, v in _BLEND_TO_PSD.items()}


def _bake_mask(layer: Layer) -> Image.Image:
    """Apply a Layer's mask to its image alpha and return a fresh RGBA image."""
    if layer.mask is None:
        return layer.image
    rgba = np.asarray(layer.image.convert("RGBA"), dtype=np.float32)
    mask = np.asarray(layer.mask.convert("L"), dtype=np.float32) / 255.0
    rgba[..., 3] = rgba[..., 3] * mask
    return Image.fromarray(rgba.astype("uint8"), mode="RGBA")


def load_psd(path: str) -> tuple[int, int, list[Layer]]:
    """Load a PSD and return ``(width, height, layers_bottom_to_top)``.

    Group layers are composited and added as a single flat layer named after
    the group. Mask data is preserved on each layer when present."""
    from psd_tools import PSDImage  # local import → no overhead when unused

    psd = PSDImage.open(path)
    layers: list[Layer] = []

    # psd-tools iterates layers bottom-to-top, matching our convention.
    for psd_layer in psd:
        layers.append(_psd_layer_to_layer(psd_layer))

    if not layers:
        # Fallback: PSD with no top-level layers (rare). Use the composite.
        composite = psd.composite() or Image.new("RGBA", psd.size, (0, 0, 0, 0))
        layers = [Layer(name="Background", image=composite.convert("RGBA"))]

    return psd.width, psd.height, layers


def _psd_layer_to_layer(psd_layer: Any) -> Layer:
    """Convert one psd-tools layer (pixel layer or group) to our Layer."""
    if psd_layer.is_group():
        # Composite the group's children into a single image. Loses internal
        # structure but preserves visual fidelity.
        composite = psd_layer.composite() or Image.new(
            "RGBA", psd_layer.size or (1, 1), (0, 0, 0, 0)
        )
        img = composite.convert("RGBA")
        offset = (psd_layer.left or 0, psd_layer.top or 0)
    else:
        pil = psd_layer.topil()
        if pil is None:
            # Empty layer — synthesize a 1x1 transparent stand-in so downstream
            # ops don't crash on size (0, 0).
            pil = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        img = pil.convert("RGBA")
        offset = (psd_layer.left or 0, psd_layer.top or 0)

    blend_key = (
        psd_layer.blend_mode.value.decode("ascii")
        if hasattr(psd_layer.blend_mode, "value")
        else "norm"
    )
    blend_mode = _PSD_TO_BLEND.get(blend_key, "normal")

    # PSD mask is on .mask attribute; .topil() returns a grayscale image or None.
    mask_img = None
    try:
        if psd_layer.mask is not None and not psd_layer.mask.disabled:
            mask_pil = psd_layer.mask.topil()
            if mask_pil is not None:
                # PSD masks can be smaller than the layer (mask bounds), but
                # our model expects same-size. Pad to layer size.
                if mask_pil.size != img.size:
                    full = Image.new("L", img.size, 255)
                    mleft = (psd_layer.mask.left or 0) - offset[0]
                    mtop = (psd_layer.mask.top or 0) - offset[1]
                    full.paste(mask_pil, (mleft, mtop))
                    mask_img = full
                else:
                    mask_img = mask_pil.convert("L")
    except Exception:
        mask_img = None

    return Layer(
        name=psd_layer.name or "Layer",
        image=img,
        offset=offset,
        opacity=float(psd_layer.opacity) / 255.0,
        visible=bool(psd_layer.visible),
        blend_mode=blend_mode,
        mask=mask_img,
    )


def save_psd(layers: list[Layer], canvas_size: tuple[int, int],
             path: str) -> dict[str, Any]:
    """Write a layered PSD. Masks are baked into layer alpha (psd-tools 1.17
    doesn't expose mask write); everything else round-trips."""
    from psd_tools import PSDImage
    from psd_tools.constants import BlendMode

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    psd = PSDImage.new(mode="RGBA", size=canvas_size)
    masks_baked = 0
    for layer in layers:
        img = _bake_mask(layer)
        if layer.mask is not None:
            masks_baked += 1
        bm_key = _BLEND_TO_PSD.get(layer.blend_mode, "norm")
        bm = BlendMode(bm_key.encode("ascii"))
        psd.create_pixel_layer(
            image=img,
            name=layer.name,
            left=int(layer.offset[0]),
            top=int(layer.offset[1]),
            opacity=max(0, min(255, int(round(layer.opacity * 255)))),
            blend_mode=bm,
        )
        # Visibility isn't a create_pixel_layer kwarg; set it on the layer.
        psd[-1].visible = bool(layer.visible)

    psd.save(path)
    return {
        "path": str(Path(path).resolve()),
        "format": "PSD",
        "layers": len(layers),
        "masks_baked_to_alpha": masks_baked,
        "size_bytes": Path(path).stat().st_size,
    }
