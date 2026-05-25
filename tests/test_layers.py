"""Layer compositing: blend modes, opacity, offset, masks."""
import numpy as np
import pytest
from PIL import Image

from server.layers import (
    BLEND_MODES, Layer, compose_layers, invert_mask, make_white_mask,
)


def _solid(size, color):
    return Image.new("RGBA", size, color)


def test_normal_blend_top_opaque_covers_bottom():
    bg = Layer(name="bg", image=_solid((4, 4), (255, 0, 0, 255)))
    top = Layer(name="top", image=_solid((4, 4), (0, 255, 0, 255)))
    out = compose_layers([bg, top], (4, 4))
    assert out.getpixel((0, 0))[:3] == (0, 255, 0)


def test_invisible_layer_skipped():
    bg = Layer(name="bg", image=_solid((4, 4), (255, 0, 0, 255)))
    top = Layer(name="top", image=_solid((4, 4), (0, 255, 0, 255)),
                visible=False)
    out = compose_layers([bg, top], (4, 4))
    assert out.getpixel((0, 0))[:3] == (255, 0, 0)


def test_opacity_blends_proportionally():
    bg = Layer(name="bg", image=_solid((4, 4), (0, 0, 0, 255)))
    top = Layer(name="top", image=_solid((4, 4), (255, 255, 255, 255)),
                opacity=0.5)
    out = compose_layers([bg, top], (4, 4))
    r, g, b, _ = out.getpixel((0, 0))
    # halfway between black and white
    assert 120 < r < 135 and 120 < g < 135 and 120 < b < 135


def test_multiply_darkens():
    bg = Layer(name="bg", image=_solid((4, 4), (255, 128, 64, 255)))
    top = Layer(name="top", image=_solid((4, 4), (128, 128, 128, 255)),
                blend_mode="multiply")
    out = compose_layers([bg, top], (4, 4))
    r, g, b, _ = out.getpixel((0, 0))
    # multiply: 255*128/255=128, 128*128/255=64, 64*128/255=32 (±1 rounding)
    assert abs(r - 128) <= 2 and abs(g - 64) <= 2 and abs(b - 32) <= 2


def test_screen_lightens():
    bg = Layer(name="bg", image=_solid((4, 4), (100, 100, 100, 255)))
    top = Layer(name="top", image=_solid((4, 4), (100, 100, 100, 255)),
                blend_mode="screen")
    out = compose_layers([bg, top], (4, 4))
    # screen(100, 100) = 255 - (155*155)/255 ≈ 161
    r = out.getpixel((0, 0))[0]
    assert r > 100  # lightened


def test_difference_zeros_self():
    """A layer differenced with itself produces black."""
    same = Layer(name="bg", image=_solid((4, 4), (200, 150, 100, 255)))
    diff = Layer(name="top", image=_solid((4, 4), (200, 150, 100, 255)),
                 blend_mode="difference")
    out = compose_layers([same, diff], (4, 4))
    assert out.getpixel((0, 0))[:3] == (0, 0, 0)


def test_offset_positions_layer():
    bg = Layer(name="bg", image=_solid((10, 10), (255, 0, 0, 255)))
    top = Layer(name="top", image=_solid((4, 4), (0, 255, 0, 255)),
                offset=(3, 3))
    out = compose_layers([bg, top], (10, 10))
    # (0, 0) is bg (red)
    assert out.getpixel((0, 0))[:3] == (255, 0, 0)
    # (5, 5) is in the offset top layer (green)
    assert out.getpixel((5, 5))[:3] == (0, 255, 0)


def test_negative_offset_clips():
    bg = Layer(name="bg", image=_solid((10, 10), (255, 0, 0, 255)))
    top = Layer(name="top", image=_solid((4, 4), (0, 255, 0, 255)),
                offset=(-2, -2))
    out = compose_layers([bg, top], (10, 10))
    # Top covers (0,0)-(1,1); (5,5) still bg
    assert out.getpixel((0, 0))[:3] == (0, 255, 0)
    assert out.getpixel((5, 5))[:3] == (255, 0, 0)


def test_mask_hides_pixels():
    bg = Layer(name="bg", image=_solid((4, 4), (255, 0, 0, 255)))
    # Top covers everything in green, but mask is half black/half white
    mask = Image.new("L", (4, 4), 0)
    for y in range(4):
        for x in range(2, 4):
            mask.putpixel((x, y), 255)
    top = Layer(name="top", image=_solid((4, 4), (0, 255, 0, 255)),
                mask=mask)
    out = compose_layers([bg, top], (4, 4))
    # Left half: mask=0 → bg shows through → red
    assert out.getpixel((0, 0))[:3] == (255, 0, 0)
    # Right half: mask=255 → top shows → green
    assert out.getpixel((3, 0))[:3] == (0, 255, 0)


def test_invert_mask():
    m = Image.new("L", (4, 4), 100)
    inverted = invert_mask(m)
    assert inverted.getpixel((0, 0)) == 155


def test_make_white_mask():
    m = make_white_mask((10, 5))
    assert m.size == (10, 5)
    assert m.getpixel((0, 0)) == 255


def test_all_blend_modes_compose_without_error():
    """Smoke test: every documented mode produces a valid RGBA image."""
    bg = Layer(name="bg", image=_solid((4, 4), (100, 150, 200, 255)))
    for mode in BLEND_MODES:
        top = Layer(name="top", image=_solid((4, 4), (200, 100, 50, 200)),
                    blend_mode=mode)
        out = compose_layers([bg, top], (4, 4))
        assert out.size == (4, 4) and out.mode == "RGBA", f"mode {mode}"


def test_layer_copy_is_independent():
    img = _solid((4, 4), (100, 100, 100, 255))
    a = Layer(name="a", image=img, opacity=0.5)
    b = a.copy()
    b.opacity = 1.0
    b.image.putpixel((0, 0), (0, 0, 0, 255))
    assert a.opacity == 0.5
    assert a.image.getpixel((0, 0)) != (0, 0, 0, 255)
