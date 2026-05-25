"""PSD layered roundtrip: layers, opacity, blend modes, visibility preserved."""
from pathlib import Path

import pytest
from PIL import Image

from server.layers import Layer
from server.psd_io import load_psd, save_psd


def _solid(size, color):
    return Image.new("RGBA", size, color)


def test_roundtrip_preserves_layer_count(tmp_path: Path):
    layers = [
        Layer(name="BG", image=_solid((50, 30), (255, 0, 0, 255))),
        Layer(name="Mid", image=_solid((50, 30), (0, 255, 0, 128))),
        Layer(name="Top", image=_solid((50, 30), (0, 0, 255, 200))),
    ]
    out = tmp_path / "doc.psd"
    info = save_psd(layers, (50, 30), str(out))
    assert info["layers"] == 3
    assert out.exists()

    w, h, loaded = load_psd(str(out))
    assert (w, h) == (50, 30)
    assert len(loaded) == 3
    assert [l.name for l in loaded] == ["BG", "Mid", "Top"]


def test_roundtrip_preserves_opacity_and_blend(tmp_path: Path):
    layers = [
        Layer(name="bg", image=_solid((20, 20), (200, 200, 200, 255))),
        Layer(name="top", image=_solid((20, 20), (255, 0, 0, 255)),
              opacity=0.5, blend_mode="multiply"),
    ]
    out = tmp_path / "doc.psd"
    save_psd(layers, (20, 20), str(out))

    _, _, loaded = load_psd(str(out))
    top = loaded[1]
    # opacity is stored as 0-255 in PSD; expect roughly 0.5
    assert abs(top.opacity - 0.5) < 0.01
    assert top.blend_mode == "multiply"


def test_roundtrip_preserves_visibility(tmp_path: Path):
    layers = [
        Layer(name="bg", image=_solid((10, 10), (0, 0, 0, 255))),
        Layer(name="hidden", image=_solid((10, 10), (255, 0, 0, 255)),
              visible=False),
    ]
    out = tmp_path / "doc.psd"
    save_psd(layers, (10, 10), str(out))
    _, _, loaded = load_psd(str(out))
    assert loaded[1].visible is False


def test_roundtrip_preserves_offset(tmp_path: Path):
    layers = [
        Layer(name="bg", image=_solid((40, 40), (255, 255, 255, 255))),
        Layer(name="floating", image=_solid((10, 10), (0, 0, 255, 255)),
              offset=(15, 10)),
    ]
    out = tmp_path / "doc.psd"
    save_psd(layers, (40, 40), str(out))
    _, _, loaded = load_psd(str(out))
    assert loaded[1].offset == (15, 10)


def test_mask_baked_into_alpha_on_save(tmp_path: Path):
    # Mask: top half visible (white), bottom half hidden (black)
    mask = Image.new("L", (10, 10), 0)
    for y in range(5):
        for x in range(10):
            mask.putpixel((x, y), 255)
    layers = [
        Layer(name="bg", image=_solid((10, 10), (255, 0, 0, 255))),
        Layer(name="masked", image=_solid((10, 10), (0, 255, 0, 255)),
              mask=mask),
    ]
    out = tmp_path / "doc.psd"
    info = save_psd(layers, (10, 10), str(out))
    assert info["masks_baked_to_alpha"] == 1

    _, _, loaded = load_psd(str(out))
    top = loaded[1]
    # Mask was baked into alpha, so bottom half of top layer is transparent.
    # Reading back: the layer image's alpha at (0, 8) (bottom) should be 0.
    bottom_alpha = top.image.getpixel((0, 8))[3]
    top_alpha = top.image.getpixel((0, 0))[3]
    assert bottom_alpha < top_alpha  # bottom masked away
