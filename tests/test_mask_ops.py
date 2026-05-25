"""Mask post-processing — feathering, expand/contract, refine, RGBA combine."""
import numpy as np
import pytest
from PIL import Image

from server import mask_ops


def _disc_mask(size=64, radius=20):
    """Return a binary L-mode disc mask centered in the image."""
    arr = np.zeros((size, size), dtype=np.uint8)
    cx = cy = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    arr[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius] = 255
    return Image.fromarray(arr, mode="L")


def test_feather_gaussian_softens_edges():
    m = _disc_mask()
    out = mask_ops.feather_mask(m, radius=4, method="gaussian")
    assert out.mode == "L"
    arr = np.asarray(out)
    # Edge pixels should be intermediate values (not pure 0 or 255)
    edge_vals = arr[20:24, 32:36]
    assert (edge_vals > 0).any() and (edge_vals < 255).any()


def test_feather_zero_radius_is_noop():
    m = _disc_mask()
    out = mask_ops.feather_mask(m, radius=0)
    assert (np.asarray(out) == np.asarray(m)).all()


def test_feather_inside_keeps_outer_boundary():
    """Inside-feather: pixels outside the original mask must stay 0."""
    m = _disc_mask()
    out = mask_ops.feather_mask(m, radius=5, method="inside")
    orig = np.asarray(m)
    feat = np.asarray(out)
    # Wherever orig was 0, feat must also be 0
    assert (feat[orig == 0] == 0).all()


def test_feather_outside_keeps_inner_boundary():
    """Outside-feather: pixels inside the original mask must stay 255."""
    m = _disc_mask()
    out = mask_ops.feather_mask(m, radius=5, method="outside")
    orig = np.asarray(m)
    feat = np.asarray(out)
    # Wherever orig was 255, feat must also be 255
    assert (feat[orig == 255] == 255).all()
    # And the outside should have grown a halo
    assert (feat[orig == 0] > 0).any()


def test_feather_matte_falls_back_without_source():
    """matte without source_image should still produce a result (fallback
    to gaussian) instead of crashing."""
    m = _disc_mask()
    out = mask_ops.feather_mask(m, radius=3, method="matte")
    assert out.mode == "L"
    assert out.size == m.size


def test_feather_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        mask_ops.feather_mask(_disc_mask(), radius=2, method="nonexistent")


def test_expand_mask_grows():
    m = _disc_mask(radius=10)
    expanded = mask_ops.expand_mask(m, pixels=5)
    assert np.asarray(expanded).sum() > np.asarray(m).sum()


def test_contract_mask_shrinks():
    m = _disc_mask(radius=15)
    contracted = mask_ops.contract_mask(m, pixels=5)
    assert np.asarray(contracted).sum() < np.asarray(m).sum()


def test_expand_zero_pixels_is_noop():
    m = _disc_mask()
    assert np.asarray(mask_ops.expand_mask(m, pixels=0)).tolist() == np.asarray(m).tolist()


def test_refine_mask_fills_small_holes():
    # Disc with a small hole punched out
    m = _disc_mask(radius=20)
    arr = np.asarray(m).copy()
    arr[30:34, 30:34] = 0
    holed = Image.fromarray(arr, mode="L")
    refined = mask_ops.refine_mask(holed, fill_holes_px=3)
    # The hole should be closed
    assert np.asarray(refined)[30:34, 30:34].mean() > 100


def test_apply_mask_as_alpha_returns_rgba():
    img = Image.new("RGB", (64, 64), "red")
    mask = _disc_mask()
    rgba = mask_ops.apply_mask_as_alpha(img, mask)
    assert rgba.mode == "RGBA"
    assert rgba.size == (64, 64)
    # Center pixel should be opaque red, corner transparent
    cx_pixel = rgba.getpixel((32, 32))
    corner = rgba.getpixel((0, 0))
    assert cx_pixel[:3] == (255, 0, 0) and cx_pixel[3] == 255
    assert corner[3] == 0


def test_apply_mask_as_alpha_resizes_mismatched_mask():
    img = Image.new("RGB", (128, 128), "blue")
    mask = _disc_mask(size=64)  # smaller
    rgba = mask_ops.apply_mask_as_alpha(img, mask)
    assert rgba.size == (128, 128)


def test_apply_mask_handles_rgba_alpha_input():
    """If a mask is passed as RGBA, alpha channel should be used."""
    img = Image.new("RGB", (64, 64), "red")
    mask_arr = np.zeros((64, 64, 4), dtype=np.uint8)
    mask_arr[..., 3] = np.asarray(_disc_mask())
    rgba_mask = Image.fromarray(mask_arr, mode="RGBA")
    out = mask_ops.apply_mask_as_alpha(img, rgba_mask)
    assert out.mode == "RGBA"
