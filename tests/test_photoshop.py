"""Photoshop tools: gradients, adjustments, channels, painting, layer effects."""
import numpy as np
import pytest
from PIL import Image

from server import adjustments, channels, gradients, layer_effects, painting
from server.layers import Layer


# ---- gradients -------------------------------------------------------------

def test_linear_gradient_interpolates_endpoints():
    g = gradients.make_gradient(
        (100, 1), "linear",
        [{"position": 0.0, "color": "#000000"},
         {"position": 1.0, "color": "#ffffff"}],
    )
    assert g.getpixel((0, 0))[:3] == (0, 0, 0)
    assert g.getpixel((99, 0))[:3] == (255, 255, 255)
    # midpoint should be roughly gray
    r, gr, b, _ = g.getpixel((50, 0))
    assert 120 <= r <= 135


def test_radial_gradient_center_inside():
    g = gradients.make_gradient(
        (40, 40), "radial",
        [{"position": 0.0, "color": "white"},
         {"position": 1.0, "color": "black"}],
        center_x=20, center_y=20, radius=20,
    )
    # center is white (first stop)
    assert g.getpixel((20, 20))[:3] == (255, 255, 255)
    # corner is black (beyond radius)
    assert g.getpixel((0, 0))[:3] == (0, 0, 0)


def test_multistop_gradient():
    g = gradients.make_gradient(
        (300, 1), "linear",
        [{"position": 0.0, "color": "red"},
         {"position": 0.5, "color": "lime"},
         {"position": 1.0, "color": "blue"}],
    )
    assert g.getpixel((0, 0))[:3] == (255, 0, 0)
    assert g.getpixel((150, 0))[:3][1] > 200  # green-dominant midpoint
    assert g.getpixel((299, 0))[:3] == (0, 0, 255)


def test_unknown_gradient_type_raises():
    with pytest.raises(ValueError):
        gradients.make_gradient((10, 10), "spiral",
                                [{"position": 0.0, "color": "black"},
                                 {"position": 1.0, "color": "white"}])


def test_gradient_map_remaps_luminance():
    src = Image.new("RGB", (10, 1), (128, 128, 128))
    out = gradients.gradient_map(src, [
        {"position": 0.0, "color": "red"},
        {"position": 1.0, "color": "blue"},
    ])
    r, g, b, _ = out.getpixel((0, 0))
    # gray maps to mid-purple-ish in the red→blue ramp
    assert r > 100 and b > 100


# ---- adjustments -----------------------------------------------------------

def test_hue_shift_rotates_color():
    img = Image.new("RGB", (4, 4), (255, 0, 0))
    out = adjustments.hue_saturation_lightness(img, hue=66)  # ~120° shift
    r, g, b, _ = out.getpixel((0, 0))
    # Red shifted ~120° → green should dominate
    assert g > r and g > b


def test_saturation_to_zero_desaturates():
    img = Image.new("RGB", (4, 4), (255, 0, 0))
    out = adjustments.hue_saturation_lightness(img, saturation=-100)
    r, g, b, _ = out.getpixel((0, 0))
    assert abs(r - g) < 5 and abs(g - b) < 5  # gray-ish


def test_levels_input_black_white_clip():
    img = Image.new("RGB", (4, 4), (100, 100, 100))
    out = adjustments.levels(img, in_black=100, in_white=200)
    # 100 was input black so should become 0 output black
    assert out.getpixel((0, 0))[0] == 0


def test_levels_gamma_brightens_below_one():
    img = Image.new("RGB", (4, 4), (128, 128, 128))
    out = adjustments.levels(img, gamma=2.0)
    assert out.getpixel((0, 0))[0] > 128


def test_curves_identity_does_nothing():
    img = Image.new("RGB", (4, 4), (100, 100, 100))
    out = adjustments.curves(img, rgb_curve=[(0, 0), (255, 255)])
    assert abs(out.getpixel((0, 0))[0] - 100) <= 1


def test_threshold_produces_pure_black_white():
    grad = Image.new("RGB", (10, 1))
    for x in range(10):
        v = int(x * 28)
        grad.putpixel((x, 0), (v, v, v))
    out = adjustments.threshold(grad, level=128)
    for x in range(10):
        r, g, b, _ = out.getpixel((x, 0))
        assert (r, g, b) in [(0, 0, 0), (255, 255, 255)]


def test_vibrance_boosts_less_saturated_more():
    less_sat = Image.new("RGB", (1, 1), (180, 120, 120))
    more_sat = Image.new("RGB", (1, 1), (255, 0, 0))
    less_after = adjustments.vibrance(less_sat, amount=50).getpixel((0, 0))
    more_after = adjustments.vibrance(more_sat, amount=50).getpixel((0, 0))
    # less-saturated should have grown closer to red than already-saturated did
    less_delta = abs(less_after[0] - less_after[1])
    more_delta = abs(more_after[0] - more_after[1])
    assert less_delta > 60  # got more saturated
    assert more_delta == 255  # already maxed, no further change possible


def test_channel_mixer_swap_channels():
    img = Image.new("RGB", (4, 4), (255, 0, 0))
    # Map red → blue
    out = adjustments.channel_mixer(
        img, r_mix=(0, 0, 0), g_mix=(0, 0, 0), b_mix=(1, 0, 0),
    )
    assert out.getpixel((0, 0))[2] == 255  # blue dominant


def test_auto_levels_stretches_range():
    img = Image.new("RGB", (50, 1))
    for x in range(50):
        v = 100 + x  # 100..149
        img.putpixel((x, 0), (v, v, v))
    out = adjustments.auto_levels(img, clip=0.0)
    # After stretch, the darkest and brightest should hit 0/255
    px0 = out.getpixel((0, 0))
    px_end = out.getpixel((49, 0))
    assert px0[0] < 10
    assert px_end[0] > 245


def test_auto_levels_ignores_transparent_pixels():
    """Regression: transparent background must not skew the histogram.

    Old behaviour: the (0,0,0,0) transparent pixels acted as 0-value samples
    in the histogram, so the channel min was pinned at 0 regardless of the
    actual opaque content's range. With them excluded, a mid-range opaque
    region should stretch out toward 0/255."""
    img = Image.new("RGBA", (50, 1), (0, 0, 0, 0))
    # Opaque pixels span only 100..150 in the original.
    for x in range(10, 21):
        v = 100 + (x - 10) * 5  # 100, 105, ..., 150
        img.putpixel((x, 0), (v, v, v, 255))
    out = adjustments.auto_levels(img, clip=0.0)
    # The brightest opaque pixel (originally 150) should now hit near 255,
    # and the darkest (originally 100) should hit near 0 — proof that the
    # stretch is computed from the opaque range alone, not pulled to 0 by
    # the transparent region.
    darkest = out.getpixel((10, 0))[0]
    brightest = out.getpixel((20, 0))[0]
    assert darkest < 15
    assert brightest > 240
    # Transparent pixels still have alpha 0
    assert out.getpixel((0, 0))[3] == 0


def test_auto_contrast_skips_fully_transparent_image():
    img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    out = adjustments.auto_contrast(img)
    # No crash, no quantile from empty array
    assert out.size == (20, 20)


# ---- channels --------------------------------------------------------------

def test_extract_then_merge_roundtrip():
    src = Image.new("RGB", (8, 8), (200, 100, 50))
    r = channels.extract_channel(src, "R")
    g = channels.extract_channel(src, "G")
    b = channels.extract_channel(src, "B")
    merged = channels.merge_channels(r, g, b)
    assert merged.getpixel((0, 0)) == (200, 100, 50)


def test_extract_luminance():
    src = Image.new("RGB", (4, 4), (255, 0, 0))
    L = channels.extract_channel(src, "L")
    # red luminance: 0.299*255 ≈ 76
    assert abs(L.getpixel((0, 0)) - 76) <= 2


def test_merge_channels_with_alpha():
    r = Image.new("L", (4, 4), 100)
    g = Image.new("L", (4, 4), 150)
    b = Image.new("L", (4, 4), 200)
    a = Image.new("L", (4, 4), 128)
    out = channels.merge_channels(r, g, b, a)
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0)) == (100, 150, 200, 128)


def test_extract_rejects_bad_channel():
    src = Image.new("RGB", (4, 4))
    with pytest.raises(ValueError):
        channels.extract_channel(src, "Z")


# ---- painting --------------------------------------------------------------

def test_clone_stamp_copies_source():
    img = Image.new("RGBA", (50, 50), (255, 255, 255, 255))
    # Put a red dot at (10, 10)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            img.putpixel((10 + dx, 10 + dy), (255, 0, 0, 255))
    out = painting.clone_stamp(img, [(35, 35)], source_x=10, source_y=10,
                               size=10, hardness=1.0, opacity=1.0)
    # Pixel at (35, 35) should now be red
    assert out.getpixel((35, 35))[0] > 200


def test_clone_stamp_handles_source_off_image():
    """Regression: when the source point sits near the image edge so that
    the brush sample would extend out of bounds, the destination must NOT
    get garbage (older code sampled from a misaligned in-bounds window)."""
    # Bottom-right is red, everywhere else white
    img = Image.new("RGBA", (50, 50), (255, 255, 255, 255))
    for x in range(40, 50):
        for y in range(40, 50):
            img.putpixel((x, y), (255, 0, 0, 255))
    # Source at (48, 48) — brush radius 10 makes the sample window extend
    # beyond image bounds (52, 52). The destination at (10, 10) should still
    # come from coords that overlap the red corner (i.e., turn pink-ish on
    # the side where source falls inside the image).
    out = painting.clone_stamp(img, [(10, 10)], source_x=48, source_y=48,
                               size=20, hardness=1.0, opacity=1.0)
    # The pixel at (8, 8) maps to source (46, 46) which is red — should be red.
    # The pixel at (12, 12) maps to source (50, 50) which is OUT of bounds —
    # should be unchanged (still white) under the new edge-aware code.
    px_in_bounds = out.getpixel((8, 8))
    px_out_of_bounds = out.getpixel((12, 12))
    assert px_in_bounds[0] > 200 and px_in_bounds[1] < 50  # got red
    assert px_out_of_bounds == (255, 255, 255, 255)         # unchanged


def test_dodge_brightens_along_stroke():
    img = Image.new("RGBA", (40, 40), (100, 100, 100, 255))
    out = painting.dodge_brush(img, [(20, 20)], size=20, hardness=1.0,
                               exposure=1.0)
    center = out.getpixel((20, 20))
    edge = out.getpixel((0, 0))
    assert center[0] > 120  # center brightened
    assert edge[0] == 100   # edge untouched


def test_burn_darkens_along_stroke():
    img = Image.new("RGBA", (40, 40), (200, 200, 200, 255))
    out = painting.burn_brush(img, [(20, 20)], size=20, hardness=1.0,
                              exposure=1.0)
    center = out.getpixel((20, 20))
    assert center[0] < 180


def test_blur_brush_runs():
    img = Image.new("RGBA", (40, 40), (100, 100, 100, 255))
    out = painting.blur_brush(img, [(10, 10), (30, 30)],
                              size=10, strength=1.0, radius=3.0)
    assert out.size == (40, 40)


def test_sharpen_brush_runs():
    img = Image.new("RGBA", (40, 40), (100, 100, 100, 255))
    out = painting.sharpen_brush(img, [(20, 20)], size=10, strength=1.0)
    assert out.size == (40, 40)


# ---- layer effects ---------------------------------------------------------

def test_drop_shadow_creates_padded_layer():
    src = Layer(name="src", image=Image.new("RGBA", (20, 20), (255, 0, 0, 255)))
    shadow = layer_effects.make_drop_shadow_layer(src, offset_x=5, offset_y=5,
                                                  blur=4.0)
    # shadow image should be bigger than source due to padding
    assert shadow.image.width > src.image.width
    assert shadow.image.height > src.image.height
    # offset is adjusted negatively (padding compensation)
    assert shadow.offset[0] < 0 and shadow.offset[1] < 0


def test_outer_glow_uses_screen_blend():
    src = Layer(name="src", image=Image.new("RGBA", (20, 20), (255, 0, 0, 255)))
    glow = layer_effects.make_outer_glow_layer(src)
    assert glow.blend_mode == "screen"


def test_layer_stroke_inside_position():
    src = Layer(name="src", image=Image.new("RGBA", (20, 20), (255, 0, 0, 255)))
    stroke = layer_effects.make_stroke_layer(src, width=3, position="inside")
    # stroke layer exists and has some opaque pixels
    arr = np.asarray(stroke.image)
    assert arr[..., 3].max() > 0


def test_layer_stroke_outside_grows_layer():
    src = Layer(name="src", image=Image.new("RGBA", (20, 20), (255, 0, 0, 255)))
    stroke = layer_effects.make_stroke_layer(src, width=3, position="outside")
    # stroke layer is larger than source (padding for outward growth)
    assert stroke.image.width > src.image.width
