"""Transforms change dims/pixels as expected."""
from PIL import Image

from server import transforms


def test_crop_changes_dimensions():
    img = Image.new("RGB", (40, 40), "white")
    out = transforms.crop(img, 5, 5, 25, 35)
    assert out.size == (20, 30)


def test_resize_changes_dimensions():
    img = Image.new("RGB", (40, 40), "white")
    out = transforms.resize(img, 10, 20, resample="nearest")
    assert out.size == (10, 20)


def test_rotate_expand_grows_canvas():
    img = Image.new("RGB", (40, 20), "white")
    out = transforms.rotate(img, 45, expand=True)
    assert out.size[0] >= 40 and out.size[1] >= 40


def test_flip_horizontal_inverts_x():
    img = Image.new("RGB", (4, 1), "white")
    img.putpixel((0, 0), (255, 0, 0))
    out = transforms.flip(img, "horizontal")
    assert out.getpixel((3, 0)) == (255, 0, 0)


def test_grayscale_drops_color():
    img = Image.new("RGB", (4, 4), "red")
    out = transforms.grayscale(img)
    px = out.getpixel((0, 0))
    if isinstance(px, tuple):
        assert px[0] == px[1] == px[2]
    else:
        assert 0 <= int(px) <= 255  # L mode


def test_invert_flips_channels():
    img = Image.new("RGB", (1, 1), (10, 20, 30))
    out = transforms.invert(img)
    assert out.getpixel((0, 0)) == (245, 235, 225)


def test_add_border_grows_canvas():
    img = Image.new("RGB", (10, 10), "white")
    out = transforms.add_border(img, 3, "black")
    assert out.size == (16, 16)


def test_apply_filter_blur_runs():
    img = Image.new("RGB", (16, 16), "white")
    out = transforms.apply_filter(img, "gaussian_blur", radius=1.5)
    assert out.size == (16, 16)
