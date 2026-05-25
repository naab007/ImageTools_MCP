"""Drawing primitives mutate pixels at expected locations."""
from PIL import Image

from server import drawing


def _rgba(img, x, y):
    return img.convert("RGBA").getpixel((x, y))


def test_pixel_sets_color():
    img = Image.new("RGB", (10, 10), "white")
    drawing.draw_pixel(img, 5, 5, "red")
    assert _rgba(img, 5, 5)[:3] == (255, 0, 0)
    assert _rgba(img, 0, 0)[:3] == (255, 255, 255)


def test_line_marks_endpoints():
    img = Image.new("RGB", (20, 20), "white")
    drawing.draw_line(img, 2, 2, 17, 17, "black", width=1)
    assert _rgba(img, 2, 2)[:3] == (0, 0, 0)
    assert _rgba(img, 17, 17)[:3] == (0, 0, 0)


def test_rectangle_fill_and_outline():
    img = Image.new("RGB", (30, 30), "white")
    drawing.draw_rectangle(img, 5, 5, 25, 25, fill="green", outline="red", width=2)
    # interior is green
    assert _rgba(img, 15, 15)[:3] == (0, 128, 0)
    # edge is red
    assert _rgba(img, 5, 15)[:3] == (255, 0, 0)


def test_flood_fill_changes_region():
    img = Image.new("RGB", (10, 10), "white")
    drawing.flood_fill(img, 0, 0, "blue")
    assert _rgba(img, 5, 5)[:3] == (0, 0, 255)


def test_pick_color_returns_pixel():
    img = Image.new("RGBA", (4, 4), (10, 20, 30, 40))
    assert drawing.pick_color(img, 2, 2) == (10, 20, 30, 40)


def test_brush_marks_endpoints():
    img = Image.new("RGB", (40, 40), "white")
    drawing.draw_brush(img, [(5, 5), (35, 35)], "red", size=3)
    assert _rgba(img, 5, 5)[:3] == (255, 0, 0)
    assert _rgba(img, 35, 35)[:3] == (255, 0, 0)


def test_eraser_clears_alpha_on_rgba():
    img = Image.new("RGBA", (20, 20), "red")
    drawing.eraser(img, [(5, 10), (15, 10)], size=4)
    # middle of stroke is transparent
    assert img.getpixel((10, 10))[3] == 0
