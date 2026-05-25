"""Stateless pipeline + one-shot file-in/file-out helpers."""
from pathlib import Path

import pytest
from PIL import Image

from server import io_formats, stateless


def _make(tmp: Path, name: str, color: str = "white",
          size: tuple[int, int] = (40, 30)) -> Path:
    p = tmp / name
    Image.new("RGB", size, color).save(p)
    return p


def test_pipeline_resize_then_grayscale(tmp_path: Path):
    src = _make(tmp_path, "src.png", "red")
    out = stateless.apply_pipeline(io_formats.load_image(str(src)), [
        {"op": "resize", "width": 20, "height": 15},
        {"op": "grayscale"},
    ])
    assert out.size == (20, 15)
    px = out.getpixel((0, 0))
    if isinstance(px, tuple):
        assert px[0] == px[1] == px[2]


def test_pipeline_drawing_then_save(tmp_path: Path):
    src = _make(tmp_path, "src.png", "white", size=(60, 60))
    img = io_formats.load_image(str(src))
    out = stateless.apply_pipeline(img, [
        {"op": "draw_rectangle", "x1": 5, "y1": 5, "x2": 55, "y2": 55,
         "fill": "blue"},
        {"op": "draw_text", "x": 10, "y": 10, "text": "Hi",
         "color": "white", "size": 12},
    ])
    dst = tmp_path / "out.png"
    info = io_formats.save_image(out, str(dst))
    assert Path(info["path"]).exists()
    # interior of rectangle is blue
    assert out.getpixel((30, 30))[:3] == (0, 0, 255)


def test_thumbnail_preserves_aspect(tmp_path: Path):
    src = _make(tmp_path, "src.png", "white", size=(800, 400))
    out = stateless.apply_pipeline(io_formats.load_image(str(src)),
                                   [{"op": "thumbnail", "max_size": 200}])
    assert out.size == (200, 100)


def test_thumbnail_skips_upscale(tmp_path: Path):
    src = _make(tmp_path, "src.png", "white", size=(50, 50))
    out = stateless.apply_pipeline(io_formats.load_image(str(src)),
                                   [{"op": "thumbnail", "max_size": 500}])
    assert out.size == (50, 50)


def test_unknown_op_raises(tmp_path: Path):
    src = _make(tmp_path, "src.png")
    with pytest.raises(ValueError) as excinfo:
        stateless.apply_pipeline(io_formats.load_image(str(src)),
                                 [{"op": "no_such_op"}])
    assert "no_such_op" in str(excinfo.value)


def test_bad_op_spec_raises(tmp_path: Path):
    src = _make(tmp_path, "src.png")
    img = io_formats.load_image(str(src))
    with pytest.raises(ValueError):
        stateless.apply_pipeline(img, ["not a dict"])
    with pytest.raises(ValueError):
        stateless.apply_pipeline(img, [{"missing_op_key": 1}])


def test_op_error_includes_index_and_name(tmp_path: Path):
    src = _make(tmp_path, "src.png")
    img = io_formats.load_image(str(src))
    with pytest.raises(ValueError) as excinfo:
        # crop with zero-area rectangle — transforms.crop raises ValueError
        stateless.apply_pipeline(img, [
            {"op": "grayscale"},
            {"op": "crop", "x1": 5, "y1": 5, "x2": 5, "y2": 5},
        ])
    msg = str(excinfo.value)
    assert "operation 1" in msg and "crop" in msg


def test_all_ops_listed():
    expected = {"resize", "crop", "rotate", "flip", "grayscale", "invert",
                "adjust", "apply_filter", "draw_rectangle", "draw_text",
                "thumbnail"}
    assert expected.issubset(set(stateless.ALL_OPS))
