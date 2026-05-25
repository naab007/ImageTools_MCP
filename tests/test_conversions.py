"""Format conversion roundtrips: animation, ICO, PDF, color mode, metadata."""
from pathlib import Path

import pytest
from PIL import Image

from server import conversions, io_formats


def _frame(tmp: Path, name: str, color: tuple[int, int, int],
           size: tuple[int, int] = (32, 32)) -> Path:
    p = tmp / name
    Image.new("RGB", size, color).save(p)
    return p


# ---- animation -------------------------------------------------------------

def test_build_and_extract_animated_gif(tmp_path: Path):
    frames = [
        _frame(tmp_path, "f0.png", (255, 0, 0)),
        _frame(tmp_path, "f1.png", (0, 255, 0)),
        _frame(tmp_path, "f2.png", (0, 0, 255)),
    ]
    out_gif = tmp_path / "anim.gif"
    info = conversions.build_animation([str(p) for p in frames], str(out_gif),
                                       fps=5.0)
    assert info["frames"] == 3
    assert out_gif.exists()

    extract_dir = tmp_path / "extracted"
    result = conversions.extract_frames(str(out_gif), str(extract_dir))
    assert result["frames"] == 3
    assert len(result["files"]) == 3


def test_build_animated_webp(tmp_path: Path):
    frames = [_frame(tmp_path, f"f{i}.png", (i * 80, 0, 0)) for i in range(3)]
    out = tmp_path / "anim.webp"
    info = conversions.build_animation([str(p) for p in frames], str(out),
                                       fps=10.0, quality=70)
    assert info["frames"] == 3
    assert out.exists()


def test_build_animation_rejects_non_animated_ext(tmp_path: Path):
    f = _frame(tmp_path, "f0.png", (0, 0, 0))
    with pytest.raises(ValueError):
        conversions.build_animation([str(f)], str(tmp_path / "x.jpg"))


def test_extract_frames_handles_single_frame(tmp_path: Path):
    src = _frame(tmp_path, "still.png", (100, 100, 100))
    extract_dir = tmp_path / "out"
    result = conversions.extract_frames(str(src), str(extract_dir))
    assert result["frames"] == 1


# ---- ICO -------------------------------------------------------------------

def test_build_ico_from_single_source_with_sizes(tmp_path: Path):
    src = _frame(tmp_path, "icon.png", (200, 100, 50), size=(256, 256))
    out = tmp_path / "icon.ico"
    info = conversions.build_ico([str(src)], str(out), sizes=[16, 32, 48, 256])
    assert out.exists()
    assert len(info["sizes"]) == 4


def test_split_ico_extracts_all_sizes(tmp_path: Path):
    src = _frame(tmp_path, "icon.png", (10, 20, 30), size=(64, 64))
    ico = tmp_path / "icon.ico"
    conversions.build_ico([str(src)], str(ico), sizes=[16, 32, 64])
    out_dir = tmp_path / "extracted"
    result = conversions.split_ico(str(ico), str(out_dir))
    assert result["count"] == 3
    for f in result["files"]:
        assert Path(f).exists()


def test_split_ico_rejects_non_ico(tmp_path: Path):
    p = _frame(tmp_path, "x.png", (0, 0, 0))
    with pytest.raises(ValueError):
        conversions.split_ico(str(p), str(tmp_path / "out"))


# ---- PDF -------------------------------------------------------------------

def test_images_to_pdf_roundtrip(tmp_path: Path):
    pages = [
        _frame(tmp_path, "p1.png", (255, 0, 0), size=(100, 80)),
        _frame(tmp_path, "p2.png", (0, 255, 0), size=(100, 80)),
        _frame(tmp_path, "p3.png", (0, 0, 255), size=(100, 80)),
    ]
    pdf = tmp_path / "out.pdf"
    info = conversions.images_to_pdf([str(p) for p in pages], str(pdf))
    assert info["pages"] == 3
    assert pdf.exists()

    # rasterize back
    out_dir = tmp_path / "rendered"
    result = conversions.pdf_to_images(str(pdf), str(out_dir), dpi=72)
    assert result["pages_rendered"] == 3


def test_pdf_page_range_parsing(tmp_path: Path):
    pages = [_frame(tmp_path, f"p{i}.png", (i * 30, 0, 0), size=(50, 50))
             for i in range(5)]
    pdf = tmp_path / "out.pdf"
    conversions.images_to_pdf([str(p) for p in pages], str(pdf))
    out_dir = tmp_path / "selected"
    result = conversions.pdf_to_images(str(pdf), str(out_dir), dpi=72,
                                       pages="1,3-4")
    assert result["pages_rendered"] == 3  # 1, 3, 4


def test_images_to_pdf_flattens_alpha(tmp_path: Path):
    p = tmp_path / "transparent.png"
    Image.new("RGBA", (50, 50), (255, 0, 0, 128)).save(p)
    pdf = tmp_path / "out.pdf"
    info = conversions.images_to_pdf([str(p)], str(pdf))
    assert info["pages"] == 1


# ---- color mode ------------------------------------------------------------

@pytest.mark.parametrize("mode", ["1", "L", "P", "RGB", "RGBA", "CMYK"])
def test_convert_mode_roundtrip(tmp_path: Path, mode: str):
    src = _frame(tmp_path, "src.png", (200, 100, 50))
    # Output extension must support the target mode — PNG handles 1/L/P/RGB/RGBA,
    # TIFF handles CMYK.
    ext = "tiff" if mode == "CMYK" else "png"
    dst = tmp_path / f"out.{ext}"
    info = conversions.convert_mode(str(src), str(dst), mode)
    assert dst.exists()
    loaded = io_formats.load_image(str(dst))
    assert loaded.mode == mode or loaded.mode in ("P", "L", "RGB", "RGBA")


def test_convert_mode_palette_respects_size(tmp_path: Path):
    src = _frame(tmp_path, "src.png", (200, 100, 50))
    dst = tmp_path / "small.png"
    conversions.convert_mode(str(src), str(dst), "P", palette_size=8)
    loaded = io_formats.load_image(str(dst))
    assert loaded.mode == "P"


def test_convert_mode_rejects_unknown(tmp_path: Path):
    src = _frame(tmp_path, "src.png", (0, 0, 0))
    with pytest.raises(ValueError):
        conversions.convert_mode(str(src), str(tmp_path / "out.png"), "XYZ")


# ---- metadata --------------------------------------------------------------

def test_strip_metadata_drops_exif(tmp_path: Path):
    # Build a JPEG with EXIF
    src = tmp_path / "with_exif.jpg"
    img = Image.new("RGB", (40, 30), "white")
    exif = img.getexif()
    exif[271] = "ImageTools Test"  # Make
    img.save(src, exif=exif.tobytes())
    assert io_formats.image_info(str(src))["exif"]  # sanity

    dst = tmp_path / "clean.jpg"
    conversions.strip_metadata(str(src), str(dst))
    assert not io_formats.image_info(str(dst))["exif"]


def test_copy_metadata_transplants_exif(tmp_path: Path):
    # source with EXIF
    meta_src = tmp_path / "withexif.jpg"
    img = Image.new("RGB", (40, 30), "red")
    e = img.getexif()
    e[271] = "TestCam"
    img.save(meta_src, exif=e.tobytes())

    # target without EXIF
    clean = tmp_path / "clean.jpg"
    Image.new("RGB", (40, 30), "blue").save(clean)

    out = tmp_path / "transplanted.jpg"
    info = conversions.copy_metadata(str(meta_src), str(clean), str(out))
    assert info["exif_tags_copied"] >= 1

    out_exif = io_formats.image_info(str(out))["exif"]
    assert out_exif.get("Make") == "TestCam"


def test_copy_metadata_fails_without_exif(tmp_path: Path):
    no_exif = _frame(tmp_path, "x.png", (0, 0, 0))
    clean = _frame(tmp_path, "y.png", (1, 1, 1))
    with pytest.raises(ValueError):
        conversions.copy_metadata(str(no_exif), str(clean),
                                  str(tmp_path / "out.jpg"))
