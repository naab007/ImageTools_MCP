"""Canvas store with layer stack: lifecycle, layer ops, undo/redo."""
from PIL import Image

from server.canvas import CanvasStore
from server.layers import Layer


def test_put_image_creates_single_layer_canvas():
    s = CanvasStore()
    img = Image.new("RGB", (10, 10), "red")
    cid = s.put_image(img)
    e = s.entry(cid)
    assert (e.width, e.height) == (10, 10)
    assert len(e.layers) == 1
    assert e.active_index == 0
    s.close(cid)


def test_close_makes_entry_unreachable():
    s = CanvasStore()
    cid = s.put_image(Image.new("RGB", (4, 4)))
    s.close(cid)
    try:
        s.entry(cid)
    except KeyError:
        return
    raise AssertionError("expected KeyError after close")


def test_compose_returns_rgba_image_of_canvas_size():
    s = CanvasStore()
    cid = s.put_image(Image.new("RGB", (20, 15), "blue"))
    composite = s.compose(cid)
    assert composite.size == (20, 15)
    assert composite.mode == "RGBA"


def test_add_remove_layer():
    s = CanvasStore()
    cid = s.put_image(Image.new("RGB", (10, 10)))
    idx = s.add_layer(cid, name="L2")
    assert idx == 1
    assert len(s.entry(cid).layers) == 2
    s.remove_layer(cid, 1)
    assert len(s.entry(cid).layers) == 1


def test_cannot_remove_last_layer():
    s = CanvasStore()
    cid = s.put_image(Image.new("RGB", (4, 4)))
    try:
        s.remove_layer(cid, 0)
    except ValueError:
        return
    raise AssertionError("expected ValueError removing last layer")


def test_reorder_layer_updates_active_index():
    s = CanvasStore()
    cid = s.put_image(Image.new("RGB", (4, 4)))
    s.add_layer(cid, name="A")
    s.add_layer(cid, name="B")
    # Stack now: [bg=0, A=1, B=2 (active)]
    s.reorder_layer(cid, 2, 0)  # move B to bottom
    e = s.entry(cid)
    assert [l.name for l in e.layers] == ["B", "Background", "A"]
    assert e.layers[e.active_index].name == "B"


def test_snapshot_then_undo_restores_layer_stack():
    s = CanvasStore()
    cid = s.put_image(Image.new("RGB", (4, 4), "white"))
    s.snapshot(cid)
    s.add_layer(cid, name="New")
    assert len(s.entry(cid).layers) == 2
    assert s.undo(cid) is True
    assert len(s.entry(cid).layers) == 1
    assert s.redo(cid) is True
    assert len(s.entry(cid).layers) == 2


def test_duplicate_layer_creates_independent_copy():
    s = CanvasStore()
    cid = s.put_image(Image.new("RGB", (4, 4), "red"))
    s.duplicate_layer(cid, 0)
    e = s.entry(cid)
    assert len(e.layers) == 2
    # Mutating one doesn't affect the other (different image objects).
    e.layers[0].image.putpixel((0, 0), (255, 255, 0, 255))
    assert e.layers[1].image.getpixel((0, 0)) != (255, 255, 0, 255)


def test_merge_down_combines_two_layers():
    s = CanvasStore()
    cid = s.put_image(Image.new("RGBA", (4, 4), (255, 0, 0, 255)),
                      layer_name="Background")
    s.add_layer(cid, name="Top")
    e = s.entry(cid)
    # Make top layer pure green opaque
    e.layers[1].image = Image.new("RGBA", (4, 4), (0, 255, 0, 255))
    s.merge_down(cid, 1)
    assert len(e.layers) == 1
    # Result should be green (top opaque over red)
    assert e.layers[0].image.getpixel((0, 0))[:3] == (0, 255, 0)


def test_flatten_collapses_all_layers():
    s = CanvasStore()
    cid = s.put_image(Image.new("RGB", (4, 4), "white"))
    s.add_layer(cid)
    s.add_layer(cid)
    s.flatten(cid)
    assert len(s.entry(cid).layers) == 1
    assert s.entry(cid).layers[0].name == "Background"


def test_summary_reports_layer_counts():
    s = CanvasStore()
    cid = s.put_image(Image.new("RGB", (4, 4)))
    s.add_layer(cid)
    [info] = [i for i in s.summary() if i["canvas_id"] == cid]
    assert info["n_layers"] == 2
    assert info["active_index"] == 1
