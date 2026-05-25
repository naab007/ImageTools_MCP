"""SAM 2 module — concurrency, lazy load, and routing tests.

The real sam2 package isn't installed in CI (it's an optional extra), so
we monkeypatch ``_check_available``, ``_get_predictor``, and
``_get_mask_generator`` to return fakes. This verifies the cache-state +
idle-timeout + input-shape machinery without needing the actual model.
"""
import threading
import time

import numpy as np
import pytest
from PIL import Image

from server import sam


# ---- timeout config --------------------------------------------------------

def test_set_idle_timeout_clamps_negative():
    info = sam.set_idle_timeout(-5)
    assert info["idle_timeout_s"] == 0.0
    assert info["auto_evict_enabled"] is False
    sam.set_idle_timeout(3600)


def test_set_idle_timeout_zero_disables():
    info = sam.set_idle_timeout(0)
    assert info["auto_evict_enabled"] is False
    sam.set_idle_timeout(3600)


# ---- sweeper ---------------------------------------------------------------

def test_sweep_skips_when_recently_touched(monkeypatch):
    """Regression for TOCTOU: a _touch right before the check should keep
    the predictor alive; only stale state evicts."""
    monkeypatch.setattr(sam, "_idle_timeout_s", 0.1)
    sam._predictor = object()
    sam._loaded_model = "fake"
    sam._touch()

    sam._sweep_once()
    assert sam._predictor is not None, "fresh model should not be evicted"

    sam._last_used = time.monotonic() - 10
    sam._sweep_once()
    assert sam._predictor is None, "stale model should be evicted"


def test_touch_during_sweep_doesnt_race():
    """_touch under heavy concurrency must not raise."""
    sam._predictor = object()
    sam._loaded_model = "fake"
    sam._touch()

    stop = threading.Event()
    errors: list[BaseException] = []

    def hammer_touch():
        while not stop.is_set():
            try:
                sam._touch()
            except BaseException as e:
                errors.append(e)
                return

    def hammer_sweep():
        while not stop.is_set():
            try:
                sam._sweep_once()
            except BaseException as e:
                errors.append(e)
                return

    threads = [threading.Thread(target=hammer_touch) for _ in range(3)]
    threads += [threading.Thread(target=hammer_sweep) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert not errors, f"races detected: {errors[:3]}"
    sam._predictor = None
    sam._loaded_model = None
    sam._last_used = None


# ---- segmentation routing --------------------------------------------------

class _FakePredictor:
    """Minimal stand-in for SAM2ImagePredictor that records what it was
    called with and returns a synthetic mask."""

    def __init__(self):
        self.set_image_calls = []
        self.predict_calls = []

    def set_image(self, rgb):
        self.set_image_calls.append(rgb.shape)

    def predict(self, *, point_coords=None, point_labels=None, box=None,
                multimask_output=True):
        self.predict_calls.append({
            "points": None if point_coords is None else point_coords.tolist(),
            "labels": None if point_labels is None else point_labels.tolist(),
            "box": None if box is None else box.tolist(),
            "multimask": multimask_output,
        })
        # Two candidate masks, second is "better"
        masks = np.zeros((2, 64, 64), dtype=bool)
        masks[1, 10:30, 10:30] = True
        scores = np.array([0.5, 0.9])
        return masks, scores, None


def _install_fake(monkeypatch):
    """Monkeypatch availability + predictor loader so tests run dep-free."""
    fake = _FakePredictor()
    monkeypatch.setattr(sam, "_check_available", lambda: None)
    monkeypatch.setattr(sam, "_get_predictor", lambda model=sam.DEFAULT_MODEL: fake)
    # _run_predictor imports torch; stub it via a minimal shim that supports
    # ``torch.inference_mode()`` as a context manager.
    import sys, types
    if "torch" not in sys.modules:
        torch_stub = types.SimpleNamespace()
        class _InferenceCtx:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        torch_stub.inference_mode = _InferenceCtx
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch_stub.backends = types.SimpleNamespace()
        sys.modules["torch"] = torch_stub
    return fake


def test_segment_with_points_picks_best_mask(monkeypatch):
    fake = _install_fake(monkeypatch)
    img = Image.new("RGB", (64, 64), "white")
    mask, score = sam.segment_with_points(img, [[15, 15]], [1])
    # Best of the two candidate scores (0.5, 0.9) is 0.9
    assert score == pytest.approx(0.9)
    assert mask.mode == "L"
    assert mask.size == (64, 64)
    # The "better" candidate had pixels set in (10..30, 10..30)
    assert mask.getpixel((20, 20)) == 255
    assert mask.getpixel((50, 50)) == 0
    # Predictor saw our coords + label
    assert fake.predict_calls[0]["points"] == [[15, 15]]
    assert fake.predict_calls[0]["labels"] == [1]


def test_segment_with_points_default_labels(monkeypatch):
    fake = _install_fake(monkeypatch)
    img = Image.new("RGB", (64, 64), "white")
    sam.segment_with_points(img, [[10, 10], [20, 20]])
    assert fake.predict_calls[-1]["labels"] == [1, 1]


def test_segment_with_points_labels_length_mismatch(monkeypatch):
    _install_fake(monkeypatch)
    img = Image.new("RGB", (64, 64), "white")
    with pytest.raises(ValueError, match="labels length"):
        sam.segment_with_points(img, [[10, 10], [20, 20]], [1])


def test_segment_with_points_empty(monkeypatch):
    _install_fake(monkeypatch)
    img = Image.new("RGB", (64, 64), "white")
    with pytest.raises(ValueError, match="at least one point"):
        sam.segment_with_points(img, [])


def test_segment_with_box(monkeypatch):
    fake = _install_fake(monkeypatch)
    img = Image.new("RGB", (64, 64), "white")
    mask, score = sam.segment_with_box(img, 5, 5, 60, 60)
    assert score > 0
    assert mask.mode == "L"
    assert fake.predict_calls[-1]["box"] == [5, 5, 60, 60]


def test_mask_array_to_image_bool_to_255():
    arr = np.zeros((4, 4), dtype=bool)
    arr[1, 1] = True
    img = sam._mask_array_to_image(arr)
    assert img.getpixel((1, 1)) == 255
    assert img.getpixel((0, 0)) == 0


def test_mask_array_to_image_float():
    arr = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    img = sam._mask_array_to_image(arr)
    # 1.5 clamps to 1.0 → 255
    assert img.getpixel((1, 1)) == 255
    assert img.getpixel((0, 1)) == 255  # 1.0
    assert img.getpixel((0, 0)) == 0


# ---- status / load / unload ------------------------------------------------

def test_status_unavailable_returns_friendly_message():
    info = sam.status()
    assert info["available"] is False
    assert "sam2" in info["reason"].lower()


def test_status_when_loaded(monkeypatch):
    monkeypatch.setattr(sam, "_check_available", lambda: None)
    monkeypatch.setattr(sam, "_device", lambda: "cpu")
    sam._predictor = object()
    sam._loaded_model = "facebook/sam2.1-hiera-tiny"
    sam._touch()
    info = sam.status()
    assert info["available"] is True
    assert info["predictor_loaded"] is True
    assert info["model"] == "facebook/sam2.1-hiera-tiny"
    assert info["idle_s"] is not None
    sam._predictor = None
    sam._loaded_model = None
    sam._last_used = None


def test_unload_clears_state():
    sam._predictor = object()
    sam._loaded_model = "x"
    sam._touch()
    result = sam.unload()
    assert result["unloaded"] is True
    assert sam._predictor is None
    assert sam._loaded_model is None
    assert sam._last_used is None
