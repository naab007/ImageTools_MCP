"""Concurrency tests for the SD/Qwen sweeper + touch paths.

These exercise the cache-state machinery WITHOUT actually loading any model
(the dependencies aren't installed in CI). We poke ``sd._cache`` /
``sd._last_used`` directly and run the sweep/touch operations under thread
contention to catch dict-mutated-during-iteration and TOCTOU races.
"""
import threading
import time

import pytest

from server import qwen, sd


# ---- sd sweeper ------------------------------------------------------------

def test_sd_touch_during_sweep_doesnt_race(monkeypatch):
    """Regression for the iter/mutate race: a flurry of _touch calls while
    _sweep_once iterates the dict must not raise."""
    sd._cache.clear()
    sd._last_used.clear()
    monkeypatch.setattr(sd, "_idle_timeout_s", 9999)  # don't evict mid-test
    # Seed many entries so iteration takes a non-zero amount of time
    for i in range(200):
        key = f"txt2img:model-{i}"
        sd._cache[key] = object()
        sd._last_used[key] = time.monotonic()

    stop = threading.Event()
    errors: list[BaseException] = []

    def hammer_touch():
        i = 0
        while not stop.is_set():
            try:
                sd._touch(f"txt2img:model-{i % 200}")
            except BaseException as e:
                errors.append(e)
                return
            i += 1

    def hammer_sweep():
        while not stop.is_set():
            try:
                sd._sweep_once()
            except BaseException as e:
                errors.append(e)
                return

    threads = [threading.Thread(target=hammer_touch) for _ in range(3)]
    threads += [threading.Thread(target=hammer_sweep) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert not errors, f"races detected: {errors[:3]}"


def test_sd_status_snapshots_under_lock(monkeypatch):
    """sd_status must not raise during concurrent _touch calls — it now
    snapshots the cache+timestamps inside the lock before iterating."""
    sd._cache.clear()
    sd._last_used.clear()
    monkeypatch.setattr(sd, "_check_available", lambda: None)
    monkeypatch.setattr(sd, "_device", lambda: "cpu")
    for i in range(100):
        key = f"txt2img:model-{i}"
        sd._cache[key] = object()
        sd._last_used[key] = time.monotonic()

    stop = threading.Event()
    errors: list[BaseException] = []

    def hammer_touch():
        i = 0
        while not stop.is_set():
            sd._touch(f"txt2img:model-{i % 100}")
            i += 1

    def hammer_status():
        while not stop.is_set():
            try:
                result = sd.sd_status()
                assert "loaded" in result
            except BaseException as e:
                errors.append(e)
                return

    threads = [threading.Thread(target=hammer_touch) for _ in range(2)]
    threads += [threading.Thread(target=hammer_status) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert not errors, f"status race: {errors[:3]}"


def test_sd_set_idle_timeout_clamps_negative():
    info = sd.set_idle_timeout(-10)
    assert info["idle_timeout_s"] == 0.0
    assert info["auto_evict_enabled"] is False
    sd.set_idle_timeout(3600)  # restore


def test_sd_set_idle_timeout_zero_disables():
    info = sd.set_idle_timeout(0)
    assert info["auto_evict_enabled"] is False
    sd.set_idle_timeout(3600)


def test_sd_sweep_evicts_stale_entries(monkeypatch):
    """Direct test that _sweep_once removes entries older than the timeout."""
    sd._cache.clear()
    sd._last_used.clear()
    monkeypatch.setattr(sd, "_idle_timeout_s", 0.1)
    # One fresh, one stale
    sd._cache["txt2img:fresh"] = object()
    sd._last_used["txt2img:fresh"] = time.monotonic()
    sd._cache["txt2img:stale"] = object()
    sd._last_used["txt2img:stale"] = time.monotonic() - 10  # 10s ago

    sd._sweep_once()

    assert "txt2img:fresh" in sd._cache
    assert "txt2img:stale" not in sd._cache
    assert "txt2img:stale" not in sd._last_used
    sd._cache.clear()
    sd._last_used.clear()


# ---- qwen sweeper ----------------------------------------------------------

def test_qwen_sweep_skips_when_recently_touched(monkeypatch):
    """Regression for the TOCTOU race: a _touch right before the sweep's
    timestamp check should keep the pipeline alive."""
    monkeypatch.setattr(qwen, "_idle_timeout_s", 0.1)
    # Simulate a loaded pipeline
    qwen._pipe = object()
    qwen._pipe_model = "fake"
    qwen._touch()  # last_used = now

    qwen._sweep_once()
    assert qwen._pipe is not None, "fresh pipeline should not be evicted"

    # Make it stale
    qwen._last_used = time.monotonic() - 10
    qwen._sweep_once()
    assert qwen._pipe is None, "stale pipeline should be evicted"


def test_qwen_set_idle_timeout_clamps_negative():
    info = qwen.set_idle_timeout(-5)
    assert info["idle_timeout_s"] == 0.0
    qwen.set_idle_timeout(3600)


def test_qwen_status_reports_idle_when_loaded(monkeypatch):
    monkeypatch.setattr(qwen, "_check_available", lambda: None)
    monkeypatch.setattr(qwen, "_device", lambda: "cpu")
    qwen._pipe = object()
    qwen._pipe_model = "fake"
    qwen._touch()
    info = qwen.status()
    assert info["available"] is True
    assert info["pipeline_loaded"] is True
    assert info["idle_s"] is not None
    assert info["idle_s"] >= 0
    qwen._pipe = None
    qwen._pipe_model = None
    qwen._last_used = None


def test_qwen_status_reports_no_idle_when_unloaded(monkeypatch):
    monkeypatch.setattr(qwen, "_check_available", lambda: None)
    monkeypatch.setattr(qwen, "_device", lambda: "cpu")
    qwen._pipe = None
    qwen._last_used = None
    info = qwen.status()
    assert info["pipeline_loaded"] is False
    assert info["idle_s"] is None


# ---- multi-image validation ------------------------------------------------

def test_qwen_edit_rejects_empty_list(monkeypatch):
    """edit_image with [] should raise ValueError, not crash deep inside PIL."""
    monkeypatch.setattr(qwen, "_check_available", lambda: None)
    monkeypatch.setattr(qwen, "_get_pipe", lambda model=None: object())
    with pytest.raises(ValueError, match="at least one input image"):
        qwen.edit_image([], "prompt")


def test_qwen_edit_rejects_non_image_entries(monkeypatch):
    """edit_image with [None, ...] should raise TypeError with a helpful
    message, not AttributeError deep in the convert call."""
    from PIL import Image as PILImage

    monkeypatch.setattr(qwen, "_check_available", lambda: None)
    monkeypatch.setattr(qwen, "_get_pipe", lambda model=None: object())

    good = PILImage.new("RGB", (4, 4), "red")
    with pytest.raises(TypeError, match="must be a PIL Image"):
        qwen.edit_image([good, None, "hello"], "prompt")
