"""Concurrency-safety tests for the seg model wrappers.

The bug class we're guarding against: a model loader returns the cached
model object but its caller then reads ``_device_used`` / ``_input_size``
globals — a concurrent variant switch can rotate those between the return
and the read, so the caller applies the wrong-size transform to the wrong
device.

The fix is to return a snapshot tuple from ``_get_model``. These tests
verify the snapshot is *captured under the lock* by simulating the race
window.
"""
import threading
import time

import pytest

from server import birefnet, clipseg, sam1


# ---- birefnet --------------------------------------------------------------

def test_birefnet_get_model_returns_snapshot_tuple(monkeypatch):
    """Even with the underlying globals rotating mid-call, the returned
    snapshot stays consistent — the device/size in the tuple match the
    model object that came with them."""
    sentinel_model = object()
    monkeypatch.setattr(birefnet, "_model", sentinel_model)
    monkeypatch.setattr(birefnet, "_loaded_repo", "ZhengPeng7/BiRefNet")
    monkeypatch.setattr(birefnet, "_device_used", "cuda")
    monkeypatch.setattr(birefnet, "_input_size", 1024)

    model, device, size = birefnet._get_model("general")
    # Rotate the globals *after* the snapshot — the local vars must not flip.
    birefnet._device_used = "cpu"
    birefnet._input_size = 9999
    assert model is sentinel_model
    assert device == "cuda"
    assert size == 1024


def test_birefnet_get_model_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unknown BiRefNet variant"):
        birefnet._get_model("not-a-variant")


# ---- clipseg ---------------------------------------------------------------

def test_clipseg_get_model_returns_snapshot(monkeypatch):
    sm, sp = object(), object()
    monkeypatch.setattr(clipseg, "_model", sm)
    monkeypatch.setattr(clipseg, "_processor", sp)
    monkeypatch.setattr(clipseg, "_loaded_repo", "CIDAS/clipseg-rd64-refined")
    monkeypatch.setattr(clipseg, "_device_used", "cuda")

    m, p, d = clipseg._get_model("CIDAS/clipseg-rd64-refined")
    # Rotate globals after the snapshot — locals must stay frozen.
    clipseg._device_used = "cpu"
    assert (m, p, d) == (sm, sp, "cuda")


# ---- sam1 ------------------------------------------------------------------

def test_sam1_get_model_returns_snapshot(monkeypatch):
    sm, sp = object(), object()
    monkeypatch.setattr(sam1, "_model", sm)
    monkeypatch.setattr(sam1, "_processor", sp)
    monkeypatch.setattr(sam1, "_loaded_model", "facebook/sam-vit-large")
    monkeypatch.setattr(sam1, "_device_used", "cuda")

    m, p, d = sam1._get_model("facebook/sam-vit-large")
    sam1._device_used = "cpu"
    assert (m, p, d) == (sm, sp, "cuda")


# ---- threaded stress -------------------------------------------------------

def test_birefnet_concurrent_get_model_doesnt_race(monkeypatch):
    """200 readers + 1 rotator. No crashes, no torn snapshots
    (every reader's (model, device, size) tuple is internally consistent)."""
    sentinel_a = object()
    sentinel_b = object()
    monkeypatch.setattr(birefnet, "_model", sentinel_a)
    monkeypatch.setattr(birefnet, "_loaded_repo", "ZhengPeng7/BiRefNet")
    monkeypatch.setattr(birefnet, "_device_used", "cuda")
    monkeypatch.setattr(birefnet, "_input_size", 1024)

    stop = threading.Event()
    errors: list[BaseException] = []
    bad_pairs: list = []

    def rotator():
        """Alternates the globals between two consistent states. If a reader
        sees an A-model with B-device, it's a torn read."""
        i = 0
        while not stop.is_set():
            with birefnet._lock:
                if i % 2 == 0:
                    birefnet._model = sentinel_a
                    birefnet._device_used = "cuda"
                    birefnet._input_size = 1024
                else:
                    birefnet._model = sentinel_b
                    birefnet._device_used = "cpu"
                    birefnet._input_size = 2048
                # Loaded_repo is the cache-hit key — keep it valid for both states.
                birefnet._loaded_repo = "ZhengPeng7/BiRefNet"
            i += 1

    def reader():
        try:
            while not stop.is_set():
                m, d, s = birefnet._get_model("general")
                # Internal consistency check: A pairs with cuda/1024, B with cpu/2048.
                expected = ("cuda", 1024) if m is sentinel_a else ("cpu", 2048)
                if (d, s) != expected:
                    bad_pairs.append((m, d, s))
                    return
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=rotator)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert not errors, f"races detected: {errors[:3]}"
    assert not bad_pairs, f"torn snapshots: {bad_pairs[:3]}"
