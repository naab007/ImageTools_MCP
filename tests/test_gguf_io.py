"""Tests for the GGUF helper utility — path detection, quant level sniffing,
availability checks, dtype mapping. Avoids actually downloading model files
or installing the gguf/diffusers packages."""
import pytest

from server import gguf_io


# ---- detection -------------------------------------------------------------

def test_is_gguf_path_true_for_gguf():
    assert gguf_io.is_gguf_path("foo/bar.gguf") is True
    assert gguf_io.is_gguf_path("foo/bar.GGUF") is True
    assert gguf_io.is_gguf_path("C:\\models\\flux1-Q4.gguf") is True


def test_is_gguf_path_false_for_others():
    assert gguf_io.is_gguf_path("foo/bar.safetensors") is False
    assert gguf_io.is_gguf_path("foo/bar.pt") is False
    assert gguf_io.is_gguf_path("foo/bar.bin") is False
    assert gguf_io.is_gguf_path("") is False
    assert gguf_io.is_gguf_path(None) is False


# ---- quant level sniff -----------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("qwen-image-edit-Q4_K_S.gguf", "Q4_K_S"),
    ("flux1-dev-Q8_0.gguf",         "Q8_0"),
    ("foo-Q2_K.gguf",               "Q2_K"),
    ("bar-Q5_K_M.gguf",             "Q5_K_M"),
    ("baz-BF16.gguf",               "BF16"),
    ("model-F16.gguf",              "F16"),
])
def test_detect_quant_level_known(filename, expected):
    assert gguf_io.detect_quant_level(filename) == expected


def test_detect_quant_level_unknown_returns_none():
    assert gguf_io.detect_quant_level("noquant.gguf") is None
    assert gguf_io.detect_quant_level("custom-foo.gguf") is None


def test_detect_quant_level_prefers_longer_match():
    """Q4_K_S should match before Q4_K (longer wins)."""
    assert gguf_io.detect_quant_level("model-Q4_K_S.gguf") == "Q4_K_S"
    assert gguf_io.detect_quant_level("model-Q4_K_M.gguf") == "Q4_K_M"


# ---- availability check ----------------------------------------------------

def test_check_available_raises_without_deps(monkeypatch):
    """If diffusers or gguf isn't installed, the helper surfaces a clear
    error rather than an obscure ImportError deep in the stack."""
    import sys
    # Pretend diffusers is missing — works whether or not it's actually
    # installed because monkeypatch isolates the module.
    monkeypatch.setitem(sys.modules, "diffusers", None)
    monkeypatch.setitem(sys.modules, "gguf", None)
    with pytest.raises(RuntimeError, match="GGUF support is unavailable"):
        gguf_io._check_available()


# ---- compute dtype mapping -------------------------------------------------

def test_gguf_quantization_config_unknown_dtype(monkeypatch):
    """An unknown ``compute_dtype`` should raise ValueError before reaching
    diffusers — we don't want a confusing crash inside transformers."""
    monkeypatch.setattr(gguf_io, "_check_available", lambda: None)
    # Use monkeypatch.setitem so the stubs are cleaned up after the test —
    # otherwise later tests inherit a broken torch/diffusers and break.
    import sys, types
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        bfloat16=object(), float16=object(), float32=object(),
    ))
    class _Cfg:
        def __init__(self, **kw): pass
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(
        GGUFQuantizationConfig=_Cfg,
    ))
    with pytest.raises(ValueError, match="unknown compute_dtype"):
        gguf_io.gguf_quantization_config("not-a-real-dtype")


# ---- download_gguf input validation ----------------------------------------

def test_download_gguf_rejects_non_gguf_filename(monkeypatch):
    """Catch typos before issuing a network request."""
    import sys, types
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        hf_hub_download=lambda **kw: "/tmp/fake",
    ))
    with pytest.raises(ValueError, match="should end in .gguf"):
        gguf_io.download_gguf("org/repo", "model.safetensors")
