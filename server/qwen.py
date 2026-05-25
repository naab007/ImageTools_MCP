"""Qwen-Image-Edit pipeline with multi-LoRA support.

Qwen-Image-Edit (``Qwen/Qwen-Image-Edit`` on HuggingFace) is a diffusion
model that edits an input image according to a text prompt — useful for
inpaint-style and instruction-driven edits without explicit masks.

This module wraps the diffusers ``QwenImageEditPipeline`` with:

- Lazy loading: the pipeline is constructed on first use and cached. Call
  ``load_pipeline`` ahead of time to pre-warm the cache for batch workflows
  where the first edit shouldn't pay the (potentially minute-scale) model
  load cost.
- Multi-LoRA support: load any number of LoRA adapters (each with a name),
  then set per-adapter weights via ``set_lora_weights``. Diffusers'
  ``set_adapters`` combines them at inference time.
- Single point of truth for adapter state — ``_loaded_loras`` mirrors what's
  attached to the pipeline so ``list`` / ``unload`` calls don't need to
  introspect the pipeline.

The ``[qwen]`` extra brings in diffusers + transformers + accelerate + torch.
Without it, every tool returns a friendly error.
"""
from __future__ import annotations

import os

# transformers v5's parallel weight materializer (`spawn_materialize` +
# ThreadPoolExecutor) segfaults on Windows when loading large safetensors
# shards (e.g. the Qwen2-VL text encoder) alongside an mmap'd GGUF. Force
# the single-threaded fallback before any transformers import.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import gc
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

# Re-entrant so helpers that take the lock can call other lock-takers.
_lock = threading.RLock()
_pipe: Any = None
_pipe_model: str | None = None
# Path to the active GGUF-quantized transformer (if any). None = full
# precision from the base repo.
_pipe_gguf: str | None = None

# ControlNet state — shares the Edit pipeline's transformer / text encoder /
# VAE / scheduler / tokenizer at inference time (only the ControlNet model
# itself is extra weight, ~1-2 GB). When ``_controlnet`` is loaded,
# ``qwen_controlnet_generate`` / ``qwen_controlnet_inpaint`` build a
# ``QwenImageControlNetPipeline`` / ``QwenImageControlNetInpaintPipeline``
# that references those shared components plus the ControlNet model.
_controlnet: Any = None
_controlnet_repo: str | None = None
_cn_pipe_t2i: Any = None  # lazily-built QwenImageControlNetPipeline
_cn_pipe_inpaint: Any = None  # lazily-built QwenImageControlNetInpaintPipeline

# Idle-eviction: the sweeper thread (started on first load) wakes every
# ``_SWEEP_INTERVAL_S`` seconds and unloads the pipeline if it's been idle
# longer than ``_idle_timeout_s``. Setting timeout to 0 disables eviction.
_last_used: float | None = None
_idle_timeout_s: float = 3600.0  # 1 hour
_SWEEP_INTERVAL_S: float = 300.0  # 5 minutes
_sweeper_thread: threading.Thread | None = None


def _touch() -> None:
    """Mark the pipeline as recently used. Takes the lock for consistency
    with the sweeper's check-then-unload sequence."""
    global _last_used
    with _lock:
        _last_used = time.monotonic()


def _start_sweeper_if_needed() -> None:
    global _sweeper_thread
    if _sweeper_thread is not None and _sweeper_thread.is_alive():
        return
    _sweeper_thread = threading.Thread(
        target=_sweep_loop, daemon=True, name="qwen-idle-sweeper",
    )
    _sweeper_thread.start()


def _sweep_loop() -> None:
    while True:
        time.sleep(_SWEEP_INTERVAL_S)
        try:
            _sweep_once()
        except Exception:
            pass


def _sweep_once() -> None:
    """Evict the pipeline atomically if and only if it's been idle past the
    limit at the moment we hold the lock. Without the lock here we could
    unload a pipeline that was just touched by a concurrent inference call."""
    if _idle_timeout_s <= 0:
        return
    with _lock:
        if _pipe is None or _last_used is None:
            return
        if time.monotonic() - _last_used <= _idle_timeout_s:
            return
        # Stale enough — re-confirmed under the lock, safe to unload.
    unload()


def set_idle_timeout(seconds: float) -> dict[str, Any]:
    """Set how long the Qwen pipeline stays resident after its last use.
    Pass 0 to disable auto-eviction entirely (manual unload only)."""
    global _idle_timeout_s
    _idle_timeout_s = max(0.0, float(seconds))
    return {
        "idle_timeout_s": _idle_timeout_s,
        "sweep_interval_s": _SWEEP_INTERVAL_S,
        "auto_evict_enabled": _idle_timeout_s > 0,
    }


@dataclass
class LoraEntry:
    name: str           # adapter handle used by diffusers
    source: str         # repo id or local path the LoRA came from
    weight: float = 1.0
    weight_name: str | None = None  # filename within multi-file LoRA repos


_loaded_loras: dict[str, LoraEntry] = {}


# ---- availability ----------------------------------------------------------

def _check_available() -> None:
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Qwen-Image-Edit is unavailable. Install the optional extra: "
            "`pip install image-tools-mcp[qwen]`. Pulls diffusers, transformers, "
            "accelerate, and torch (large download)."
        ) from e


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype():
    import torch

    return torch.bfloat16 if _device() != "cpu" else torch.float32


# ---- pipeline management ---------------------------------------------------

def _build_gguf_pipeline(model: str, gguf_path: str):
    _trace(f"_build_gguf_pipeline({model!r}, {gguf_path!r})")
    _trace("step: importing gguf_io")
    """Build a QwenImageEditPipeline whose transformer comes from a GGUF file.

    The unsloth/city96 Qwen-Image-Edit-2511 GGUFs already use diffusers-style
    tensor names (``transformer_blocks.N.attn.to_q.weight`` etc.) — they load
    through ``from_single_file`` directly once diffusers >= 0.38 + accelerate
    + the gguf package are installed. Diffusers' identity ``checkpoint_mapping_fn``
    for QwenImageTransformer2DModel passes the GGUF state dict straight through.
    """
    from . import gguf_io
    _trace("step: gguf_io ok; importing diffusers symbols")
    try:
        # Qwen-Image-Edit-2511 ships ``QwenImageEditPlusPipeline`` in its
        # model_index.json. Old (2509) checkpoints used ``QwenImageEditPipeline``.
        # We import both and pick whichever the repo declares.
        from diffusers import (  # type: ignore
            QwenImageEditPipeline, QwenImageEditPlusPipeline,
            QwenImageTransformer2DModel,
        )
        _trace("step: diffusers symbols imported")
    except ImportError as e:
        raise RuntimeError(
            "GGUF loading needs diffusers with Qwen-Image-Edit support "
            "(>= 0.38, plus the [gguf] extra). Upgrade with "
            "`pip install -U 'diffusers>=0.38' gguf accelerate`."
        ) from e

    _trace("imports ok; resolving pipe class...")
    pipe_cls = _resolve_pipe_class(
        model, QwenImageEditPipeline, QwenImageEditPlusPipeline,
    )
    _trace(f"pipe_cls = {pipe_cls.__name__}")
    # On a 24 GB consumer GPU the bf16 Qwen2.5-VL text encoder (~16 GB) plus
    # the Q4 GGUF transformer (~13 GB) exceeds VRAM and torch reports the
    # overflow as an access violation in storage.__getitem__ rather than a
    # clean OOM. Pre-load the text encoder in bnb 4-bit (~4 GB) and pass it
    # into the pipeline so neither overflows.
    return _build_pipe_with_bnb_text_encoder(model, gguf_path, pipe_cls)


def _build_pipe_with_bnb_text_encoder(model: str, gguf_path: str, pipe_cls):
    _trace("entering _build_pipe_with_bnb_text_encoder")
    from . import gguf_io
    from diffusers import QwenImageTransformer2DModel  # type: ignore
    from transformers import (
        BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration,
    )
    # Step 1 — text encoder in 4-bit nf4. Loads on cuda:0 with ~4 GB VRAM.
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=_dtype(),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    _trace("loading 4-bit text_encoder...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model, subfolder="text_encoder",
        torch_dtype=_dtype(),
        quantization_config=bnb_cfg,
        low_cpu_mem_usage=True,
        device_map="cuda:0" if _device() == "cuda" else None,
    )
    _trace("text_encoder loaded")
    if _device() == "cuda":
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    # Step 2 — Q4 GGUF transformer. Loaded after the text encoder so its
    # 13 GB mmap doesn't compete with the safetensor shards.
    compute = "bfloat16" if _device() == "cuda" else "float32"
    quant_cfg = gguf_io.gguf_quantization_config(compute)
    _trace("loading GGUF transformer...")
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_path,
        config=model,
        subfolder="transformer",
        quantization_config=quant_cfg,
        torch_dtype=_dtype(),
    )
    _trace("GGUF transformer loaded")
    if _device() == "cuda":
        import torch
        transformer = transformer.to("cuda")
        gc.collect()
        torch.cuda.empty_cache()
        _trace("transformer pushed to cuda")
    _trace("building pipeline...")
    pipe = pipe_cls.from_pretrained(
        model, transformer=transformer, text_encoder=text_encoder,
        torch_dtype=_dtype(), low_cpu_mem_usage=True,
    )
    _trace("pipeline built")
    return pipe


def _resolve_pipe_class(model: str, *candidates):
    """Look up the pipeline class declared in the repo's ``model_index.json``.

    Prefer the locally-cached copy so a slow / blocked HEAD request never
    stalls a Qwen load. A handful of well-known checkpoints are resolved
    by name to skip the lookup entirely.
    """
    # Fast path — well-known checkpoints. Qwen-Image-Edit-2511 ships
    # ``QwenImageEditPlusPipeline``; earlier 2509 / vanilla use ``QwenImageEditPipeline``.
    by_name = {c.__name__: c for c in candidates}
    if "2511" in model and "QwenImageEditPlusPipeline" in by_name:
        return by_name["QwenImageEditPlusPipeline"]

    try:
        from huggingface_hub import hf_hub_download
        import json
        # local_files_only=True first — we only need a tiny JSON, no need to
        # hit the network. Most users have already loaded the model once, so
        # this finds the cached copy. If not cached, fall through.
        try:
            path = hf_hub_download(model, "model_index.json", local_files_only=True)
        except Exception:
            path = hf_hub_download(model, "model_index.json")
        with open(path) as f:
            name = json.load(f).get("_class_name", "")
        if name in by_name:
            return by_name[name]
    except Exception:
        pass
    return candidates[0]


def _build_pipe_from_components(model: str, gguf_path: str, pipe_cls):
    """Assemble a Qwen-Image-Edit pipeline component-by-component.

    The pipeline only invokes ``self.text_encoder(... output_hidden_states=True)``
    and reads ``outputs.hidden_states[-1]``; the LM head of
    ``Qwen2_5_VLForConditionalGeneration`` is never used, so the plain
    ``Qwen2_5_VLModel`` backbone is enough. Skipping that wrapper also
    sidesteps the specific safetensor shard that access-violates inside
    transformers' ``_materialize_copy`` on Windows.

    Components are loaded text-encoder-first because the 13 GB GGUF mmap
    held during transformer loading collides with the 16 GB text-encoder
    mmap on Windows (page-file error 1455).
    """
    from diffusers import (  # type: ignore
        AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
        QwenImageTransformer2DModel,
    )
    from transformers import AutoModel, AutoTokenizer, AutoProcessor
    from . import gguf_io
    # ``AutoModel`` resolves to ``Qwen2_5_VLModel`` for the 2511 text encoder
    # AND handles the ``model.* -> language_model.*`` state-dict key
    # remapping that ``Qwen2_5_VLModel.from_pretrained`` skips. Loading the
    # full ``Qwen2_5_VLForConditionalGeneration`` access-violates inside
    # ``transformers.core_model_loading._materialize_copy`` on Windows.
    text_encoder = AutoModel.from_pretrained(
        model, subfolder="text_encoder",
        torch_dtype=_dtype(),
        low_cpu_mem_usage=True,
        device_map="cuda:0" if _device() == "cuda" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer")
    processor = AutoProcessor.from_pretrained(model, subfolder="processor")
    vae = AutoencoderKLQwenImage.from_pretrained(
        model, subfolder="vae", torch_dtype=_dtype(),
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model, subfolder="scheduler",
    )
    # Load the GGUF transformer last — its 13 GB mmap is the largest
    # virtual reservation and would crowd out the safetensor loads above.
    compute = "bfloat16" if _device() == "cuda" else "float32"
    quant_cfg = gguf_io.gguf_quantization_config(compute)
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_path,
        config=model,
        subfolder="transformer",
        quantization_config=quant_cfg,
        torch_dtype=_dtype(),
    )
    if _device() == "cuda":
        import torch
        transformer = transformer.to("cuda")
        gc.collect()
        torch.cuda.empty_cache()
    pipe = pipe_cls(
        scheduler=scheduler, vae=vae,
        text_encoder=text_encoder, tokenizer=tokenizer,
        processor=processor, transformer=transformer,
    )
    return pipe


def _get_pipe(model: str = "Qwen/Qwen-Image-Edit",
              gguf_path: str | None = None):
    """Load or return the cached Qwen-Image-Edit pipeline.

    When ``gguf_path`` is provided, the transformer is loaded from the GGUF
    file (quantized — much less VRAM) and combined with the rest of the
    pipeline from ``model``. The cache key is ``(model, gguf_path)`` so
    switching between full-precision and quantized variants doesn't trash
    the other's cache.
    """
    global _pipe, _pipe_model, _pipe_gguf
    global _cn_pipe_t2i, _cn_pipe_inpaint
    with _lock:
        if (_pipe is not None and _pipe_model == model
                and _pipe_gguf == gguf_path):
            _touch()
            return _pipe
        # Switching models or quant variants — drop the old one + adapters.
        if _pipe is not None:
            _pipe = None
            _loaded_loras.clear()
            # The CN pipelines reference the just-dropped transformer / VAE /
            # text_encoder — invalidate them too so a subsequent
            # ``controlnet_generate`` rebuilds against the new components.
            _cn_pipe_t2i = None
            _cn_pipe_inpaint = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if gguf_path:
            pipe = _build_gguf_pipeline(model, gguf_path)
            # _build_gguf_pipeline already places the GGUF transformer and the
            # 4-bit text encoder on cuda:0 — the small remaining components
            # (vae, tokenizer, processor, scheduler) are the only thing we
            # still need to push to GPU. Calling .to() on the whole pipeline
            # would route through GGMLTensor.__torch_function__ and access-
            # violate, so move just those.
            if _device() == "cuda":
                import torch
                for comp_name in ("vae",):
                    comp = getattr(pipe, comp_name, None)
                    if comp is not None and hasattr(comp, "to"):
                        comp.to("cuda")
                torch.cuda.empty_cache()
        else:
            from diffusers import QwenImageEditPipeline  # type: ignore
            pipe = QwenImageEditPipeline.from_pretrained(
                model, torch_dtype=_dtype(),
            )
            pipe = pipe.to(_device())
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        _pipe = pipe
        _pipe_model = model
        _pipe_gguf = gguf_path
        _touch()
        active_pipe = _pipe
    _start_sweeper_if_needed()
    return active_pipe


def status() -> dict[str, Any]:
    """Report whether Qwen is installed, the device, and what's loaded."""
    try:
        _check_available()
    except RuntimeError as e:
        return {"available": False, "reason": str(e)}
    # Snapshot all module globals under the lock — otherwise a concurrent
    # load/unload can flip _pipe/_pipe_model/_loaded_loras mid-read.
    with _lock:
        pipe_loaded = _pipe is not None
        last_used = _last_used
        pipe_model = _pipe_model
        pipe_gguf = _pipe_gguf
        loras_snapshot = list(_loaded_loras.values())
        cn_loaded = _controlnet is not None
        cn_repo = _controlnet_repo
    idle_s = None
    if pipe_loaded and last_used is not None:
        idle_s = round(time.monotonic() - last_used, 1)
    info: dict[str, Any] = {
        "available": True,
        "device": _device(),
        "pipeline_loaded": pipe_loaded,
        "model": pipe_model,
        "gguf_path": pipe_gguf,
        "quantization": _gguf_io_safe_detect(pipe_gguf),
        "idle_s": idle_s,
        "idle_timeout_s": _idle_timeout_s,
        "auto_evict_enabled": _idle_timeout_s > 0,
        "loras": [
            {"name": e.name, "source": e.source, "weight": e.weight,
             "weight_name": e.weight_name}
            for e in loras_snapshot
        ],
        "controlnet": {"loaded": cn_loaded, "repo": cn_repo},
    }
    return info


def _gguf_io_safe_detect(path: str | None) -> str | None:
    """Try to read the quant level from a GGUF path; return None if anything
    goes wrong (e.g. gguf_io can't be imported)."""
    if not path:
        return None
    try:
        from . import gguf_io
        return gguf_io.detect_quant_level(path)
    except Exception:
        return None


def _trace(msg: str) -> None:
    import sys, os
    line = f"[qwen.{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass
    # Also write to a known file so we can find traces when stderr is
    # redirected or buffered.
    try:
        with open(os.path.join(os.environ.get("TEMP", "."), "qwen_trace.log"),
                  "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_pipeline(model: str = "Qwen/Qwen-Image-Edit", *,
                  gguf_path: str | None = None) -> dict[str, Any]:
    """Pre-load the Qwen pipeline so subsequent edits don't pay model-load
    latency. Idempotent: re-loading the same ``(model, gguf_path)`` pair is
    a no-op that returns ``was_already_loaded: True``.

    Loading a different model OR a different GGUF quantization evicts the
    previous pipeline AND its LoRA adapters — adapters are pipeline-specific
    in diffusers, so you'll need to re-attach them after a switch.

    ``gguf_path`` accepts a local file path to a GGUF-quantized
    QwenImageTransformer2DModel checkpoint (drops VRAM from ~12 GB to ~3-6 GB
    depending on quant level). Use ``download_gguf`` to fetch one from
    HuggingFace; see :mod:`gguf_io`.
    """
    _trace(f"load_pipeline(model={model!r}, gguf_path={gguf_path!r})")
    _check_available()
    _trace("_check_available ok")
    was_loaded = (_pipe is not None and _pipe_model == model
                  and _pipe_gguf == gguf_path)
    t0 = time.perf_counter()
    _trace("calling _get_pipe ...")
    _get_pipe(model, gguf_path=gguf_path)
    elapsed = time.perf_counter() - t0
    _trace(f"_get_pipe returned in {elapsed:.1f}s")
    return {
        "model": model,
        "gguf_path": gguf_path,
        "quantization": _gguf_io_safe_detect(gguf_path),
        "device": _device(),
        "was_already_loaded": was_loaded,
        "load_time_s": round(elapsed, 3),
        "loras": _adapter_summary(),
    }


def unload() -> dict[str, Any]:
    """Drop the pipeline, all LoRA adapters, AND any ControlNet pipelines
    that share its components. Frees GPU memory."""
    global _pipe, _pipe_model, _pipe_gguf, _last_used
    global _cn_pipe_t2i, _cn_pipe_inpaint
    with _lock:
        had_pipe = _pipe is not None
        n_loras = len(_loaded_loras)
        _pipe = None
        _pipe_model = None
        _pipe_gguf = None
        _last_used = None
        _loaded_loras.clear()
        # Drop CN pipelines too — they reference the just-freed components.
        # The ControlNetModel itself (``_controlnet``) survives so the user
        # can ``qwen_load`` again and resume using it without re-downloading.
        _cn_pipe_t2i = None
        _cn_pipe_inpaint = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"unloaded_pipeline": had_pipe, "unloaded_loras": n_loras}


# ---- ControlNet management -------------------------------------------------

def load_controlnet(repo: str) -> dict[str, Any]:
    """Load a Qwen-Image ControlNet model. Future
    ``qwen_controlnet_generate`` / ``qwen_controlnet_inpaint`` calls will
    run ControlNet-aware pipelines that accept a ``control_image`` +
    ``conditioning_scale``.

    Public repos (verified 2026-05-25):
      - ``InstantX/Qwen-Image-ControlNet-Union`` (3.5 GB) — single model
        handling canny / depth / pose / soft_edge via a control type input
      - ``InstantX/Qwen-Image-ControlNet-Inpainting`` — inpaint flavor
      - ``DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny`` —
        canny-only, smaller
      - ``DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth`` —
        depth-only, smaller

    Loading a new ControlNet evicts any cached ControlNet pipelines. Use
    ``qwen_unload_controlnet`` to revert to plain Qwen-Image-Edit."""
    _check_available()
    try:
        from diffusers import QwenImageControlNetModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Qwen ControlNet needs diffusers >= 0.38. "
            "`pip install -U 'diffusers>=0.38'`."
        ) from e
    global _controlnet, _controlnet_repo, _cn_pipe_t2i, _cn_pipe_inpaint
    _trace(f"load_controlnet({repo!r})")
    with _lock:
        # Reset CN state up-front so a failed load leaves a clean "no CN"
        # state rather than a partly-populated one.
        if _controlnet is not None:
            _controlnet = None
            _controlnet_repo = None
            _cn_pipe_t2i = None
            _cn_pipe_inpaint = None
            gc.collect()
        t0 = time.perf_counter()
        cn = QwenImageControlNetModel.from_pretrained(repo, torch_dtype=_dtype())
        if _device() == "cuda":
            try:
                cn = cn.to("cuda")
            except Exception:
                pass
        _controlnet = cn
        _controlnet_repo = repo
        elapsed = time.perf_counter() - t0
    _trace(f"load_controlnet done in {elapsed:.1f}s")
    return {
        "loaded": repo,
        "device": _device(),
        "load_time_s": round(elapsed, 3),
    }


def unload_controlnet() -> dict[str, Any]:
    """Detach the loaded ControlNet. Subsequent ``qwen_controlnet_*`` calls
    will fail until another is loaded; plain ``qwen_edit_image`` is
    unaffected (it doesn't use ControlNet)."""
    global _controlnet, _controlnet_repo, _cn_pipe_t2i, _cn_pipe_inpaint
    with _lock:
        had = _controlnet is not None
        prev = _controlnet_repo
        _controlnet = None
        _controlnet_repo = None
        _cn_pipe_t2i = None
        _cn_pipe_inpaint = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"unloaded": had, "previous_repo": prev}


def controlnet_status() -> dict[str, Any]:
    """Report whether a Qwen ControlNet is loaded and which repo."""
    with _lock:
        return {
            "loaded": _controlnet is not None,
            "repo": _controlnet_repo,
            "device": _device(),
            "t2i_pipe_built": _cn_pipe_t2i is not None,
            "inpaint_pipe_built": _cn_pipe_inpaint is not None,
        }


def _patch_qwen_cn_img_shapes(transformer) -> None:
    """Defensive wrap of ``transformer.forward`` to coerce a flat ``img_shapes``
    into the nested form the transformer expects.

    diffusers 0.38's ``QwenImageControlNetPipeline`` (and inpaint variant)
    construct ``img_shapes = [(1, h, w)] * batch_size`` — a list of tuples.
    The transformer's forward iterates expecting ``[[(c,h,w), ...], ...]``
    (list of lists of tuples) and calls ``prod(sample[0])`` which fails
    with ``TypeError: 'int' object is not iterable`` on the flat form.

    The Edit pipelines build it correctly (``[[(1, h, w)]] * batch_size``),
    so the wrapper is a no-op there — it only normalises when the inner
    element is a bare tuple. Idempotent: a ``_imgshapes_patched`` flag
    prevents double-wrapping when the user toggles CN on/off repeatedly.
    """
    if getattr(transformer, "_imgshapes_patched", False):
        return
    orig_forward = transformer.forward

    def _wrapped_forward(*args, **kwargs):
        shapes = kwargs.get("img_shapes")
        if shapes:
            first = shapes[0]
            # Flat form (CN pipelines): inner item is the (c, h, w) tuple.
            # Nested form (Edit pipelines): inner item is a list/tuple OF tuples.
            if isinstance(first, tuple) and len(first) > 0 and not isinstance(first[0], (list, tuple)):
                kwargs["img_shapes"] = [[s] for s in shapes]
        return orig_forward(*args, **kwargs)

    transformer.forward = _wrapped_forward
    transformer._imgshapes_patched = True


def _get_cn_pipe(kind: str):
    """Return a Qwen ControlNet pipeline (``t2i`` or ``inpaint``), building
    it lazily on first use. Shares the Edit pipeline's transformer / text
    encoder / VAE / scheduler / tokenizer so no extra heavy weights are
    needed beyond the ControlNet model itself.
    """
    global _cn_pipe_t2i, _cn_pipe_inpaint
    with _lock:
        if _controlnet is None:
            raise RuntimeError(
                "No Qwen ControlNet loaded. Call qwen_load_controlnet(repo=...) "
                "first."
            )
        if kind == "t2i" and _cn_pipe_t2i is not None:
            _touch()
            return _cn_pipe_t2i
        if kind == "inpaint" and _cn_pipe_inpaint is not None:
            _touch()
            return _cn_pipe_inpaint
        # Need the Edit pipeline's components — ensure it's loaded.
        edit = _get_pipe(
            _pipe_model or "Qwen/Qwen-Image-Edit-2511",
            gguf_path=_pipe_gguf,
        )
        from diffusers import (  # type: ignore
            QwenImageControlNetPipeline,
            QwenImageControlNetInpaintPipeline,
        )
        cls = {
            "t2i": QwenImageControlNetPipeline,
            "inpaint": QwenImageControlNetInpaintPipeline,
        }[kind]
        pipe = cls(
            scheduler=edit.scheduler,
            vae=edit.vae,
            text_encoder=edit.text_encoder,
            tokenizer=edit.tokenizer,
            transformer=edit.transformer,
            controlnet=_controlnet,
        )
        # Apply the diffusers-0.38 ``img_shapes``-coercion patch.
        _patch_qwen_cn_img_shapes(edit.transformer)
        if kind == "t2i":
            _cn_pipe_t2i = pipe
        else:
            _cn_pipe_inpaint = pipe
        _touch()
        return pipe


def controlnet_generate(
    prompt: str, control_image: Image.Image, *,
    negative_prompt: str | None = None,
    width: int = 1024, height: int = 1024,
    steps: int = 30, guidance: float = 4.0,
    true_cfg_scale: float | None = None,
    controlnet_conditioning_scale: float = 1.0,
    seed: int | None = None,
) -> Image.Image:
    """Qwen-Image text-to-image WITH ControlNet conditioning. The
    ``control_image`` is the structural cue (canny edges, depth map, pose,
    etc.) — produce it with the ``canny_edges`` preprocessor or similar.

    ``controlnet_conditioning_scale`` ∈ 0.0-2.0 controls how strictly the
    output follows the control image (1.0 = nominal). ``steps`` 20-30 is a
    good range.

    Note this is NOT the same as ``qwen_edit_image`` — that one conditions
    on an INPUT image's content (img2img-style edit). ControlNet conditions
    on a STRUCTURE cue while the rest is text-generated.

    LoRA caveat: LoRAs attached via ``qwen_load_lora`` modify the transformer
    in-place and are shared with the ControlNet pipeline (both reference
    the same transformer). For Lightning-distilled inference, set
    ``true_cfg_scale=1.0`` and ``steps=4-8``. To run "clean" ControlNet
    without LoRA influence, ``qwen_set_lora_weights({...: 0.0})`` first."""
    _check_available()
    pipe = _get_cn_pipe("t2i")
    import torch
    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device()).manual_seed(int(seed))
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": int(width), "height": int(height),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "control_image": control_image.convert("RGB"),
        "controlnet_conditioning_scale": float(controlnet_conditioning_scale),
        "generator": generator,
    }
    if true_cfg_scale is not None:
        kwargs["true_cfg_scale"] = float(true_cfg_scale)
    with _lock:
        _touch()
    result = pipe(**kwargs)
    return result.images[0]


def controlnet_inpaint(
    prompt: str, control_image: Image.Image, control_mask: Image.Image, *,
    negative_prompt: str | None = None,
    width: int | None = None, height: int | None = None,
    steps: int = 30, guidance: float = 4.0,
    true_cfg_scale: float | None = None,
    controlnet_conditioning_scale: float = 1.0,
    seed: int | None = None,
) -> Image.Image:
    """Qwen-Image inpaint with ControlNet. ``control_mask`` is white where
    you want the model to paint (the region under control), black where
    you want it left alone."""
    _check_available()
    pipe = _get_cn_pipe("inpaint")
    import torch
    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device()).manual_seed(int(seed))
    cond = control_image.convert("RGB")
    mask = control_mask.convert("L")
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "control_image": cond,
        "control_mask": mask,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "controlnet_conditioning_scale": float(controlnet_conditioning_scale),
        "generator": generator,
    }
    if width is not None:
        kwargs["width"] = int(width)
    if height is not None:
        kwargs["height"] = int(height)
    if true_cfg_scale is not None:
        kwargs["true_cfg_scale"] = float(true_cfg_scale)
    with _lock:
        _touch()
    result = pipe(**kwargs)
    return result.images[0]


# ---- LoRA management -------------------------------------------------------

def load_lora(name: str, source: str, *, weight: float = 1.0,
              weight_name: str | None = None) -> dict[str, Any]:
    """Attach a LoRA adapter to the Qwen pipeline.

    ``name`` is the handle this LoRA is referenced by in subsequent calls
    (must be unique). ``source`` is a HuggingFace repo id (e.g.
    ``"foo/bar-style-lora"``) or a local path to a ``.safetensors`` file.
    ``weight_name`` selects a specific file when the repo contains multiple
    LoRAs in one tree.
    """
    _check_available()
    # Guard the entire LoRA mutation against concurrent eviction / model
    # switches. ``_get_pipe`` re-acquires the lock internally — RLock makes
    # that re-entry safe.
    with _lock:
        if name in _loaded_loras:
            raise ValueError(
                f"lora {name!r} already loaded. Unload it first or pick a different name."
            )
        # Use the currently cached pipeline rather than the module defaults
        # (which would silently re-build a different bf16 base model).
        pipe = _get_pipe(
            _pipe_model or "Qwen/Qwen-Image-Edit",
            gguf_path=_pipe_gguf,
        )
        kwargs: dict[str, Any] = {"adapter_name": name}
        if weight_name:
            kwargs["weight_name"] = weight_name
        pipe.load_lora_weights(source, **kwargs)
        _loaded_loras[name] = LoraEntry(
            name=name, source=source, weight=float(weight), weight_name=weight_name,
        )
        _apply_adapter_weights()
        return {"loaded": name, "active_loras": _adapter_summary()}


def set_lora_weights(weights: dict[str, float]) -> dict[str, Any]:
    """Adjust the weight of one or more loaded LoRAs. Unmentioned LoRAs keep
    their current weight; pass ``0.0`` to disable a LoRA without unloading it.

    Example: ``set_lora_weights({"style": 0.8, "character": 0.5})``.
    """
    _check_available()
    with _lock:
        if _pipe is None:
            raise RuntimeError("no Qwen pipeline loaded; call qwen_edit_image once first")
        unknown = [n for n in weights if n not in _loaded_loras]
        if unknown:
            raise ValueError(
                f"unknown lora(s): {unknown}. Loaded: {list(_loaded_loras)}"
            )
        for n, w in weights.items():
            _loaded_loras[n].weight = float(w)
        _apply_adapter_weights()
        return {"active_loras": _adapter_summary()}


def unload_lora(name: str | None = None) -> dict[str, Any]:
    """Detach one LoRA (by name) or all (omit ``name``)."""
    _check_available()
    with _lock:
        if _pipe is None:
            return {"unloaded": []}
        if name is None:
            names = list(_loaded_loras)
        else:
            if name not in _loaded_loras:
                raise ValueError(f"lora {name!r} not loaded")
            names = [name]
        for n in names:
            try:
                _pipe.delete_adapters([n])
            except Exception:
                # Older diffusers versions used delete_adapter(n) singular.
                try:
                    _pipe.delete_adapter(n)
                except Exception:
                    pass
            _loaded_loras.pop(n, None)
        if _loaded_loras:
            _apply_adapter_weights()
        else:
            try:
                _pipe.disable_lora()
            except Exception:
                pass
        return {"unloaded": names, "active_loras": _adapter_summary()}


def list_loras() -> dict[str, Any]:
    """List currently loaded LoRAs with their weights."""
    return {"loras": _adapter_summary()}


def _adapter_summary() -> list[dict[str, Any]]:
    return [
        {"name": e.name, "source": e.source, "weight": e.weight,
         "weight_name": e.weight_name}
        for e in _loaded_loras.values()
    ]


def _apply_adapter_weights() -> None:
    """Push the current ``_loaded_loras`` weights into the pipeline. Diffusers
    expects a single ``set_adapters`` call listing every active adapter; the
    pipeline combines them linearly at inference."""
    if _pipe is None or not _loaded_loras:
        return
    names = list(_loaded_loras)
    weights = [_loaded_loras[n].weight for n in names]
    _pipe.set_adapters(names, adapter_weights=weights)


# ---- inference -------------------------------------------------------------

def edit_image(init_image: Image.Image | list[Image.Image], prompt: str, *,
               negative_prompt: str | None = None,
               steps: int = 30,
               guidance: float = 4.0,
               true_cfg_scale: float | None = None,
               seed: int | None = None,
               model: str | None = None,
               gguf_path: str | None = None) -> Image.Image:
    """Run Qwen-Image-Edit on ``init_image`` with ``prompt``.

    - ``init_image`` can be a single PIL Image or a list of Images. Multi-image
      input is used by Qwen-Image-Edit-2509 and similar variants for fusion
      and reference-driven edits — the model conditions on every supplied
      image. With the base ``Qwen/Qwen-Image-Edit`` the first image is the
      primary subject.
    - ``steps`` 25-40 is a good range. Higher = slower + slightly better quality.
    - ``guidance`` 3.0-5.0 is typical; higher follows the prompt more rigidly.
    - ``true_cfg_scale`` is Qwen's true classifier-free guidance scale; pass
      a value to use it (most callers can ignore).
    - ``seed`` makes generation deterministic.

    Result is returned as an RGB Image; caller decides how to surface it
    (write to disk, put on a canvas, etc.).
    """
    # Validate inputs BEFORE touching any model machinery so bad calls fail
    # fast (and so the failure doesn't depend on torch being importable).
    images = init_image if isinstance(init_image, list) else [init_image]
    if not images:
        raise ValueError("edit_image: need at least one input image")
    bad = [i for i, img in enumerate(images) if not isinstance(img, Image.Image)]
    if bad:
        raise TypeError(
            f"edit_image: every entry of init_image must be a PIL Image; "
            f"got non-Image at index {bad}"
        )

    _check_available()
    # Default to whatever pipeline is currently cached — most callers pre-load
    # via ``load_pipeline(model=..., gguf_path=...)`` once and then call
    # ``edit_image`` repeatedly without repeating those args. Falling through
    # to ``Qwen/Qwen-Image-Edit`` here would cache-miss and silently re-
    # download the bf16 base model.
    if model is None:
        model = _pipe_model or "Qwen/Qwen-Image-Edit"
    if gguf_path is None:
        gguf_path = _pipe_gguf
    pipe = _get_pipe(model, gguf_path=gguf_path)
    import torch

    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device()).manual_seed(int(seed))

    rgb = [img.convert("RGB") for img in images]

    kwargs: dict[str, Any] = {
        # Pass a single Image when only one is supplied (broadest pipeline
        # compatibility); a list otherwise.
        "image": rgb[0] if len(rgb) == 1 else rgb,
        "prompt": prompt,
        "num_inference_steps": int(steps),
        "true_cfg_scale": float(true_cfg_scale) if true_cfg_scale is not None
                          else float(guidance),
        "generator": generator,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    result = pipe(**kwargs)
    _touch()
    return result.images[0]
