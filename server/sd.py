"""Stable Diffusion integration (optional).

Lazy: pipelines are loaded on first use, cached in process, and reused. If the
``[sd]`` extra is not installed we return a friendly error rather than
crashing the server import.

Models live in HuggingFace's default cache (``~/.cache/huggingface``) and are
addressed by repo id (e.g. ``runwayml/stable-diffusion-v1-5``,
``stabilityai/sdxl-turbo``). The first call to each pipeline downloads weights
— up to several GB. Subsequent calls are cached.

Memory: SD pipelines are heavy. The cache below holds at most one pipeline of
each type (txt2img / img2img / inpaint) at a time, keyed by repo id; switching
models releases the previous one.

For batch workflows where every millisecond of latency matters, call
``load_pipeline`` ahead of time to pre-warm the cache so the first inference
call doesn't pay the model-load cost (which can be tens of seconds).
"""
from __future__ import annotations

import gc
import threading
import time
from typing import Any

from PIL import Image

_VALID_KINDS = ("txt2img", "img2img", "inpaint")

# Re-entrant so methods that take the lock can call helpers that also take it
# (notably ``_touch`` from inside ``_get_pipe``'s critical section).
_lock = threading.RLock()
# Cache key is a 4-tuple ``(kind, repo, gguf_path_or_None, controlnet_repo_or_None)``
# Tuple keys avoid the string-split bug from ``:``-delimited cache ids on
# Windows where ``gguf_path`` is a local path like ``B:\models\flux.gguf``.
PipeKey = tuple[str, str, str | None, str | None]
_cache: dict[PipeKey, Any] = {}

# ControlNet state — at most one ControlNet loaded at a time. When set,
# ``_get_pipe`` returns a ControlNet-aware pipeline; inference functions
# accept ``control_image`` + ``controlnet_conditioning_scale``.
_controlnet: Any = None
_controlnet_repo: str | None = None
_controlnet_family: str | None = None  # "sd15" / "sdxl" / "sd3" / "flux"

# Idle-eviction state: every cached pipeline records its last-used monotonic
# timestamp. A daemon sweeper thread (started on first load) wakes every
# ``_SWEEP_INTERVAL_S`` seconds and unloads entries idle longer than
# ``_idle_timeout_s``. Set ``_idle_timeout_s`` to 0 (via ``set_idle_timeout``)
# to disable auto-eviction.
_last_used: dict[PipeKey, float] = {}
_idle_timeout_s: float = 3600.0  # 1 hour
_SWEEP_INTERVAL_S: float = 300.0  # 5 minutes
_sweeper_thread: threading.Thread | None = None


def _touch(key: PipeKey) -> None:
    """Mark a cached pipeline as recently used. Takes the lock because
    ``_sweep_once`` and ``sd_status`` iterate ``_last_used`` and a concurrent
    write here would otherwise risk a dict-mutated-during-iteration error."""
    with _lock:
        _last_used[key] = time.monotonic()


def _start_sweeper_if_needed() -> None:
    """Start the idle-eviction daemon thread once, lazily on first load."""
    global _sweeper_thread
    if _sweeper_thread is not None and _sweeper_thread.is_alive():
        return
    _sweeper_thread = threading.Thread(
        target=_sweep_loop, daemon=True, name="sd-idle-sweeper",
    )
    _sweeper_thread.start()


def _sweep_loop() -> None:
    """Background loop: every interval, evict pipelines idle past the limit."""
    while True:
        time.sleep(_SWEEP_INTERVAL_S)
        try:
            _sweep_once()
        except Exception:
            # Never let the daemon die on a transient error; the sweep retries
            # next tick. We deliberately don't log here because this thread
            # runs outside the MCP request context.
            pass


def _sweep_once() -> None:
    if _idle_timeout_s <= 0:
        return
    now = time.monotonic()
    with _lock:
        to_evict = [
            k for k, t in _last_used.items()
            if k in _cache and (now - t) > _idle_timeout_s
        ]
        for k in to_evict:
            _cache.pop(k, None)
            _last_used.pop(k, None)
    if to_evict:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def set_idle_timeout(seconds: float) -> dict[str, Any]:
    """Set how long an unused SD pipeline stays resident before the sweeper
    unloads it. Pass 0 to disable auto-eviction entirely (manual unload only)."""
    global _idle_timeout_s
    _idle_timeout_s = max(0.0, float(seconds))
    return {
        "idle_timeout_s": _idle_timeout_s,
        "sweep_interval_s": _SWEEP_INTERVAL_S,
        "auto_evict_enabled": _idle_timeout_s > 0,
    }


def _check_available() -> None:
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Stable Diffusion is unavailable. Install the optional extra: "
            "`pip install image-tools-mcp[sd]`. This pulls diffusers, "
            "transformers, accelerate, and torch (large)."
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

    return torch.float16 if _device() != "cpu" else torch.float32


def _detect_family(repo: str) -> str:
    """Tag a model repo with the architecture family we use for picking the
    right ControlNet pipeline class. Returns one of sd15/sdxl/sd3/flux.

    Token-boundary aware so e.g. ``MixL/something`` doesn't match ``xl`` and
    ``stable-diffusion-3.5`` matches ``sd3`` cleanly.
    """
    import re
    r = repo.lower()
    if re.search(r"(?:^|[/_\-.])flux", r):
        return "flux"
    if re.search(r"(?:^|[/_\-.])(sd3|stable-diffusion-3)", r):
        return "sd3"
    if re.search(r"(?:^|[/_\-.])(sdxl|xl-(?:base|refiner|turbo|1\.0|sdxl))", r):
        return "sdxl"
    # Fall back to ``sd15`` for vanilla SD 1.x/2.x — there's no reliable
    # marker so anything unrecognised lands here.
    return "sd15"


def _controlnet_pipeline_class(family: str, kind: str):
    """Pick the diffusers ControlNet pipeline class for the (family, kind) pair."""
    from diffusers import (
        StableDiffusionControlNetPipeline,
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLControlNetImg2ImgPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusion3ControlNetPipeline,
        StableDiffusion3ControlNetInpaintingPipeline,
        FluxControlNetPipeline,
        FluxControlNetImg2ImgPipeline,
        FluxControlNetInpaintPipeline,
    )
    table = {
        ("sd15", "txt2img"): StableDiffusionControlNetPipeline,
        ("sd15", "img2img"): StableDiffusionControlNetImg2ImgPipeline,
        ("sd15", "inpaint"): StableDiffusionControlNetInpaintPipeline,
        ("sdxl", "txt2img"): StableDiffusionXLControlNetPipeline,
        ("sdxl", "img2img"): StableDiffusionXLControlNetImg2ImgPipeline,
        ("sdxl", "inpaint"): StableDiffusionXLControlNetInpaintPipeline,
        ("sd3", "txt2img"): StableDiffusion3ControlNetPipeline,
        ("sd3", "inpaint"): StableDiffusion3ControlNetInpaintingPipeline,
        ("flux", "txt2img"): FluxControlNetPipeline,
        ("flux", "img2img"): FluxControlNetImg2ImgPipeline,
        ("flux", "inpaint"): FluxControlNetInpaintPipeline,
    }
    cls = table.get((family, kind))
    if cls is None:
        raise RuntimeError(
            f"No diffusers ControlNet pipeline available for "
            f"family={family!r} kind={kind!r}. Available combinations: "
            f"{sorted({(f, k) for (f, k) in table})}"
        )
    return cls


def _cached_repo_for_kind(kind: str) -> tuple[str | None, str | None]:
    """Return ``(repo, gguf_path)`` of the most-recently-touched cached
    pipeline of this kind, or ``(None, None)`` if nothing is cached. Used
    by the inference functions to honour a pre-load when the caller omits
    the model arg.
    """
    with _lock:
        keys = [k for k in _cache if k[0] == kind]
        if not keys:
            return (None, None)
        keys.sort(key=lambda k: _last_used.get(k, 0), reverse=True)
        _, repo, gguf, _ = keys[0]
        return (repo, gguf)


def _get_pipe(kind: str, repo: str, *, gguf_path: str | None = None):
    """Return cached pipeline, loading on first use.

    Cache key is the tuple ``(kind, repo, gguf_path_or_None, controlnet_repo_or_None)``.
    Loading or unloading a ControlNet evicts any pipelines bound to it.

    ``gguf_path`` (if given) swaps in a GGUF-quantized UNet/transformer —
    best supported for FLUX and SD3-class models; older SD 1.x/2.x may not
    work. ControlNet + GGUF base is not currently supported (raises) — the
    diffusers ControlNet pipelines don't accept the GGUF transformer kwargs
    we use elsewhere.
    """
    cn_repo = _controlnet_repo
    key: PipeKey = (kind, repo, gguf_path, cn_repo)
    with _lock:
        if key in _cache:
            _touch(key)
            return _cache[key]
        # Validate BEFORE evicting anything — a guard failure must not
        # destroy a working cached pipeline.
        if _controlnet is not None and gguf_path:
            raise RuntimeError(
                "ControlNet + GGUF base is not currently supported. "
                "Unload the ControlNet (sd_unload_controlnet) or omit "
                "gguf_path to use the bf16 base."
            )
        if _controlnet is not None:
            family = _detect_family(repo)
            if family != _controlnet_family:
                raise RuntimeError(
                    f"ControlNet was loaded for a {_controlnet_family!r}-family "
                    f"base, but you're loading a {family!r} pipeline "
                    f"({repo!r}). Unload the ControlNet first or pick a "
                    "matching base model."
                )

        # Only now: evict any other pipeline of the same kind to keep one
        # model resident.
        for k in list(_cache):
            if k[0] == kind:
                del _cache[k]
                _last_used.pop(k, None)
        gc.collect()

        if _controlnet is not None:
            family = _detect_family(repo)
            cls = _controlnet_pipeline_class(family, kind)
            pipe = cls.from_pretrained(
                repo, controlnet=_controlnet, torch_dtype=_dtype(),
            )
            pipe = pipe.to(_device())
        else:
            from diffusers import (
                AutoPipelineForText2Image,
                AutoPipelineForImage2Image,
                AutoPipelineForInpainting,
            )
            loader = {
                "txt2img": AutoPipelineForText2Image,
                "img2img": AutoPipelineForImage2Image,
                "inpaint": AutoPipelineForInpainting,
            }[kind]
            if gguf_path:
                pipe = _load_with_gguf(repo, gguf_path, loader)
                # Mirror the qwen.py fix: a whole-pipeline ``.to(cuda)`` routes
                # through ``GGMLTensor.__torch_function__`` and access-violates on
                # Windows. The GGUF transformer is already on CUDA (placed by
                # ``_load_with_gguf``); push only the remaining smaller components.
                if _device() == "cuda":
                    for comp_name in ("vae", "text_encoder", "text_encoder_2",
                                      "text_encoder_3"):
                        comp = getattr(pipe, comp_name, None)
                        if comp is not None and hasattr(comp, "to"):
                            try:
                                comp.to("cuda")
                            except Exception:
                                pass
            else:
                pipe = loader.from_pretrained(repo, torch_dtype=_dtype())
                pipe = pipe.to(_device())
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        _cache[key] = pipe
        _touch(key)
    _start_sweeper_if_needed()
    return pipe


def _load_with_gguf(repo: str, gguf_path: str, loader):
    """Build a pipeline whose transformer/UNet comes from a GGUF file. We
    auto-pick the right model class based on the base repo name — FLUX and
    SD3 use transformer-based backbones, classic SD 1.x/2.x uses UNet."""
    from . import gguf_io

    quant_cfg = gguf_io.gguf_quantization_config(
        "bfloat16" if _device() == "cuda" else "float32",
    )
    repo_lower = repo.lower()

    def _to_cuda(t):
        # The pipeline-wide ``.to()`` is unsafe on a GGUF transformer (see
        # qwen.py for the access-violation we saw). Place the transformer
        # explicitly here so the caller can leave it alone.
        if _device() == "cuda":
            try:
                t = t.to("cuda")
                gc.collect()
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        return t

    if "flux" in repo_lower:
        from diffusers import FluxTransformer2DModel
        transformer = FluxTransformer2DModel.from_single_file(
            gguf_path, quantization_config=quant_cfg, torch_dtype=_dtype(),
        )
        transformer = _to_cuda(transformer)
        return loader.from_pretrained(
            repo, transformer=transformer, torch_dtype=_dtype(),
        )
    if "sd3" in repo_lower or "stable-diffusion-3" in repo_lower:
        from diffusers import SD3Transformer2DModel
        transformer = SD3Transformer2DModel.from_single_file(
            gguf_path, quantization_config=quant_cfg, torch_dtype=_dtype(),
        )
        transformer = _to_cuda(transformer)
        return loader.from_pretrained(
            repo, transformer=transformer, torch_dtype=_dtype(),
        )
    # Older SD 1.x/2.x — GGUF support is community-maintained and varies by
    # model. Surface a clear error so the caller can pick a different
    # quantization path (e.g. bnb 4-bit).
    raise RuntimeError(
        f"GGUF quantization for {repo!r} is not supported by the built-in "
        "loader. Use FLUX or SD3 (their transformer-based backbones load "
        "cleanly from GGUF), or omit gguf_path to run at fp16/bf16."
    )


# ---- ControlNet management ------------------------------------------------

def load_controlnet(repo: str, *, family: str | None = None) -> dict[str, Any]:
    """Load a ControlNet model into memory. Future ``sd_generate`` /
    ``sd_img2img`` / ``sd_inpaint`` calls will run ControlNet-aware
    pipelines that accept a ``control_image`` + ``conditioning_scale``.

    ``repo`` is a HuggingFace ControlNet repo id, e.g.:
        - SD1.5 canny:  ``lllyasviel/sd-controlnet-canny``
        - SD1.5 depth:  ``lllyasviel/sd-controlnet-depth``
        - SDXL canny:   ``diffusers/controlnet-canny-sdxl-1.0``
        - SDXL depth:   ``diffusers/controlnet-depth-sdxl-1.0``
        - FLUX canny:   ``InstantX/FLUX.1-dev-Controlnet-Canny``

    ``family`` is one of ``sd15`` / ``sdxl`` / ``sd3`` / ``flux`` and must
    match the base model you'll use at inference. Omit to auto-detect from
    the repo id.

    Loading a different ControlNet evicts any cached SD pipelines (each
    base+ControlNet combo is its own cache key). Use ``sd_unload_controlnet``
    to revert to plain SD inference.
    """
    _check_available()
    global _controlnet, _controlnet_repo, _controlnet_family
    fam = family or _detect_family(repo)
    if fam not in ("sd15", "sdxl", "sd3", "flux"):
        raise ValueError(
            f"family must be one of sd15/sdxl/sd3/flux, got {fam!r}"
        )
    # Pick the ControlNetModel class to match the family. SD3 + FLUX use
    # bespoke models; SD1.5 + SDXL share the same ControlNetModel class.
    if fam == "flux":
        from diffusers import FluxControlNetModel
        cn_cls = FluxControlNetModel
    elif fam == "sd3":
        from diffusers import SD3ControlNetModel
        cn_cls = SD3ControlNetModel
    else:
        from diffusers import ControlNetModel
        cn_cls = ControlNetModel
    with _lock:
        # Unloading the previous ControlNet also evicts cached pipelines
        # bound to it (they keep a reference to the old model object).
        # Clear ALL three globals up-front so a failed ``from_pretrained``
        # below leaves a clean "no ControlNet loaded" state rather than a
        # half-populated one (repo/family set but model None).
        if _controlnet is not None:
            for k in list(_cache):
                if k[3] is not None:
                    _cache.pop(k, None)
                    _last_used.pop(k, None)
            _controlnet = None
            _controlnet_repo = None
            _controlnet_family = None
            gc.collect()
        t0 = time.perf_counter()
        cn = cn_cls.from_pretrained(repo, torch_dtype=_dtype())
        if _device() == "cuda":
            try:
                cn = cn.to("cuda")
            except Exception:
                pass
        # Atomically publish all three on success.
        _controlnet = cn
        _controlnet_repo = repo
        _controlnet_family = fam
        elapsed = time.perf_counter() - t0
    return {
        "loaded": repo,
        "family": fam,
        "device": _device(),
        "load_time_s": round(elapsed, 3),
    }


def unload_controlnet() -> dict[str, Any]:
    """Detach the loaded ControlNet. Subsequent inference calls fall back
    to plain SD pipelines. Also evicts any cached ControlNet-bound
    pipelines, since they pinned the old model.
    """
    global _controlnet, _controlnet_repo, _controlnet_family
    with _lock:
        had = _controlnet is not None
        prev = _controlnet_repo
        _controlnet = None
        _controlnet_repo = None
        _controlnet_family = None
        for k in list(_cache):
            if k[3] is not None:
                _cache.pop(k, None)
                _last_used.pop(k, None)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"unloaded": had, "previous_repo": prev}


def controlnet_status() -> dict[str, Any]:
    with _lock:
        return {
            "loaded": _controlnet is not None,
            "repo": _controlnet_repo,
            "family": _controlnet_family,
            "device": _device(),
        }


def sd_status() -> dict[str, Any]:
    """Report whether SD can run, what's currently loaded, and how long each
    pipeline has been idle."""
    try:
        _check_available()
    except RuntimeError as e:
        return {"available": False, "reason": str(e)}
    now = time.monotonic()
    with _lock:
        # Snapshot inside the lock so iteration can't race with _touch / sweep.
        snapshot = [(k, _last_used.get(k, now)) for k in _cache]
        cn = {
            "loaded": _controlnet is not None,
            "repo": _controlnet_repo,
            "family": _controlnet_family,
        }
    pipelines = [
        {
            "kind": k[0], "model": k[1], "gguf_path": k[2],
            "controlnet": k[3], "idle_s": round(now - last, 1),
        }
        for k, last in snapshot
    ]
    return {
        "available": True,
        "device": _device(),
        "loaded": pipelines,
        "controlnet": cn,
        "idle_timeout_s": _idle_timeout_s,
        "auto_evict_enabled": _idle_timeout_s > 0,
    }


def load_pipeline(kind: str, model: str, *,
                  gguf_path: str | None = None) -> dict[str, Any]:
    """Pre-load a pipeline into the cache so subsequent inference calls don't
    pay the model-load cost. Idempotent: re-loading an already-cached
    ``(kind, model, gguf_path)`` is a no-op that returns
    ``was_already_loaded: True``.

    Loading a NEW model of the same kind evicts the previous one (one
    pipeline of each kind resident at a time).

    ``gguf_path`` swaps in a GGUF-quantized UNet/transformer for the heavy
    component. Best supported for FLUX (e.g. ``black-forest-labs/FLUX.1-dev``)
    and SD3 models — older SD 1.x/2.x will raise a clear error. Drops VRAM
    from ~12 GB to ~3-6 GB depending on quant level.
    """
    _check_available()
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"kind must be one of {list(_VALID_KINDS)}, got {kind!r}"
        )
    key: PipeKey = (kind, model, gguf_path, _controlnet_repo)
    was_loaded = key in _cache
    t0 = time.perf_counter()
    _get_pipe(kind, model, gguf_path=gguf_path)
    elapsed = time.perf_counter() - t0
    return {
        "kind": kind,
        "model": model,
        "gguf_path": gguf_path,
        "controlnet": _controlnet_repo,
        "device": _device(),
        "was_already_loaded": was_loaded,
        "load_time_s": round(elapsed, 3),
        "loaded": [
            {"kind": k[0], "model": k[1], "gguf_path": k[2], "controlnet": k[3]}
            for k in _cache
        ],
    }


def sd_unload(kind: str | None = None) -> dict[str, Any]:
    """Free pipelines. Pass ``kind`` (txt2img/img2img/inpaint) to drop just
    that one, or ``None`` to drop everything. Does NOT touch the loaded
    ControlNet — use ``unload_controlnet`` for that."""
    with _lock:
        if kind is None:
            n = len(_cache)
            _cache.clear()
            _last_used.clear()
        else:
            removed = [k for k in _cache if k[0] == kind]
            n = len(removed)
            for k in removed:
                del _cache[k]
                _last_used.pop(k, None)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"unloaded": n}


def _maybe_controlnet_kwargs(
    kind: str,
    control_image: Image.Image | None,
    conditioning_scale: float | None,
) -> dict[str, Any]:
    """Build the ControlNet-specific call kwargs for a (kind, family) pair.

    The kwarg name depends on both:
      - ``txt2img`` + sd15/sdxl: the conditioning *is* the image → ``image=``
      - ``img2img`` / ``inpaint`` + any: init goes as ``image=``, conditioning
        as ``control_image=``
      - any kind + flux/sd3: conditioning always as ``control_image=``
    """
    out: dict[str, Any] = {}
    if control_image is None:
        if _controlnet is not None:
            raise ValueError(
                "A ControlNet is loaded but no control_image was supplied. "
                "Either provide control_image or unload the ControlNet."
            )
        return out
    if _controlnet is None:
        raise RuntimeError(
            "control_image was supplied but no ControlNet is loaded. "
            "Call sd_load_controlnet(repo=...) first."
        )
    cond = control_image.convert("RGB")
    family = _controlnet_family or "sd15"
    if kind == "txt2img" and family in ("sd15", "sdxl"):
        out["image"] = cond
    else:
        out["control_image"] = cond
    if conditioning_scale is not None:
        out["controlnet_conditioning_scale"] = float(conditioning_scale)
    return out


def sd_txt2img(prompt: str, *, negative_prompt: str | None = None,
               width: int = 512, height: int = 512, steps: int = 25,
               guidance: float = 7.5, seed: int | None = None,
               model: str | None = None,
               gguf_path: str | None = None,
               control_image: Image.Image | None = None,
               controlnet_conditioning_scale: float | None = None,
               ) -> Image.Image:
    _check_available()
    if model is None:
        cached_repo, cached_gguf = _cached_repo_for_kind("txt2img")
        model = cached_repo or "runwayml/stable-diffusion-v1-5"
        if gguf_path is None:
            gguf_path = cached_gguf
    pipe = _get_pipe("txt2img", model, gguf_path=gguf_path)
    import torch

    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device()).manual_seed(int(seed))
    extra = _maybe_controlnet_kwargs(
        "txt2img", control_image, controlnet_conditioning_scale,
    )
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=int(width), height=int(height),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
        **extra,
    )
    _touch(("txt2img", model, gguf_path, _controlnet_repo))
    return result.images[0]


def sd_img2img(init_image: Image.Image, prompt: str, *,
               strength: float = 0.6, negative_prompt: str | None = None,
               steps: int = 25, guidance: float = 7.5,
               seed: int | None = None,
               model: str | None = None,
               gguf_path: str | None = None,
               control_image: Image.Image | None = None,
               controlnet_conditioning_scale: float | None = None,
               ) -> Image.Image:
    _check_available()
    if model is None:
        cached_repo, cached_gguf = _cached_repo_for_kind("img2img")
        model = cached_repo or "runwayml/stable-diffusion-v1-5"
        if gguf_path is None:
            gguf_path = cached_gguf
    pipe = _get_pipe("img2img", model, gguf_path=gguf_path)
    import torch

    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device()).manual_seed(int(seed))
    init = init_image.convert("RGB")
    extra = _maybe_controlnet_kwargs(
        "img2img", control_image, controlnet_conditioning_scale,
    )
    # img2img + ControlNet: ``image=`` is the init (passed below), and
    # ``control_image=`` is the conditioning (in ``extra``). No conflict.
    result = pipe(
        prompt=prompt,
        image=init,
        strength=float(strength),
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
        **extra,
    )
    _touch(("img2img", model, gguf_path, _controlnet_repo))
    return result.images[0]


def sd_inpaint(init_image: Image.Image, mask_image: Image.Image,
               prompt: str, *, negative_prompt: str | None = None,
               steps: int = 25, guidance: float = 7.5,
               seed: int | None = None,
               model: str | None = None,
               gguf_path: str | None = None,
               control_image: Image.Image | None = None,
               controlnet_conditioning_scale: float | None = None,
               ) -> Image.Image:
    _check_available()
    if model is None:
        cached_repo, cached_gguf = _cached_repo_for_kind("inpaint")
        model = cached_repo or "runwayml/stable-diffusion-inpainting"
        if gguf_path is None:
            gguf_path = cached_gguf
    pipe = _get_pipe("inpaint", model, gguf_path=gguf_path)
    import torch

    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device()).manual_seed(int(seed))
    extra = _maybe_controlnet_kwargs(
        "inpaint", control_image, controlnet_conditioning_scale,
    )
    result = pipe(
        prompt=prompt,
        image=init_image.convert("RGB"),
        mask_image=mask_image.convert("L"),
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
        **extra,
    )
    _touch(("inpaint", model, gguf_path, _controlnet_repo))
    return result.images[0]
