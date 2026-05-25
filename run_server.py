"""Launcher for claude-code registration.

Claude Code spawns MCP servers without a working directory, so module imports
relative to the project root fail. This shim adds its own directory to
``sys.path`` before importing the server entry point.

The heavy AI imports (torch / diffusers / transformers / scipy) are pre-warmed
in a **background thread** so the FastMCP server responds to MCP ``initialize``
immediately. The original blocking approach timed out the MCP client (~30 s
connection timeout vs ~60 s import cost). The daemon thread completes in the
background; by the time the first heavy tool call arrives (e.g. ``qwen_load``),
the imports are cached.

NOTE: the prewarm MUST run on the main thread for the ``scipy.linalg.blas``
DLL-deadlock workaround to work (see prior troubleshooting notes). However,
the MCP client timeout forces our hand. As a compromise, we import the
LIGHTWEIGHT modules (torch, transformers, diffusers top-level) on the main
thread (~5 s), then defer the heavy sub-imports (pipeline classes that pull in
scipy) to the background thread.
"""
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# transformers v5's parallel weight materializer access-violates on Windows;
# fall back to the single-threaded loader.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")


def _prewarm_light() -> None:
    """Quick main-thread imports — just the top-level modules. These are
    fast (~5 s) and prime the module cache so the background thread's
    deeper imports don't hit the DLL-loader deadlock."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import diffusers  # noqa: F401
    except Exception:
        pass


def _prewarm_heavy() -> None:
    """Background-thread imports — the pipeline classes that transitively
    pull in scipy.optimize → scipy.linalg.blas (the slow DLL load).
    Running these on a background thread after the light imports have
    primed the module cache avoids both the MCP-timeout AND the DLL
    deadlock (the DLL is already loaded by the time scipy needs it via
    the torch → MKL → BLAS chain that ``import torch`` established above).
    """
    try:
        from diffusers import (  # noqa: F401
            QwenImageEditPipeline,
            QwenImageEditPlusPipeline,
            QwenImageTransformer2DModel,
            AutoencoderKLQwenImage,
            FlowMatchEulerDiscreteScheduler,
        )
    except Exception:
        pass
    try:
        from transformers import (  # noqa: F401
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLModel,
            AutoTokenizer,
            AutoProcessor,
        )
    except Exception:
        pass


# Phase 1 — fast, main-thread: prime torch / transformers / diffusers.
_prewarm_light()

# Phase 2 — background: heavy sub-imports (pipeline classes + scipy).
threading.Thread(target=_prewarm_heavy, daemon=True, name="prewarm").start()

from server.image_tools_server import main

if __name__ == "__main__":
    main()
