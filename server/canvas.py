"""In-memory canvas store with layers + bounded undo/redo.

Each canvas is a fixed-size document containing a list of ``Layer`` objects
(bottom-to-top z-order) plus an ``active_index`` cursor. Drawing/filter ops
target the active layer; canvas-wide transforms (resize/crop/rotate/flip)
apply to all layers. ``compose()`` flattens the stack into a single RGBA
image for save/preview.

Single-layer canvases — what ``new_canvas`` / ``open_canvas`` produce by
default — behave exactly like the pre-layers store: there's just one
layer and it's always active.

Undo/redo snapshots the entire layer stack (and active index), so any
mutation can be rolled back atomically.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from PIL import Image

from .layers import Layer, compose_layers

MAX_HISTORY = 32


@dataclass
class CanvasState:
    """Snapshot-able state. Stored on the undo/redo stacks."""
    layers: list[Layer]
    active_index: int


@dataclass
class CanvasEntry:
    width: int
    height: int
    layers: list[Layer]
    active_index: int = 0
    path: str | None = None
    undo_stack: list[CanvasState] = field(default_factory=list)
    redo_stack: list[CanvasState] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def active_layer(self) -> Layer:
        return self.layers[self.active_index]


class CanvasStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, CanvasEntry] = {}

    # --------------------------------------------------------------- lifecycle
    def put_image(self, image: Image.Image, *, canvas_id: str | None = None,
                  path: str | None = None, layer_name: str = "Background") -> str:
        """Create a single-layer canvas from a flat image. The original mode
        is preserved on save; layers are always stored RGBA internally."""
        rgba = image.convert("RGBA") if image.mode != "RGBA" else image.copy()
        layer = Layer(name=layer_name, image=rgba)
        return self.put_layers(layer.image.width, layer.image.height, [layer],
                               canvas_id=canvas_id, path=path)

    def put_layers(self, width: int, height: int, layers: list[Layer], *,
                   canvas_id: str | None = None, path: str | None = None) -> str:
        if not layers:
            raise ValueError("canvas must have at least one layer")
        with self._lock:
            cid = canvas_id or f"cv_{uuid.uuid4().hex[:8]}"
            self._items[cid] = CanvasEntry(
                width=int(width), height=int(height),
                layers=list(layers), active_index=len(layers) - 1, path=path,
            )
            return cid

    def entry(self, canvas_id: str) -> CanvasEntry:
        with self._lock:
            try:
                return self._items[canvas_id]
            except KeyError:
                raise KeyError(
                    f"unknown canvas_id {canvas_id!r}. Open one via "
                    f"new_canvas/open_canvas, or call list_canvases."
                )

    def close(self, canvas_id: str) -> None:
        with self._lock:
            self._items.pop(canvas_id, None)

    def list_ids(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._items.keys()))

    def summary(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "canvas_id": cid,
                    "width": e.width,
                    "height": e.height,
                    "n_layers": len(e.layers),
                    "active_index": e.active_index,
                    "path": e.path,
                    "undo_depth": len(e.undo_stack),
                    "redo_depth": len(e.redo_stack),
                }
                for cid, e in self._items.items()
            ]

    # --------------------------------------------------------------- read
    def compose(self, canvas_id: str) -> Image.Image:
        """Flatten visible layers into a single RGBA image."""
        e = self.entry(canvas_id)
        return compose_layers(e.layers, e.size)

    def active_layer(self, canvas_id: str) -> Layer:
        return self.entry(canvas_id).active_layer

    def get_active_image(self, canvas_id: str) -> Image.Image:
        """Image of the active layer — the write target for drawing ops."""
        return self.entry(canvas_id).active_layer.image

    def replace_active_image(self, canvas_id: str, image: Image.Image) -> None:
        with self._lock:
            e = self.entry(canvas_id)
            e.active_layer.image = (
                image if image.mode == "RGBA" else image.convert("RGBA")
            )

    # --------------------------------------------------------------- layer ops
    def add_layer(self, canvas_id: str, *, name: str | None = None,
                  fill: tuple[int, int, int, int] = (0, 0, 0, 0),
                  above: int | None = None) -> int:
        """Insert a new transparent layer. Returns its index. ``above`` is
        the index to insert above (default: top of stack)."""
        with self._lock:
            e = self.entry(canvas_id)
            img = Image.new("RGBA", e.size, fill)
            layer = Layer(name=name or f"Layer {len(e.layers) + 1}", image=img)
            idx = len(e.layers) if above is None else int(above) + 1
            idx = max(0, min(idx, len(e.layers)))
            e.layers.insert(idx, layer)
            e.active_index = idx
            return idx

    def remove_layer(self, canvas_id: str, index: int) -> None:
        with self._lock:
            e = self.entry(canvas_id)
            if len(e.layers) <= 1:
                raise ValueError("cannot remove the last layer; flatten or close the canvas instead")
            if not (0 <= index < len(e.layers)):
                raise IndexError(f"layer index {index} out of range")
            e.layers.pop(index)
            if e.active_index >= len(e.layers):
                e.active_index = len(e.layers) - 1
            elif e.active_index > index:
                e.active_index -= 1

    def duplicate_layer(self, canvas_id: str, index: int) -> int:
        with self._lock:
            e = self.entry(canvas_id)
            if not (0 <= index < len(e.layers)):
                raise IndexError(f"layer index {index} out of range")
            dup = e.layers[index].copy()
            dup.name = e.layers[index].name + " copy"
            e.layers.insert(index + 1, dup)
            e.active_index = index + 1
            return index + 1

    def reorder_layer(self, canvas_id: str, src: int, dst: int) -> None:
        with self._lock:
            e = self.entry(canvas_id)
            n = len(e.layers)
            if not (0 <= src < n):
                raise IndexError(f"src index {src} out of range")
            dst = max(0, min(dst, n - 1))
            if src == dst:
                return
            layer = e.layers.pop(src)
            e.layers.insert(dst, layer)
            # Keep active_index pointing at the same layer.
            if e.active_index == src:
                e.active_index = dst
            elif src < e.active_index <= dst:
                e.active_index -= 1
            elif dst <= e.active_index < src:
                e.active_index += 1

    def set_active(self, canvas_id: str, index: int) -> None:
        with self._lock:
            e = self.entry(canvas_id)
            if not (0 <= index < len(e.layers)):
                raise IndexError(f"layer index {index} out of range")
            e.active_index = index

    def merge_down(self, canvas_id: str, index: int) -> None:
        """Composite layer ``index`` onto the layer below it, removing the top one.

        The result keeps the bottom layer's *name*; its image is replaced with
        a canvas-sized flat composite, so offset/opacity/blend_mode/mask are
        reset to neutral values (the new image already encodes those effects).
        """
        with self._lock:
            e = self.entry(canvas_id)
            if not (1 <= index < len(e.layers)):
                raise ValueError("merge_down: need a layer index >= 1 (one below it)")
            top = e.layers[index]
            bottom = e.layers[index - 1]
            merged_full = compose_layers([bottom, top], e.size)
            bottom.image = merged_full
            bottom.offset = (0, 0)
            bottom.opacity = 1.0
            bottom.blend_mode = "normal"
            bottom.mask = None
            e.layers.pop(index)
            if e.active_index >= len(e.layers):
                e.active_index = len(e.layers) - 1
            elif e.active_index >= index:
                e.active_index -= 1

    def flatten(self, canvas_id: str) -> None:
        """Collapse all visible layers into a single Background layer."""
        with self._lock:
            e = self.entry(canvas_id)
            flat = compose_layers(e.layers, e.size)
            e.layers = [Layer(name="Background", image=flat)]
            e.active_index = 0

    # --------------------------------------------------------------- history
    def snapshot(self, canvas_id: str) -> None:
        """Capture pre-mutation state. Clears redo (new branch)."""
        with self._lock:
            e = self.entry(canvas_id)
            state = CanvasState(
                layers=[l.copy() for l in e.layers],
                active_index=e.active_index,
            )
            e.undo_stack.append(state)
            if len(e.undo_stack) > MAX_HISTORY:
                e.undo_stack.pop(0)
            e.redo_stack.clear()

    def undo(self, canvas_id: str) -> bool:
        with self._lock:
            e = self.entry(canvas_id)
            if not e.undo_stack:
                return False
            e.redo_stack.append(CanvasState(
                layers=[l.copy() for l in e.layers],
                active_index=e.active_index,
            ))
            prev = e.undo_stack.pop()
            e.layers = prev.layers
            e.active_index = prev.active_index
            return True

    def redo(self, canvas_id: str) -> bool:
        with self._lock:
            e = self.entry(canvas_id)
            if not e.redo_stack:
                return False
            e.undo_stack.append(CanvasState(
                layers=[l.copy() for l in e.layers],
                active_index=e.active_index,
            ))
            nxt = e.redo_stack.pop()
            e.layers = nxt.layers
            e.active_index = nxt.active_index
            return True

    # --------------------------------------------------------------- canvas-wide
    def map_all_layers(self, canvas_id: str, fn) -> None:
        """Apply ``fn(image) -> image`` to every layer's image. Used by
        canvas-wide transforms (resize/rotate/flip). The function should
        return a new image; size changes are caller's responsibility."""
        with self._lock:
            e = self.entry(canvas_id)
            for layer in e.layers:
                layer.image = fn(layer.image)
                if layer.mask is not None:
                    layer.mask = fn(layer.mask)

    def set_canvas_size(self, canvas_id: str, width: int, height: int) -> None:
        with self._lock:
            e = self.entry(canvas_id)
            e.width = int(width)
            e.height = int(height)


# module-level singleton — MCP tools share one store
store = CanvasStore()
