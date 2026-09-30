"""
Run a fixed-shape upscaler over an image of any size.

The Hexagon NPU wants static input shapes, so the model is compiled for one
tile (270x480 for the bundled model). Larger frames are cut into overlapping
tiles; each tile's border -- where the convolutions saw padding instead of
real neighbours -- is discarded, and only the interior is kept. With a margin
at least as wide as the network's receptive-field radius the stitched result
is identical to running the network over the whole frame at once.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def tile_starts(length: int, tile: int, margin: int) -> list[int]:
    """Tile origins along one axis. The last tile is aligned to the far edge."""
    if length <= tile:
        return [0]
    stride = tile - 2 * margin
    if stride <= 0:
        raise ValueError(f"margin {margin} too large for tile {tile}")
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def _valid_span(start: int, tile: int, length: int, margin: int) -> tuple[int, int]:
    """Part of a tile, in image coordinates, that is kept after stitching."""
    lo = start if start == 0 else start + margin
    hi = start + tile if start + tile >= length else start + tile - margin
    return lo, hi


def tile_count(height: int, width: int, tile_h: int, tile_w: int, margin: int) -> int:
    return (len(tile_starts(max(height, tile_h), tile_h, margin))
            * len(tile_starts(max(width, tile_w), tile_w, margin)))


def upscale_tiled(image: np.ndarray, run: Callable[[np.ndarray], np.ndarray],
                  tile_h: int, tile_w: int, scale: int, margin: int = 8) -> np.ndarray:
    """
    Upscale a CHW float32 image with a function that accepts exactly
    (1, C, tile_h, tile_w) and returns (1, C, tile_h*scale, tile_w*scale).
    """
    channels, height, width = image.shape

    # Frames smaller than one tile are edge-padded up to it and cropped after.
    pad_h, pad_w = max(0, tile_h - height), max(0, tile_w - width)
    if pad_h or pad_w:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    full_h, full_w = image.shape[1:]

    out = np.empty((channels, full_h * scale, full_w * scale), dtype=np.float32)
    tile = np.empty((1, channels, tile_h, tile_w), dtype=np.float32)

    for y in tile_starts(full_h, tile_h, margin):
        y0, y1 = _valid_span(y, tile_h, full_h, margin)
        for x in tile_starts(full_w, tile_w, margin):
            x0, x1 = _valid_span(x, tile_w, full_w, margin)
            tile[0] = image[:, y:y + tile_h, x:x + tile_w]
            result = run(tile)[0]
            out[:, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = result[
                :, (y0 - y) * scale:(y1 - y) * scale, (x0 - x) * scale:(x1 - x) * scale]

    return out[:, :height * scale, :width * scale]
