"""Render-material helpers shared by the table appearance code: procedural patterns, building a
sapien texture from pixels, and finding the render parts of an actor."""

from __future__ import annotations

import numpy as np
import sapien

TEXTURE_SIZE = 256


def pattern_mask(pattern: str, repeats: int, size: int = TEXTURE_SIZE) -> np.ndarray:
    """(size, size) float in [0, 1], 1 where the pattern is light and 0 where it is dark."""
    axis = (np.arange(size) + 0.5) / size
    x, y = np.meshgrid(axis, axis)
    if pattern == "checker":
        return ((np.floor(x * repeats) + np.floor(y * repeats)) % 2).astype(np.float64)
    if pattern == "stripes":
        return (((x + y) * repeats) % 1.0 < 0.5).astype(np.float64)
    if pattern == "dots":
        cell_x, cell_y = (x * repeats) % 1.0 - 0.5, (y * repeats) % 1.0 - 0.5
        return (np.hypot(cell_x, cell_y) > 0.3).astype(np.float64)
    if pattern == "noise":
        # value noise: random values on a `repeats` grid (wrapping, so it tiles), smoothly
        # interpolated, then stretched to fill [0, 1]
        rng = np.random.RandomState(0)
        grid = rng.uniform(size=(repeats, repeats))
        fx, fy = x * repeats, y * repeats
        x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = fx - x0, fy - y0
        tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
        x0, y0, x1, y1 = x0 % repeats, y0 % repeats, (x0 + 1) % repeats, (y0 + 1) % repeats
        top = grid[y0, x0] * (1 - tx) + grid[y0, x1] * tx
        bottom = grid[y1, x0] * (1 - tx) + grid[y1, x1] * tx
        field = top * (1 - ty) + bottom * ty
        return (field - field.min()) / max(field.max() - field.min(), 1e-9)
    raise ValueError(f"unknown pattern {pattern!r}")


def _texture_from_srgb(rgb: np.ndarray) -> "sapien.render.RenderTexture2D":
    """A tiling sRGB texture from (h, w, 3) floats in [0, 1]."""
    pixels = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
    pixels[..., :3] = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    pixels[..., 3] = 255
    return sapien.render.RenderTexture2D(pixels, "R8G8B8A8Unorm", 4, "linear", "repeat", True)


def _render_parts(actor) -> list:
    """Every render part of every entity of `actor`."""
    parts = []
    for entity in actor._objs:
        body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        if body is None:
            continue
        parts.extend(part for shape in body.render_shapes for part in shape.parts)
    return parts
