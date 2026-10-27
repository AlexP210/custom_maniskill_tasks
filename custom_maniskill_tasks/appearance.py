"""How the `object_colour` and `table_texture` perturbation classes change the scene, and the table
tint `table_colour` stands for. The classes themselves -- their aspects, ranges and splits -- are
in `perturbation_classes`; `PerturbationClassesMixin` calls these.

Both appliers edit render materials the scene has already built, so they run from `_load_scene`,
before the scene first renders: a material edited after that is stored but never reaches the
renderer.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from custom_maniskill_tasks.lighting import _linear_to_srgb, _remapped_texture, _srgb_to_linear
from custom_maniskill_tasks.texture import _render_parts


def table_tint(condition: dict[str, float]) -> tuple[float, float, float]:
    """The `table_tint` that `table_colour`'s drawn `condition` stands for."""
    return tuple(float(condition[c] * condition["brightness"]) for c in "rgb")


def _apply_object_colour(env, offsets: Sequence[float]) -> None:
    """Add `offsets` (r, g, b) to the colour of every render part of `env`'s object of interest,
    clipped to [0, 1]. Every original colour is read before any is written and each result is a
    function of its original, so a material shared between parallel envs is shifted once, not once
    per env. A textured object (a YCB model) has its texture shifted the same way, into a new one
    (see `_remapped_texture`)."""
    attribute = getattr(env, "object_of_interest", None)
    if attribute is None:
        raise ValueError(
            f"{type(env).__name__} has no single object of interest, so 'object_colour' has no "
            "object to recolour."
        )
    offsets = np.asarray(offsets, dtype=np.float64)
    parts = _render_parts(getattr(env, attribute))
    edits = [(part.material, list(part.material.base_color), part.material.base_color_texture) for part in parts]
    if not edits:
        raise ValueError(f"the object {attribute!r} has no render parts, so there is nothing to recolour.")
    shifted = [
        (
            material,
            [*np.clip(np.asarray(base_color[:3]) + offsets, 0.0, 1.0).tolist(), base_color[3]],
            None if texture is None else _remapped_texture(
                texture, lambda c: _srgb_to_linear(np.clip(_linear_to_srgb(c) + offsets, 0.0, 1.0))
            ),
        )
        for material, base_color, texture in edits
    ]
    for material, base_color, texture in shifted:
        material.base_color = base_color
        if texture is not None:
            material.base_color_texture = texture


def transform_pixels(
    pixels: np.ndarray, scale: float, shear: float, rotation: float, shift_x: float, shift_y: float
) -> np.ndarray:
    """`pixels` (h, w, c uint8) tiled infinitely and then scaled by `scale` about its centre,
    sheared in x by `shear`, rotated by `rotation` degrees and shifted by (`shift_x`, `shift_y`)
    periods, sampled bilinearly back onto the same grid. The identity at (1, 0, 0, 0, 0)."""
    h, w = pixels.shape[:2]
    theta = np.radians(rotation)
    rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    forward = rotate @ np.array([[1.0, shear], [0.0, 1.0]]) @ np.diag([scale, scale])
    inverse = np.linalg.inv(forward)
    x, y = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    source = (np.stack([x - 0.5 - shift_x, y - 0.5 - shift_y], axis=-1) @ inverse.T) + 0.5
    fx, fy = source[..., 0] * w - 0.5, source[..., 1] * h - 0.5
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = (fx - x0)[..., None], (fy - y0)[..., None]
    at = lambda yy, xx: pixels[yy % h, xx % w].astype(np.float64)
    top = at(y0, x0) * (1 - tx) + at(y0, x0 + 1) * tx
    bottom = at(y0 + 1, x0) * (1 - tx) + at(y0 + 1, x0 + 1) * tx
    return np.clip(np.round(top * (1 - ty) + bottom * ty), 0, 255).astype(np.uint8)


def _apply_table_texture(env, condition: dict[str, float]) -> None:
    """Move the table's textures as `condition` says. Every parallel env's table is the same model,
    so each texture is worked out once, from the first env's, and assigned to all of them (as
    `lighting._tint_table` does), which also keeps the transform from compounding on a texture
    they share. A part with no texture (a plain-coloured material) has nothing to move."""
    import sapien

    table_scene = getattr(env, "table_scene", None)
    if table_scene is None:
        raise ValueError(f"{type(env).__name__} has no `table_scene`, so there is no table texture to move.")
    entities = table_scene.table._objs

    def parts(entity):
        body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        return [part for shape in body.render_shapes for part in shape.parts]

    template = parts(entities[0])
    moved = []
    for part in template:
        texture = part.material.base_color_texture
        if texture is None:
            moved.append(None)
            continue
        pixels = texture.download()
        out = pixels.copy()
        out[..., :3] = transform_pixels(pixels[..., :3], **condition)
        moved.append(sapien.render.RenderTexture2D(
            out, texture.format, texture.mipmap_levels, texture.filter_mode,
            texture.address_mode, texture.is_srgb,
        ))
    for entity in entities:
        for part, texture in zip(parts(entity), moved):
            if texture is not None:
                part.material.base_color_texture = texture
