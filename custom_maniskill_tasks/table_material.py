"""`table_material`: make the table top look like a different material.

A single keyword naming the material: `"default"` (the stock wood, untouched), `"marble"`, `"glass"`,
`"metal"` (brushed steel), `"gold"`, `"concrete"` or `"rubber"` (matt black).

Only the *render* material of the table is edited -- its colour, roughness, metallic, specular and
transmission. The table's collision shape is a separate box built from its own friction material,
which this never touches, so contact behaviour is identical under every material and only what the
cameras see changes. (`"glass"` is a see-through look, not a see-through table: nothing falls
through it.)

It combines with the other appearance kwargs. It is applied before `texture`, so
`table_material="metal", texture="table-checker"` is a checkered table that is still metallic, the
pattern replacing the material's colour only; and before `lighting`'s `table_tint`, which still
multiplies in on top.

It is an ordinary env kwarg (`TableMaterialMixin`), applied in `_load_scene`, before the scene first
renders -- a material edited after that is stored but never reaches the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from custom_maniskill_tasks.texture import _render_parts, _texture_from_srgb, pattern_mask

DEFAULT_TABLE_MATERIAL = "default"
MATERIAL_TEXTURE_SIZE = 512


@dataclass(frozen=True)
class MaterialSpec:
    """The look of one material. Colours are sRGB in [0, 1]."""

    base_color: tuple[float, float, float]
    roughness: float
    metallic: float = 0.0
    specular: float = 0.5
    transmission: float = 0.0
    ior: float = 1.45
    texture: Callable[[], np.ndarray] | None = field(default=None, compare=False)
    """Makes an (h, w, 3) sRGB tiling colour texture; None for a plain `base_color`."""


def _fbm(octaves: tuple[int, ...], size: int = MATERIAL_TEXTURE_SIZE) -> np.ndarray:
    """Tileable fractal noise in [0, 1]: smooth value noise summed over grids of these sizes."""
    total = sum(pattern_mask("noise", n, size) / (i + 1) for i, n in enumerate(octaves))
    return (total - total.min()) / max(total.max() - total.min(), 1e-9)


def _marble_texture() -> np.ndarray:
    axis = (np.arange(MATERIAL_TEXTURE_SIZE) + 0.5) / MATERIAL_TEXTURE_SIZE
    x, y = np.meshgrid(axis, axis)
    turbulence = _fbm((3, 6, 12))
    # integer frequencies in x and y, and tiling noise, so the veins tile
    veins = np.abs(np.sin(np.pi * (2 * x + 3 * y + 2.5 * turbulence)))
    veins = 1.0 - np.clip(veins, 0, 1) ** 0.12  # thin dark lines
    base = 0.93 - 0.08 * _fbm((4, 8))
    tone = np.clip(base - 0.55 * veins, 0.0, 1.0)
    return np.stack([tone, tone, tone * 0.98 + 0.01], axis=-1)


def _brushed_metal_texture() -> np.ndarray:
    rng = np.random.RandomState(1)
    streaks = rng.uniform(size=(MATERIAL_TEXTURE_SIZE, 1))  # one value per row: horizontal grain
    tone = 0.74 + 0.07 * (streaks - 0.5) * 2 + 0.03 * (_fbm((4, 8)) - 0.5)
    return np.repeat(tone[..., None], 3, axis=-1) * np.array([0.98, 1.0, 1.02])


def _concrete_texture() -> np.ndarray:
    tone = 0.42 + 0.22 * (_fbm((6, 12, 24, 48)) - 0.5)
    return np.repeat(tone[..., None], 3, axis=-1)


MATERIALS: dict[str, MaterialSpec] = {
    "marble": MaterialSpec((0.9, 0.9, 0.9), roughness=0.12, specular=0.6, texture=_marble_texture),
    "glass": MaterialSpec(
        (0.82, 0.93, 0.92), roughness=0.02, specular=0.8, transmission=1.0, ior=1.5
    ),
    "metal": MaterialSpec(
        (0.74, 0.74, 0.76), roughness=0.35, metallic=1.0, texture=_brushed_metal_texture
    ),
    "gold": MaterialSpec((1.0, 0.77, 0.34), roughness=0.25, metallic=1.0),
    "concrete": MaterialSpec((0.45, 0.45, 0.45), roughness=0.95, specular=0.2, texture=_concrete_texture),
    "rubber": MaterialSpec((0.06, 0.06, 0.06), roughness=0.85, specular=0.2),
}


def canonical_table_material(material: str | None) -> str:
    """Resolve a `table_material` to its name, rejecting unknown ones early -- a misspelled one
    that fell through would build a working env still showing the wood."""
    if material is None or material == "":
        return DEFAULT_TABLE_MATERIAL
    if not isinstance(material, str) or material not in (DEFAULT_TABLE_MATERIAL, *MATERIALS):
        raise ValueError(
            f"Unknown table_material {material!r}, expected {DEFAULT_TABLE_MATERIAL!r} or one of "
            f"{', '.join(repr(m) for m in MATERIALS)}."
        )
    return material


def _apply_table_material(env, name: str) -> None:
    table_scene = getattr(env, "table_scene", None)
    if table_scene is None:
        raise ValueError(
            f"{type(env).__name__} has no `table_scene`, so table_material={name!r} has no table "
            "to change."
        )
    spec = MATERIALS[name]
    texture = None if spec.texture is None else _texture_from_srgb(spec.texture())
    # The same values go on every part of every parallel env's table, so a material they share is
    # set to the same thing repeatedly rather than changed cumulatively.
    for part in _render_parts(table_scene.table):
        material = part.material
        # the glb's own maps would override the scalars set here
        material.metallic_texture = None
        material.roughness_texture = None
        material.normal_texture = None
        material.emission_texture = None
        material.transmission_texture = None
        material.emission = [0.0, 0.0, 0.0, 1.0]
        material.base_color = [*spec.base_color, 1.0]
        material.base_color_texture = texture
        material.roughness = spec.roughness
        material.metallic = spec.metallic
        material.specular = spec.specular
        material.transmission = spec.transmission
        material.ior = spec.ior


class TableMaterialMixin:
    """Gives a task the `table_material` kwarg: the material the table top looks made of."""

    def __init__(self, *args, table_material=DEFAULT_TABLE_MATERIAL, **kwargs):
        self._table_material = canonical_table_material(table_material)
        super().__init__(*args, **kwargs)

    @property
    def table_material(self) -> str:
        """The material this env's table was built with; "default" is the stock wood."""
        return self._table_material

    def _set_table_material(self, condition) -> None:
        """Take `condition` as this env's material from the next reconfiguration (`domain_randomization`)."""
        self._table_material = canonical_table_material(condition)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        if self._table_material != DEFAULT_TABLE_MATERIAL:
            _apply_table_material(self, self._table_material)


def supports_table_material(task_name: str) -> bool:
    from mani_skill.utils.registration import REGISTERED_ENVS

    spec = REGISTERED_ENVS.get(task_name)
    return spec is not None and issubclass(spec.cls, TableMaterialMixin)
