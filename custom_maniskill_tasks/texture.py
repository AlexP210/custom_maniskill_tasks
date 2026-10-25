"""`texture`: put a pattern on the table and on objects.

A texture is a keyword or several joined with "+": `<target>-<pattern>` or `<target>-<pattern>-<n>`
(`"table-checker+object-stripes"`, `"cube-dots+peg-noise-12"`).

Targets:

| target | what it is |
| --- | --- |
| `table` | the table top and its other parts |
| `object` | the task's object of interest (the cube, sphere or peg) |
| `distractors` | the YCB distractors, if there are any |
| an actor's name | that one actor, e.g. `cube`, `sphere`, `peg`, `bin`, `goal_region` |

Patterns: `checker`, `stripes` (diagonal), `dots` (dark dots on light) and `noise` (smooth blotches).
`n` is how many times it repeats across the texture (checker squares or stripes, dots per side,
blotches), a default per pattern if left out.

A more specific target wins over a more general one, whatever order they are named in, so
`"object-checker+cube-dots"` puts dots on the cube and checks on anything else `object` means. A
name that is none of an actor's raises, listing the actors there are.

The table is *replaced*: its wood is swapped for the pattern in a light and a dark tone. Objects
keep their colour and gain the pattern on top, as a darkening of it: a red cube under `checker` is
red and dark red squares, so what differs is texture alone, not colour. (Both apply to an object
with a texture of its own -- a YCB model -- by darkening that.) `"default"` (or an empty list) is
none.

It is an ordinary env kwarg (`TextureMixin`), applied in `_load_scene`, before the scene first
renders -- a material edited after that is stored but never reaches the renderer. Mixed in after
`LightingMixin`, so an `object-hue-<degrees>` shift turns the colour under the pattern, and after
`DistractorsMixin` in the class list so the distractors exist to be textured.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import sapien

from custom_maniskill_tasks.lighting import _linear_to_srgb, _remapped_texture, _srgb_to_linear

DEFAULT_TEXTURE = "default"
PATTERNS = ("checker", "stripes", "dots", "noise")
DEFAULT_REPEATS = {"checker": 8, "stripes": 8, "dots": 6, "noise": 8}
CLASS_TARGETS = ("table", "object", "distractors")
"""The general targets; every other name is an actor's."""
TEXTURE_SIZE = 256

TABLE_LIGHT = np.array([0.90, 0.90, 0.88])
TABLE_DARK = np.array([0.14, 0.19, 0.30])
"""The two tones (sRGB) the table's pattern is drawn in."""
OBJECT_DARKEST = 0.35
"""How much of its colour an object keeps where the pattern is dark."""

_KEYWORD = re.compile(
    r"^(?P<target>\w+)-(?P<pattern>checker|stripes|dots|noise)(?:-(?P<repeats>\d+))?$"
)

TextureSpec = tuple[tuple[str, str, int], ...]
"""((target, pattern, repeats), ...) in the order named."""


def canonical_texture(texture: str | Sequence[str] | TextureSpec | None) -> TextureSpec:
    """Resolve a `texture` into ((target, pattern, repeats), ...), rejecting unknown patterns
    early -- a misspelled one that fell through would build a working env with no texture."""
    if texture is None:
        return ()
    if isinstance(texture, str):
        parts = texture.split("+")
    else:
        parts = list(texture)
    entries = []
    for part in parts:
        if isinstance(part, (tuple, list)):
            entries.append(tuple(part))
            continue
        if part in (DEFAULT_TEXTURE, ""):
            continue
        match = _KEYWORD.match(part)
        if match is None:
            raise ValueError(
                f"Unknown texture {part!r}, expected 'default' or a \"+\"-joined stack of "
                f"'<target>-<pattern>' or '<target>-<pattern>-<n>' with a target of 'table', "
                f"'object', 'distractors' or an actor's name, and a pattern of "
                f"{', '.join(repr(p) for p in PATTERNS)} (e.g. 'table-checker+cube-dots-4')."
            )
        pattern = match.group("pattern")
        repeats = int(match.group("repeats") or DEFAULT_REPEATS[pattern])
        if repeats < 1:
            raise ValueError(f"{part!r} repeats the pattern {repeats} times: 1 or more.")
        entries.append((match.group("target"), pattern, repeats))
    return tuple(entries)


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


def _resized(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = (np.arange(height) * mask.shape[0] // height)
    cols = (np.arange(width) * mask.shape[1] // width)
    return mask[np.ix_(rows, cols)]


def _render_parts(actor) -> list:
    """Every render part of every entity of `actor`."""
    parts = []
    for entity in actor._objs:
        body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        if body is None:
            continue
        parts.extend(part for shape in body.render_shapes for part in shape.parts)
    return parts


def _texture_table(env, pattern: str, repeats: int) -> None:
    """Replace the table's colours with `pattern` drawn in `TABLE_LIGHT` and `TABLE_DARK`."""
    table_scene = getattr(env, "table_scene", None)
    if table_scene is None:
        raise ValueError(f"{type(env).__name__} has no `table_scene`, so there is no table to texture.")
    mask = pattern_mask(pattern, repeats)[..., None]
    texture = _texture_from_srgb(TABLE_DARK * (1 - mask) + TABLE_LIGHT * mask)
    for part in _render_parts(table_scene.table):
        part.material.base_color_texture = texture


def _texture_actor(actor, pattern: str, repeats: int) -> None:
    """Darken `actor`'s colours by `pattern`, keeping their hue. Every part's original colour is
    read before any is written, and each result worked out from its original, so a material shared
    between parallel envs is patterned once, not darkened again for each env that shares it."""
    mask = OBJECT_DARKEST + (1.0 - OBJECT_DARKEST) * pattern_mask(pattern, repeats)
    edits = []
    for part in _render_parts(actor):
        material = part.material
        edits.append((material, list(material.base_color), material.base_color_texture))
    cache: dict = {}
    made = []
    for material, base_color, texture in edits:
        if texture is None:
            key = (tuple(np.round(base_color, 6)),)
            if key not in cache:
                linear = np.asarray(base_color[:3]) * mask[..., None]
                cache[key] = _texture_from_srgb(_linear_to_srgb(np.clip(linear, 0.0, 1.0)))
            made.append((material, cache[key]))
        else:
            height, width = texture.download().shape[:2]
            factor = _resized(mask, height, width)[..., None]
            made.append((material, _remapped_texture(texture, lambda c, f=factor: c * f)))
    for material, texture in made:
        material.base_color_texture = texture


class TextureMixin:
    """Gives a task the `texture` kwarg: the patterns it names, on the targets it names."""

    object_of_interest: str
    """On a task class: the attribute holding the object `object-<pattern>` means."""

    def __init__(self, *args, texture=DEFAULT_TEXTURE, **kwargs):
        self._texture_spec = texture
        self._texture = canonical_texture(texture)
        super().__init__(*args, **kwargs)

    @property
    def texture(self) -> TextureSpec:
        """The textures this env was built with; empty unless asked otherwise."""
        return self._texture

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        if not self._texture:
            return
        actors_by_name = self.scene.actors
        table = None
        by_class: dict[str, tuple[str, int]] = {}
        by_actor: dict[str, tuple[str, int]] = {}
        for target, pattern, repeats in self._texture:
            if target == "table":
                table = (pattern, repeats)
            elif target in CLASS_TARGETS:
                by_class[target] = (pattern, repeats)
            elif target in actors_by_name:
                by_actor[target] = (pattern, repeats)
            else:
                raise ValueError(
                    f"texture target {target!r} is none of 'table', 'object', 'distractors' or an "
                    f"actor in this scene: {', '.join(sorted(actors_by_name))}."
                )

        resolved: dict[str, tuple[str, int]] = {}
        if "object" in by_class:
            attribute = getattr(self, "object_of_interest", None)
            if attribute is None:
                raise ValueError(
                    f"{type(self).__name__} has no single object of interest, so 'object' means "
                    "nothing here: name the actors (e.g. 'cube-checker+peg-dots') instead."
                )
            resolved[getattr(self, attribute).name] = by_class["object"]
        if "distractors" in by_class:
            if not getattr(self, "_distractor_actors", None):
                raise ValueError("a texture was asked for on the distractors, but there are none.")
            for name in self._distractor_actors:
                resolved[name] = by_class["distractors"]
        resolved.update(by_actor)  # a named actor beats a class of them

        if table is not None:
            _texture_table(self, *table)
        for name, (pattern, repeats) in resolved.items():
            _texture_actor(actors_by_name[name], pattern, repeats)


def supports_texture(task_name: str) -> bool:
    from mani_skill.utils.registration import REGISTERED_ENVS

    spec = REGISTERED_ENVS.get(task_name)
    return spec is not None and issubclass(spec.cls, TextureMixin)
