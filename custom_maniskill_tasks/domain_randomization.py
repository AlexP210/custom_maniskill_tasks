"""`domain_randomization`: a list of appearance axes, each drawn afresh on every episode.

    make_env("PushCube-v1.1", camera_view="wrist",
             domain_randomization=["lighting", "object_colour", "table_material"])

Each name in the list switches on one axis, and on every reset the env samples a condition from
each switched-on axis independently:

| axis | what is drawn |
| --- | --- |
| `lighting` | the scene's brightness, the key light's direction and the lights' colour temperature |
| `object_colour` | an offset to each of the manipulated object's R, G and B |
| `table_colour` | a gain on each of the table's R, G and B, and a brightness |
| `table_texture` | a scale, shear, rotation and translation of the table's texture |
| `table_material` | the table's material, among `default` (wood) and `table_material.MATERIALS` |

Training and evaluation draw from disjoint parts of each axis (`domain_randomization_split`):

- a continuous parameter has a range (`RANGES`), and `"train"` draws from its central 80% while
  `"heldout"` draws from the outer 20%. A range with its neutral value in the middle (a colour
  offset, a colour temperature) has the 20% split into the two ends, 10% each; a range that
  starts at its neutral value (the key light's direction, which runs from the default to `side`)
  holds out its far 20%. The range is taken in log space where the quantity is a multiplier
  (brightness, texture scale).
- a discrete parameter has its values divided: `HELDOUT_TABLE_MATERIALS` are never drawn in
  training, and are all `"heldout"` draws.

Under `"heldout"` every parameter of every switched-on axis is drawn from its held-out part, so
those episodes are as far from training as the axes go. Parameters of an axis, and the axes, are
independent of each other.

The ranges are the ones the evaluation sweeps use (`axes.AXES`): brightness 0.25x to 8x, the key
light from the default to `side`, the temperature from `warm-set-1` (2700 K) through the default
(6500 K) to `cool-set-1` (12000 K).

The draw is made from the episode seed, so the same `reset(seed=...)` gives the same condition and
an unseeded reset a fresh one, and it never touches the episode rng the task itself draws from. It
is one draw per reset shared by every parallel env (the table's materials are shared between
them, so the table axes could not differ per env). `episode_randomization` says what was drawn.

Everything but the images changes what is built (the lights, the table, the object), so a draw
takes effect only by reconfiguring, which this does on every reset (`reconfiguration_freq=1`).
ManiSkill cannot reconfigure on a partial reset, so every env must reset together, which the
`-v1.1` ids' never-terminating episodes guarantee. The first scene is built from a draw too.

Lighting and the table's colour are applied through `LightingMixin` (the former as a stack of its
`bright-set`, `side-set`, `warm-set` / `cool-set` presets, the latter as its `table_tint`), so an
axis here cannot also be set by the corresponding fixed kwarg; asking for both raises.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from custom_maniskill_tasks.lighting import (
    DEFAULT_LIGHTING,
    _linear_to_srgb,
    _remapped_texture,
    _srgb_to_linear,
    canonical_lighting,
)
from custom_maniskill_tasks.table_material import DEFAULT_TABLE_MATERIAL, MATERIALS, canonical_table_material
from custom_maniskill_tasks.texture import _render_parts

AXES_RANDOMIZED = ("lighting", "object_colour", "table_colour", "table_texture", "table_material")
"""The names `domain_randomization` takes."""

SPLITS = ("train", "heldout")
HELDOUT_FRACTION = 0.2
"""The part of each continuous range kept out of training."""

HELDOUT_TABLE_MATERIALS = ("glass", "gold")
"""The materials only `"heldout"` draws. The two that look least like the rest: see-through, and
a coloured metal."""
TRAIN_TABLE_MATERIALS = (DEFAULT_TABLE_MATERIAL, *(m for m in MATERIALS if m not in HELDOUT_TABLE_MATERIALS))


@dataclass(frozen=True)
class Range:
    """One continuous parameter's range, and how it is split between training and held out."""

    low: float
    high: float
    log: bool = False
    """Uniform in log(value) rather than in value: for a quantity that is a multiplier."""
    two_sided: bool = True
    """The neutral value is in the middle, so the held-out part is both ends (half of it each); if
    False the neutral value is `low` and the held-out part is the top end alone."""

    def bands(self, split: str) -> list[tuple[float, float]]:
        """The sub-intervals of [0, 1] (the range's own scale) that `split` draws from."""
        h = HELDOUT_FRACTION
        if split == "train":
            return [(h / 2, 1 - h / 2)] if self.two_sided else [(0.0, 1 - h)]
        if split == "heldout":
            return [(0.0, h / 2), (1 - h / 2, 1.0)] if self.two_sided else [(1 - h, 1.0)]
        raise ValueError(f"unknown split {split!r}, expected one of {', '.join(SPLITS)}")

    def sample(self, rng, split: str) -> float:
        """One value, uniform over the bands of `split` (so the two held-out ends are equally
        likely)."""
        bands = self.bands(split)
        widths = np.array([b - a for a, b in bands])
        which = int(rng.choice(len(bands), p=widths / widths.sum()))
        a, b = bands[which]
        u = float(rng.uniform(a, b))
        if self.log:
            return float(np.exp(np.log(self.low) + u * (np.log(self.high) - np.log(self.low))))
        return self.low + u * (self.high - self.low)

    def contains(self, value: float, split: str) -> bool:
        """Whether `value` lies in `split`'s part of this range, for tests and for the evaluation
        scripts' bookkeeping."""
        if self.log:
            u = (np.log(value) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        else:
            u = (value - self.low) / (self.high - self.low)
        return any(a - 1e-9 <= u <= b + 1e-9 for a, b in self.bands(split))


RANGES: dict[str, dict[str, Range]] = {
    "lighting": {
        "brightness": Range(0.25, 8.0, log=True),
        "direction": Range(0.0, 1.0, two_sided=False),
        "temperature": Range(-1.0, 1.0),
    },
    "object_colour": {c: Range(-0.4, 0.4) for c in "rgb"},
    "table_colour": {
        **{c: Range(0.6, 1.4) for c in "rgb"},
        "brightness": Range(0.4, 1.6),
    },
    "table_texture": {
        "scale": Range(0.5, 2.0, log=True),
        "shear": Range(-0.3, 0.3),
        "rotation": Range(-45.0, 45.0),
        "shift_x": Range(-0.5, 0.5),
        "shift_y": Range(-0.5, 0.5),
    },
}
"""Every continuous parameter's range, by axis. In the units of the axes it comes from:

- `lighting.brightness` is a multiple of the default's light levels (`bright-set`); `direction` is
  how far the key light is turned from the default's towards `side` (`side-set`, 0 to 1);
  `temperature` is signed, negative `warm-set-<-t>` and positive `cool-set-<t>`, so -1 is 2700 K
  and 1 is 12000 K.
- `object_colour.<r|g|b>` is added to that channel of the object's colour, which is in [0, 1].
- `table_colour.<r|g|b>` multiplies the channel of the table's colour, and `brightness`
  multiplies all three on top.
- `table_texture`: `scale` multiplies the texture's size, `shear` is the x-shear of the texture
  plane, `rotation` is in degrees, and `shift_x`/`shift_y` are in periods of the texture (it tiles,
  so a shift of one period is no shift)."""


def canonical_domain_randomization(axes: str | Sequence[str] | None) -> tuple[str, ...]:
    """Resolve a `domain_randomization` into a tuple of axis names in a fixed order, rejecting
    unknown names early -- a misspelled one that fell through would build an env that never
    randomizes it."""
    if axes is None:
        return ()
    if isinstance(axes, str):
        axes = [axes]
    names = list(axes)
    unknown = [a for a in names if a not in AXES_RANDOMIZED]
    if unknown:
        raise ValueError(
            f"Unknown domain_randomization {unknown[0]!r}, expected a list of "
            f"{', '.join(repr(a) for a in AXES_RANDOMIZED)}."
        )
    return tuple(a for a in AXES_RANDOMIZED if a in names)


def draw_axis(axis: str, rng, split: str = "train") -> dict[str, float | str]:
    """One condition of `axis` from `split`, as {parameter: value}, using `rng` (a numpy
    Generator)."""
    if axis == "table_material":
        pool = TRAIN_TABLE_MATERIALS if split == "train" else HELDOUT_TABLE_MATERIALS
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}, expected one of {', '.join(SPLITS)}")
        return {"material": pool[int(rng.integers(len(pool)))]}
    return {name: r.sample(rng, split) for name, r in RANGES[axis].items()}


def lighting_preset(condition: dict[str, float]) -> str:
    """The `lighting` stack that `lighting`'s drawn `condition` stands for. Brightness first,
    since `bright-set` sets the levels outright and the colour temperature then tints them."""
    t = condition["temperature"]
    return "+".join((
        f"bright-set-{0.3 * condition['brightness']:.4f}-{condition['brightness']:.4f}",
        f"side-set-{condition['direction']:.4f}",
        f"{'warm' if t < 0 else 'cool'}-set-{abs(t):.4f}",
    ))


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


class DomainRandomizationMixin:
    """Gives a task the `domain_randomization` kwarg, and `domain_randomization_split` to say
    whether the draws are training or held-out ones. Mixed in ahead of `LightingMixin`, so it sees
    the kwargs first and the reconfiguration last."""

    def __init__(self, *args, domain_randomization=(), domain_randomization_split="train", **kwargs):
        self._domain_randomization = canonical_domain_randomization(domain_randomization)
        if domain_randomization_split not in SPLITS:
            raise ValueError(
                f"Unknown domain_randomization_split {domain_randomization_split!r}, expected one of "
                f"{', '.join(SPLITS)}."
            )
        self._domain_randomization_split = domain_randomization_split
        self._episode_randomization: dict[str, dict[str, float | str]] = {}
        self._fixed_lighting = canonical_lighting(kwargs.get("lighting", "default"))
        chosen = self._domain_randomization
        if "lighting" in chosen and self._fixed_lighting != DEFAULT_LIGHTING:
            raise ValueError(
                f"domain_randomization has 'lighting', which draws the lighting every episode, but "
                f"lighting={kwargs['lighting']!r} was also passed."
            )
        if "table_colour" in chosen and self._fixed_lighting.table_tint != DEFAULT_LIGHTING.table_tint:
            raise ValueError(
                f"domain_randomization has 'table_colour', which draws the table's colour every "
                f"episode, but lighting={kwargs['lighting']!r} tints the table too."
            )
        if "table_material" in chosen and canonical_table_material(
            kwargs.get("table_material")
        ) != DEFAULT_TABLE_MATERIAL:
            raise ValueError(
                f"domain_randomization has 'table_material', which draws the material every "
                f"episode, but table_material={kwargs['table_material']!r} was also passed."
            )
        if "object_colour" in chosen and getattr(type(self), "object_of_interest", None) is None:
            raise ValueError(
                f"{type(self).__name__} has no single object of interest, so 'object_colour' has no "
                "object to recolour."
            )
        if chosen:
            frequency = kwargs.setdefault("reconfiguration_freq", 1)
            if frequency != 1:
                raise ValueError(
                    f"domain_randomization={list(chosen)} is redrawn on every reset, which needs "
                    f"reconfiguration_freq=1, but reconfiguration_freq={frequency} was passed."
                )
        super().__init__(*args, **kwargs)

    @property
    def domain_randomization(self) -> tuple[str, ...]:
        return self._domain_randomization

    @property
    def episode_randomization(self) -> dict[str, dict[str, float | str]]:
        """What each randomized axis drew for the current episode, as {axis: {parameter: value}};
        empty if nothing is randomized."""
        return {axis: dict(drawn) for axis, drawn in self._episode_randomization.items()}

    def _draw(self, axis: str) -> dict[str, float | str]:
        # seeded by the episode seed and the axis, so it is reproducible and independent of the
        # task's own draws and of the other axes'
        seed = int(self._episode_seed[0]) % 2**63
        rng = np.random.default_rng([seed, zlib.crc32(axis.encode())])
        return draw_axis(axis, rng, self._domain_randomization_split)

    def _reconfigure(self, options=dict()):
        for axis in self._domain_randomization:
            self._episode_randomization[axis] = self._draw(axis)
        drawn = self._episode_randomization
        if "lighting" in drawn or "table_colour" in drawn:
            config = self._fixed_lighting
            if "lighting" in drawn:
                config = canonical_lighting(lighting_preset(drawn["lighting"]))
            if "table_colour" in drawn:
                config = replace(config, table_tint=table_tint(drawn["table_colour"]))
            self._set_lighting(config)
        if "table_material" in drawn:
            self._set_table_material(drawn["table_material"]["material"])
        super()._reconfigure(options)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        drawn = self._episode_randomization
        if "object_colour" in drawn:
            _apply_object_colour(self, [drawn["object_colour"][c] for c in "rgb"])
        if "table_texture" in drawn:
            _apply_table_texture(self, drawn["table_texture"])


def supports_domain_randomization(task_name: str) -> bool:
    from mani_skill.utils.registration import REGISTERED_ENVS

    spec = REGISTERED_ENVS.get(task_name)
    return spec is not None and issubclass(spec.cls, DomainRandomizationMixin)
