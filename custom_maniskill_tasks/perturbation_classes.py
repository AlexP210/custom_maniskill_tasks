"""Perturbation classes: the one interface for changing how a scene looks.

A *perturbation class* is a group of *aspects* that are changed together -- `lighting` has a
`temperature`, a `brightness` and a `direction`; `table_colour` a red, green and blue gain and a
`brightness`. Each aspect has a neutral value (the stock scene), a full range, and a split of that
range into the part training draws from and the part held out for evaluation.

A config names the classes it wants changed, and says per class what each aspect should be:

    lighting: default                    # every aspect at its neutral value
    lighting: randomize-train            # every aspect drawn afresh each episode, from training's range
    lighting:
      temperature: randomize-evaluation  # drawn from the held-out part of the range
      brightness: 1.5                    # fixed
                                         # direction is not named, so it is neutral

An aspect is `default`, a number (a material name, for the categorical `table_material.material`),
or `randomize-train` / `randomize-evaluation` / `randomize-full`. A class given as a bare string
applies that to all its aspects. Classes left out are `default`.

- `train` is the central 80% of each continuous range (all but `glass` and `gold` for the
  material); `evaluation` is what training leaves out -- both ends of a two-sided range, the top
  end of a one-sided one; `full` is the whole range.
- A fixed value must lie inside the aspect's full range; anything else is an error, not a clamp.
- `viewpoint.fov` is in degrees about the wrist camera's own, which depends on the robot, so the
  registry is built for a rig's nominal fov (`registry(fov_degrees)`).

This module imports nothing but numpy, so configs can be validated without a simulator.
`DomainRandomizationMixin`-style per-episode draws are `Perturbations.draw`, seeded per
(class, aspect): fixing or randomizing one aspect never changes what another draws.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

SPLITS = ("train", "evaluation", "full")
RANDOMIZE = "randomize-"
DEFAULT = "default"
EVALUATION_FRACTION = 0.2
"""The part of each continuous range kept out of training."""

DEFAULT_FOV_DEGREES = 108.0
"""The Panda wrist camera's (`cameras.PANDA_RIG`)."""

TABLE_MATERIALS = ("default", "marble", "glass", "metal", "gold", "concrete", "rubber")
"""`table_material.MATERIALS` plus the stock wood; `tests/test_perturbation_classes.py` keeps them equal."""
EVALUATION_TABLE_MATERIALS = ("glass", "gold")
"""The two that look least like the rest: see-through, and a coloured metal."""
TRAIN_TABLE_MATERIALS = tuple(m for m in TABLE_MATERIALS if m not in EVALUATION_TABLE_MATERIALS)


@dataclass(frozen=True)
class Range:
    """One continuous aspect's full range, and how it splits between training and evaluation."""

    low: float
    high: float
    log: bool = False
    """Uniform in log(value) rather than in value: for a quantity that is a multiplier."""
    two_sided: bool = True
    """The neutral value is in the middle, so the evaluation part is both ends (half each); if
    False the neutral value is `low` and the evaluation part is the top end alone."""

    def bands(self, split: str) -> list[tuple[float, float]]:
        """The sub-intervals of [0, 1] (the range's own scale) that `split` draws from."""
        h = EVALUATION_FRACTION
        if split == "full":
            return [(0.0, 1.0)]
        if split == "train":
            return [(h / 2, 1 - h / 2)] if self.two_sided else [(0.0, 1 - h)]
        if split == "evaluation":
            return [(0.0, h / 2), (1 - h / 2, 1.0)] if self.two_sided else [(1 - h, 1.0)]
        raise ValueError(f"unknown split {split!r}, expected one of {', '.join(SPLITS)}")

    def sample(self, rng, split: str) -> float:
        """One value, uniform over the bands of `split` (so the two evaluation ends are equally
        likely)."""
        bands = self.bands(split)
        widths = np.array([b - a for a, b in bands])
        a, b = bands[int(rng.choice(len(bands), p=widths / widths.sum()))]
        u = float(rng.uniform(a, b))
        if self.log:
            return float(np.exp(np.log(self.low) + u * (np.log(self.high) - np.log(self.low))))
        return self.low + u * (self.high - self.low)

    def contains(self, value: float, split: str = "full") -> bool:
        if self.log:
            if value <= 0:
                return False
            u = (np.log(value) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        else:
            u = (value - self.low) / (self.high - self.low)
        return any(a - 1e-9 <= u <= b + 1e-9 for a, b in self.bands(split))


@dataclass(frozen=True)
class Aspect:
    """One knob of a perturbation class: continuous (`range`) or categorical (`choices`)."""

    name: str
    neutral: float | str
    range: Range | None = None
    choices: Mapping[str, tuple[str, ...]] | None = None
    """For a categorical aspect: the values each split draws from."""
    unit: str = ""

    def validate(self, cls: str, value) -> float | str:
        """`value` as this aspect's own type, or ValueError if it is not a legal fixed value."""
        where = f"{cls}.{self.name}"
        if self.choices is not None:
            if value not in self.choices["full"]:
                raise ValueError(
                    f"{where}={value!r} is not one of {', '.join(map(repr, self.choices['full']))}."
                )
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"{where}={value!r}: expected 'default', a number or 'randomize-<split>' "
                f"(split one of {', '.join(SPLITS)})."
            )
        value = float(value)
        if not self.range.contains(value):
            raise ValueError(
                f"{where}={value:g} is outside its range, {self.range.low:g} to {self.range.high:g}"
                f"{' ' + self.unit if self.unit else ''}."
            )
        return value

    def sample(self, rng, split: str) -> float | str:
        if self.choices is not None:
            pool = self.choices[split]
            return pool[int(rng.integers(len(pool)))]
        return self.range.sample(rng, split)


def _continuous(name, neutral, low, high, unit="", **kw) -> Aspect:
    return Aspect(name, neutral, Range(low, high, **kw), unit=unit)


def registry(fov_degrees: float = DEFAULT_FOV_DEGREES) -> dict[str, dict[str, Aspect]]:
    """Every perturbation class and its aspects. `fov_degrees` is the wrist camera's own field of
    view, the neutral of `viewpoint.fov`, whose range is a sixth of it either side."""
    c = _continuous
    spread = fov_degrees / 6
    table_materials = {
        "train": TRAIN_TABLE_MATERIALS,
        "evaluation": EVALUATION_TABLE_MATERIALS,
        "full": TABLE_MATERIALS,
    }
    return {
        "lighting": {a.name: a for a in (
            c("brightness", 1.0, 0.25, 8.0, "x the default's light levels", log=True),
            c("direction", 0.0, 0.0, 1.0, "0 = default's key light, 1 = from the side", two_sided=False),
            c("temperature", 0.0, -1.0, 1.0, "-1 = 2700 K, 0 = 6500 K, 1 = 12000 K"),
        )},
        "object_colour": {n: c(n, 0.0, -0.4, 0.4, "added to the channel, in [0, 1]") for n in "rgb"},
        "table_colour": {a.name: a for a in (
            *(c(n, 1.0, 0.6, 1.4, "x the table's channel") for n in "rgb"),
            c("brightness", 1.0, 0.4, 1.6, "x all three channels"),
        )},
        "table_texture": {a.name: a for a in (
            c("scale", 1.0, 0.5, 2.0, "x the texture's size", log=True),
            c("shear", 0.0, -0.3, 0.3, "x-shear of the texture plane"),
            c("rotation", 0.0, -45.0, 45.0, "degrees"),
            c("shift_x", 0.0, -0.5, 0.5, "periods of the texture"),
            c("shift_y", 0.0, -0.5, 0.5, "periods of the texture"),
        )},
        "table_material": {"material": Aspect("material", "default", choices=table_materials)},
        "viewpoint": {a.name: a for a in (
            c("left", 0.0, -4.0, 4.0, "cm across the image plane, + left"),
            c("up", 0.0, -3.0, 3.0, "cm across the image plane, + up"),
            c("forward", 0.0, -3.0, 3.0, "cm along the optical axis, + forward"),
            c("pitch", 0.0, -15.0, 15.0, "degrees, + tilts the view up"),
            c("yaw", 0.0, -15.0, 15.0, "degrees, + swings the view left"),
            c("roll", 0.0, -15.0, 15.0, "degrees, + rolls right side down"),
            c("fov", fov_degrees, fov_degrees - spread, fov_degrees + spread, "degrees"),
        )},
    }


CLASSES = tuple(registry())
"""The perturbation classes, in the order they are applied."""


@dataclass(frozen=True)
class Randomized:
    """An aspect drawn afresh every episode from `split`'s part of its range."""

    split: str

    def __str__(self) -> str:
        return RANDOMIZE + self.split


def _parse_aspect(cls: str, aspect: Aspect, spec):
    if spec is None or spec == DEFAULT:
        return aspect.neutral
    if isinstance(spec, str) and spec.startswith(RANDOMIZE):
        split = spec[len(RANDOMIZE):]
        if split not in SPLITS:
            raise ValueError(
                f"{cls}.{aspect.name}={spec!r}: expected 'randomize-' followed by one of "
                f"{', '.join(SPLITS)}."
            )
        return Randomized(split)
    return aspect.validate(cls, spec)


@dataclass(frozen=True)
class Perturbations:
    """A resolved perturbation config: for every aspect of every class, a fixed value or a
    `Randomized`."""

    values: Mapping[str, Mapping[str, float | str | Randomized]]
    fov_degrees: float = DEFAULT_FOV_DEGREES

    def __getitem__(self, cls: str) -> Mapping[str, float | str | Randomized]:
        return self.values[cls]

    def is_neutral(self, cls: str, name: str) -> bool:
        return self.values[cls][name] == registry(self.fov_degrees)[cls][name].neutral

    def is_default_class(self, cls: str) -> bool:
        return all(self.is_neutral(cls, n) for n in self.values[cls])

    @property
    def is_default(self) -> bool:
        return all(self.is_default_class(c) for c in self.values)

    @property
    def active(self) -> tuple[str, ...]:
        """The classes with any aspect off neutral, in application order."""
        return tuple(c for c in CLASSES if not self.is_default_class(c))

    def randomized(self, cls: str | None = None) -> dict[tuple[str, str], str]:
        """{(class, aspect): split} for every randomized aspect (of `cls`, if given)."""
        return {
            (c, n): v.split
            for c, aspects in self.values.items() if cls in (None, c)
            for n, v in aspects.items() if isinstance(v, Randomized)
        }

    @property
    def is_randomized(self) -> bool:
        """Whether anything is redrawn per episode, which needs the scene rebuilt every reset."""
        return bool(self.randomized())

    def draw(self, cls: str, seed: int) -> dict[str, float | str]:
        """`cls`'s aspects for the episode `seed`: the fixed ones as they are, the randomized
        ones drawn from their split. Each (class, aspect) has its own stream, so the draws are
        reproducible and independent of what else is fixed or randomized."""
        aspects = registry(self.fov_degrees)[cls]
        out = {}
        for name, value in self.values[cls].items():
            if isinstance(value, Randomized):
                rng = np.random.default_rng(
                    [int(seed) % 2**63, zlib.crc32(cls.encode()), zlib.crc32(name.encode())]
                )
                value = aspects[name].sample(rng, value.split)
            out[name] = value
        return out

    def to_config(self) -> dict[str, Any]:
        """The canonical plain-dict form (what gets recorded into an env's kwargs and any
        trajectory json): only the classes and aspects that are not neutral. Resolving it gives
        back an equal `Perturbations`."""
        out: dict[str, Any] = {}
        for cls in self.active:
            out[cls] = {
                n: str(v) if isinstance(v, Randomized) else v
                for n, v in self.values[cls].items() if not self.is_neutral(cls, n)
            }
        return out


def resolve_perturbations(
    config: Mapping[str, Any] | None, fov_degrees: float = DEFAULT_FOV_DEGREES
) -> Perturbations:
    """Resolve a user's perturbation config (see the module docstring) into a `Perturbations`,
    rejecting unknown classes, unknown aspects, malformed values and out-of-range fixed values
    early -- one that fell through would build an env that silently perturbs nothing."""
    classes = registry(fov_degrees)
    config = {} if config is None else dict(config)
    unknown = [c for c in config if c not in classes]
    if unknown:
        raise ValueError(
            f"Unknown perturbation class {unknown[0]!r}, expected any of "
            f"{', '.join(map(repr, classes))}."
        )
    values: dict[str, dict] = {}
    for cls, aspects in classes.items():
        spec = config.get(cls)
        if spec is None or isinstance(spec, str):
            values[cls] = {n: _parse_aspect(cls, a, spec) for n, a in aspects.items()}
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"{cls}={spec!r}: expected 'default', 'randomize-<split>' or a mapping of aspects.")
        bad = [n for n in spec if n not in aspects]
        if bad:
            raise ValueError(
                f"{cls} has no aspect {bad[0]!r}, expected any of {', '.join(map(repr, aspects))}."
            )
        values[cls] = {n: _parse_aspect(cls, a, spec.get(n)) for n, a in aspects.items()}
    return Perturbations(values, fov_degrees)
