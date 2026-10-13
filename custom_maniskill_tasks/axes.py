"""Which visual perturbations can be dialled up, and how: the registry sweeps are built from.

An *axis* is one continuous (or counted) knob of a perturbation kwarg -- `blur` sigma, brightness
scale, camera pitch -- with the keyword string that sets it, the value at which it changes nothing
(`neutral`) and a default grid of values to sweep. Sweep scripts ask `conditions(axis_name)` for
what to run instead of hard-coding preset strings, and `python -m custom_maniskill_tasks.axes`
prints the registry.

This module is standard-library only and has no import of the rest of the package, so a sweep
driver on a host without torch or ManiSkill can load it by file path (see
`analysis/visual_robustness/run_sweep.py`). The parsers are imported lazily, by `resolve`, which
only the tests need. What a driver needs to know without them -- which values an axis accepts --
is on the axis as `lo` / `hi` and `valid`, and `tests/test_axes.py` checks those against the
parsers.

`tests/test_axes.py` keeps this honest: every keyword the parsers accept is on an axis here or
listed in `NOT_GRADED` with the reason it has no scale, every kwarg the perturbation mixins take is
either on an axis or in `NOT_GRADED`, every grid value parses, and every axis's neutral value
resolves to the default scene. A new perturbation that is not classified fails the test.

Signed axes (`viewpoint.pitch`, ...) take a negative value for the opposite keyword, because the
keyword grammar itself has no signs: `-5` on `viewpoint.pitch` is `pitch-down-5`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable


def _num(v: float) -> str:
    return f"{v:g}"


def _even(low: float, high: float, n: int = 6) -> tuple[float, ...]:
    """`n` points evenly spaced from `low` to `high`, both included: the default way an axis's grid
    is set. An open lower end, "(0, 2]", is `_even(0.01, 2)`."""
    return tuple(round(low + (high - low) * i / (n - 1), 6) for i in range(n))


def _count(v: float) -> int:
    if v != int(v):
        raise ValueError(f"distractors is a whole number of objects, not {v!r}.")
    return int(v)


def _signed(positive: str, negative: str) -> Callable[[float], str]:
    return lambda v: f"{positive}-{_num(v)}" if v >= 0 else f"{negative}-{_num(-v)}"


@dataclass(frozen=True)
class Axis:
    """One graded knob of a perturbation kwarg."""

    name: str
    """`<family>.<knob>`, e.g. `corruption.blur`."""
    kwarg: str
    """The env kwarg it is passed through."""
    condition: Callable[[float], str | int]
    """The kwarg value that sets the knob to `v` and touches nothing else."""
    unit: str
    neutral: float
    """The value at which the scene is the default one; the sweep's origin. It need not be on `grid`:
    the unperturbed `default` run is run once and drawn at this value whatever the grid is."""
    grid: tuple[float, ...]
    """A default set of values to sweep, in the order they should be plotted."""
    scale: str = "linear"
    """How the grid is spaced: `"linear"` or `"log"` (for plotting)."""
    kind: str = "monotonic"
    """`"monotonic"`: severity rises away from neutral. `"cyclic"`: wraps round (hue), so plot
    it as a circle or a bar chart, not a line. `"count"`: a whole number."""
    keywords: tuple[str, ...] = ()
    """The grammar keywords this axis covers, which `NOT_GRADED` and the other axes must not."""
    note: str = ""
    lo: float | None = None
    hi: float | None = None
    """The values the parser accepts, inclusive, or None for no limit that side. Checked against
    the parser by `tests/test_axes.py`, so a driver can skip an impossible value without it."""

    def valid(self, v: float) -> str | None:
        """Why the parser would refuse `v`, or None if it would take it."""
        if self.kind == "count" and v != int(v):
            return f"{self.name} is a whole number of objects, not {v:g}"
        if (self.lo is not None and v < self.lo) or (self.hi is not None and v > self.hi):
            return f"{self.name} takes {self.lo if self.lo is not None else '-inf'} to {self.hi if self.hi is not None else 'inf'}, not {v:g}"
        return None

    def __post_init__(self):
        assert self.name.split(".")[0] in FAMILIES and self.kind in ("monotonic", "cyclic", "count")
        assert self.scale in ("linear", "log")


FAMILIES = ("lighting", "corruption", "viewpoint", "distractors")

AXES: dict[str, Axis] = {a.name: a for a in (
    Axis("lighting.brightness", "lighting", lambda v: f"bright-set-{_num(round(0.3 * v, 4))}-{_num(v)}",
         "x default level", 1, (0.25, 0.5, 1, 2.5, 4, 8), scale="log",
         keywords=("bright-set",), lo=0, note="ambient 0.3k and both lights k"),
    Axis("lighting.warm", "lighting", lambda v: f"warm-set-{_num(v)}", "0 = 6500 K, 1 = 2700 K",
         0, (0, 0.2, 0.4, 0.6, 0.8, 1), keywords=("warm-set",), lo=0, hi=1.6, note="constant luminance; the parser allows up to 1.6"),
    Axis("lighting.cool", "lighting", lambda v: f"cool-set-{_num(v)}", "0 = 6500 K, 1 = 12000 K",
         0, (0, 0.2, 0.4, 0.6, 0.8, 1), keywords=("cool-set",), lo=0, hi=1.6, note="constant luminance; the parser allows up to 1.6"),
    Axis("lighting.side", "lighting", lambda v: f"side-set-{_num(v)}", "0 = default, 1 = side",
         0, (0, 0.2, 0.4, 0.6, 0.8, 1), keywords=("side-set",), lo=0, hi=1, note="key light direction"),
    Axis("lighting.table", "lighting", lambda v: f"table-set-{_num(v)}", "x table colour",
         1, _even(0.01, 2), keywords=("table-set",), lo=0, note="grey scale; the r-g-b form is not on the axis"),
    Axis("lighting.object_hue", "lighting", lambda v: f"object-hue-{_num(v)}", "degrees",
         0, _even(0, 60), kind="cyclic", keywords=("object-hue",), lo=0, hi=359.99,
         note="0 and 360 are the same; 180 is the far end"),

    Axis("corruption.blur", "corruption", lambda v: f"blur-{_num(v)}", "px sigma",
         0, (0, 0.2, 0.4, 0.6, 0.8, 1), keywords=("blur",), lo=0),
    Axis("corruption.noise", "corruption", lambda v: f"noise-{_num(v)}", "std, fraction of full scale",
         0, (0, 0.1, 0.2, 0.3, 0.4, 0.5), keywords=("noise",), lo=0, note="0.05 is about 13 grey levels"),
    Axis("corruption.lowres", "corruption", lambda v: f"lowres-{_num(v)}", "x downscale factor",
         1, _even(1, 6), scale="log", keywords=("lowres",), lo=1),

    Axis("viewpoint.lateral", "viewpoint", _signed("left", "right"), "cm, + left",
         0, _even(-6, 3), keywords=("left", "right")),
    Axis("viewpoint.vertical", "viewpoint", _signed("up", "down"), "cm, + up",
         0, _even(-1, 5), keywords=("up", "down")),
    Axis("viewpoint.depth", "viewpoint", _signed("forward", "back"), "cm, + forward",
         0, _even(-3, 3), keywords=("forward", "back"), note="large `back` shows the hand mount"),
    Axis("viewpoint.pitch", "viewpoint", _signed("pitch-up", "pitch-down"), "degrees, + up",
         0, _even(-10, 20), keywords=("pitch-up", "pitch-down")),
    Axis("viewpoint.yaw", "viewpoint", _signed("yaw-left", "yaw-right"), "degrees, + left",
         0, _even(-20, 10), keywords=("yaw-left", "yaw-right")),
    Axis("viewpoint.roll", "viewpoint", _signed("roll-left", "roll-right"), "degrees, + left side down",
         0, _even(-15, 15), keywords=("roll-left", "roll-right")),
    Axis("viewpoint.fov", "viewpoint", lambda v: f"fov-{_num(v)}", "degrees",
         108, _even(90, 125), keywords=("fov",), lo=10, hi=170, note="allowed 10 to 170"),

    Axis("distractors.count", "distractors", _count, "YCB objects",
         0, (0, 1, 2, 3), kind="count", lo=0, hi=8, note="up to about 3 always fit"),
)}


NOT_GRADED: dict[str, str] = {
    "table_material": "categorical: seven named materials with no order between them",
    "texture": "categorical: pattern x target; the repeat count `n` changes spatial frequency only, "
               "and there is no value at which a pattern is no pattern",
    "lighting:named": "the named presets (`dim`, `side`, `shadows`...) are points, not scales; their "
                      "graded relatives are the axes above",
}
"""Perturbation kwargs (or `kwarg:part`) with no axis, and why. `tests/test_axes.py` requires every
kwarg the mixins take to be here or on an axis."""

@dataclass(frozen=True)
class Category:
    """A set of ungraded conditions of one kwarg: named alternatives with no scale between them."""

    name: str
    kwarg: str
    conditions: tuple[str, ...]
    """The kwarg values to evaluate, each a different scene from the default and from each other."""
    note: str = ""


CATEGORIES: dict[str, Category] = {c.name: c for c in (
    Category("table_material", "table_material", ("marble", "glass", "metal", "gold", "concrete", "rubber"),
             note="visual only; contact is identical under every material"),
    Category("texture.table", "texture", tuple(f"table-{p}" for p in ("checker", "stripes", "dots", "noise")),
             note="the table replaced by the pattern in a light and a dark tone"),
    Category("texture.object", "texture", tuple(f"object-{p}" for p in ("checker", "stripes", "dots", "noise")),
             note="the task's object keeps its colour and gains the pattern as a darkening of it"),
)}
"""The conditions the graded axes do not cover, from `NOT_GRADED`: the named table materials, and a pattern
on the table or on the object. None of the named lighting presets is here (`PRESETS_LEFT_OUT`). Texture on
the distractors is left out too (it needs `distractors` > 0, which is its own axis), and so are stacks of
conditions. `tests/test_axes.py` checks each against its parser, and that every material and every
pattern is here."""


PRESETS_LEFT_OUT = {
    "dim": "lighting.brightness", "very-dim": "lighting.brightness",
    "bright": "lighting.brightness", "very-bright": "lighting.brightness",
    "warm": "lighting.warm", "very-warm": "lighting.warm",
    "cool": "lighting.cool", "very-cool": "lighting.cool",
    "side": "lighting.side",
    "dark-table": "lighting.table", "very-dark-table": "lighting.table",
    "random": "the graded lighting axes (it draws a different condition per parallel env)",
    "shadows": "left out of the comparison by choice: no graded axis covers it",
}
"""The named lighting presets left out of the ungraded comparison, and why: a graded axis covers most of
them, and `shadows` was dropped. `tests/test_axes.py` requires every preset to be here or `default`, so a
new one has to be placed."""


RANDOMIZE = "randomize"
"""The value any perturbation kwarg takes to draw its condition afresh on every reset, from the runs
`run_sweep.py` and `run_ungraded.py` evaluate (see `randomization_pool`). `randomize.py` does the
drawing."""

DEFAULT_CONDITIONS: dict[str, str | int] = {
    "lighting": "default", "corruption": "default", "viewpoint": "default",
    "texture": "default", "table_material": "default", "distractors": 0,
}
"""Each perturbation kwarg's value that changes nothing."""


@functools.lru_cache(maxsize=None)
def randomization_pool(kwarg: str) -> tuple[str | int, ...]:
    """The kwarg values `kwarg="randomize"` draws from, one per run the evaluation scripts make
    for that kwarg: the unperturbed scene once (`DEFAULT_CONDITIONS`), every non-neutral point on
    each of its axes' grids that the parser accepts (as `run_sweep.plan` does), and each condition
    of its ungraded categories (as `run_ungraded.plan` does). Each is equally likely, so an axis
    with more grid points is drawn more often, as it is run more often."""
    if kwarg not in DEFAULT_CONDITIONS:
        raise KeyError(f"{kwarg!r} is not a perturbation kwarg, expected one of {', '.join(DEFAULT_CONDITIONS)}")
    pool = [DEFAULT_CONDITIONS[kwarg]]
    for a in AXES.values():
        if a.kwarg == kwarg:
            pool += [a.condition(v) for v in a.grid if v != a.neutral and a.valid(v) is None]
    for category in CATEGORIES.values():
        if category.kwarg == kwarg:
            pool += category.conditions
    return tuple(pool)


def draw_condition(kwarg: str, rng, accept: Callable[[str | int], bool] | None = None) -> str | int:
    """One value from `randomization_pool(kwarg)`, uniformly, using `rng` (anything with numpy's
    `Generator.integers`). `accept` drops the values a particular env cannot take (`object-` textures
    on a task with no single object)."""
    pool = randomization_pool(kwarg)
    if accept is not None:
        pool = tuple(c for c in pool if accept(c))
    return pool[int(rng.integers(len(pool)))]


def resolve(kwarg: str, value):
    """`value` as the config the env would build from it, for comparing two spellings. Imports the
    parsers, and so torch and ManiSkill, which nothing else in this module does."""
    from custom_maniskill_tasks.corruption import canonical_corruption
    from custom_maniskill_tasks.distractors import canonical_distractors
    from custom_maniskill_tasks.lighting import canonical_lighting
    from custom_maniskill_tasks.table_material import canonical_table_material
    from custom_maniskill_tasks.texture import canonical_texture
    from custom_maniskill_tasks.viewpoint import canonical_viewpoint

    return {
        "lighting": canonical_lighting, "corruption": canonical_corruption,
        "viewpoint": canonical_viewpoint, "distractors": canonical_distractors,
        "table_material": canonical_table_material, "texture": canonical_texture,
    }[kwarg](value)


def axis(name: str) -> Axis:
    try:
        return AXES[name]
    except KeyError:
        raise KeyError(f"Unknown axis {name!r}, expected one of {', '.join(AXES)}") from None


def conditions(name: str, values=None) -> list[tuple[float, str | int]]:
    """`(value, kwarg value)` for each of `values` (default: the axis's grid), in order. The kwarg
    value goes straight into `make_env(..., **{axis(name).kwarg: it})`."""
    a = axis(name)
    return [(v, a.condition(v)) for v in (a.grid if values is None else values)]


def graded_kwargs() -> tuple[str, ...]:
    """The env kwargs that have at least one axis."""
    return tuple(dict.fromkeys(a.kwarg for a in AXES.values()))


def markdown_table() -> str:
    """The registry as the table the README shows."""
    rows = ["| axis | kwarg | neutral | unit | default grid | notes |", "| --- | --- | --- | --- | --- | --- |"]
    for a in AXES.values():
        grid = ", ".join(f"{v:g}" for v in a.grid) + (" (log)" if a.scale == "log" else "")
        note = "; ".join(x for x in (a.kind if a.kind != "monotonic" else "", a.note) if x)
        rows.append(f"| `{a.name}` | `{a.kwarg}` | {a.neutral:g} | {a.unit} | {grid} | {note} |")
    rows += ["", "No axis: " + "; ".join(f"`{k}` ({why})" for k, why in NOT_GRADED.items()) + ".",
             "", "Ungraded conditions (`CATEGORIES`, run by `run_ungraded.py`):", ""]
    rows += [f"- `{c.name}` (`{c.kwarg}`): {', '.join(f'`{x}`' for x in c.conditions)}" for c in CATEGORIES.values()]
    return "\n".join(rows)


if __name__ == "__main__":
    print(markdown_table())
