"""The graded-axis registry (`custom_maniskill_tasks/axes.py`) against the parsers it describes:
`python tests/test_axes.py`. No simulator needed.
"""

import inspect
import re
from pathlib import Path

import numpy as np

from custom_maniskill_tasks import LIGHTING_PRESETS, MATERIALS, PATTERNS, PerturbationsMixin, LightingMixin, corruption, lighting, viewpoint
from custom_maniskill_tasks import axes as axes_module
from custom_maniskill_tasks.axes import (
    AXES, CATEGORIES, NOT_GRADED, PRESETS_LEFT_OUT, conditions, graded_kwargs,
    markdown_table, resolve,
)

README = Path(__file__).resolve().parent.parent / "README.md"
PLUMBING = {"sensor_configs"}  # kwargs the mixins take for each other, not perturbations
TABLE_MARKERS = ("<!-- axes:start -->", "<!-- axes:end -->")


def _normal(kwarg, config):
    """`config` with the spellings of one scene made equal: a fov given outright at the default's
    value is the default fov."""
    if kwarg == "viewpoint":
        return (config.translation, config.yaw, config.pitch, config.roll, round(config.field_of_view, 6))
    return config


def test_every_mixin_kwarg_is_graded_or_explained():
    taken = set()
    for cls in (*PerturbationsMixin.__mro__, LightingMixin):
        init = cls.__dict__.get("__init__")
        if init is not None:
            taken |= {n for n, p in inspect.signature(init).parameters.items() if p.kind is p.KEYWORD_ONLY}
    taken -= PLUMBING
    assert taken >= {"lighting", "texture", "table_material", "distractors", "viewpoint", "corruption"}, taken
    unclassified = taken - set(graded_kwargs()) - set(NOT_GRADED)
    assert not unclassified, f"perturbation kwargs on no axis and not in NOT_GRADED: {sorted(unclassified)}"
    assert not set(graded_kwargs()) - taken, "an axis for a kwarg no mixin takes"


def _names_in(pattern: re.Pattern) -> set[str]:
    group = re.search(r"\(\?P<name>([^)]*)\)", pattern.pattern)
    return set(group.group(1).split("|"))


def test_every_parser_keyword_is_on_an_axis():
    covered = {k for a in AXES.values() for k in a.keywords}
    for module, pattern in ((corruption, corruption._KEYWORD), (viewpoint, viewpoint._AXIS)):
        missing = _names_in(pattern) - covered
        assert not missing, f"{module.__name__} keywords on no axis: {sorted(missing)}"
    # the lighting sliders are one regex each, named `_<SLIDER>_SET` / `_OBJECT_HUE`; a new one is a
    # new module-level pattern, which has to be claimed by an axis keyword
    sliders = {n: p for n, p in vars(lighting).items() if isinstance(p, re.Pattern)}
    assert {"_BRIGHT_SET", "_SIDE_SET", "_HUE_SET", "_TABLE_SET", "_OBJECT_HUE"} <= set(sliders)
    for name in sliders:
        stem = name.strip("_").lower().replace("_", "-")
        claimed = any(k == stem or (name == "_HUE_SET" and k in ("warm-set", "cool-set")) for k in covered)
        assert claimed, f"lighting.{name} is a slider with no axis (keyword {stem!r})"
    stale = covered - _names_in(corruption._KEYWORD) - _names_in(viewpoint._AXIS) - {
        k for n in sliders for k in [n.strip("_").lower().replace("_", "-")] + ["warm-set", "cool-set"]}
    assert not stale, f"axis keywords no parser has: {sorted(stale)}"


def test_grids_parse_and_name_their_keywords():
    for a in AXES.values():
        for value, kwarg_value in conditions(a.name):
            resolve(a.kwarg, kwarg_value)  # raises on anything the parser would reject
            if isinstance(kwarg_value, str):
                assert any(kwarg_value.startswith(k + "-") for k in a.keywords), (a.name, kwarg_value)
        values = [v for v, _ in conditions(a.name)]
        assert len(set(values)) == len(values) >= 3, a.name


def test_neutral_value_is_the_default_scene():
    for a in AXES.values():
        neutral = _normal(a.kwarg, resolve(a.kwarg, a.condition(a.neutral)))
        default = _normal(a.kwarg, resolve(a.kwarg, "default" if a.kwarg != "distractors" else 0))
        assert neutral == default, f"{a.name}: neutral {a.neutral} is not the default scene"


def test_non_neutral_values_change_the_scene_and_are_distinct():
    for a in AXES.values():
        keys = []
        for value, kwarg_value in conditions(a.name):
            config = resolve(a.kwarg, kwarg_value)
            if value != a.neutral:
                default = resolve(a.kwarg, "default" if a.kwarg != "distractors" else 0)
                assert _normal(a.kwarg, config) != _normal(a.kwarg, default), (a.name, value)
            keys.append(repr(_normal(a.kwarg, config)))
        assert len(set(keys)) == len(keys), f"{a.name}: two grid values build the same scene"


def test_signed_axes_are_two_keywords_and_mirror_each_other():
    for name in ("viewpoint.lateral", "viewpoint.pitch", "viewpoint.depth"):
        a = AXES[name]
        plus, minus = a.condition(3), a.condition(-3)
        assert plus != minus and not minus.endswith("--3")
        cp, cm = resolve("viewpoint", plus), resolve("viewpoint", minus)
        assert np.allclose(np.array(cp.translation), -np.array(cm.translation))
        assert np.isclose(cp.pitch, -cm.pitch)


def test_a_count_axis_refuses_a_fraction_rather_than_rounding():
    for bad in (0.1, 2.5):
        try:
            AXES["distractors.count"].condition(bad)
        except ValueError:
            continue
        raise AssertionError(f"distractors.count accepted {bad}")
    assert AXES["distractors.count"].condition(3.0) == 3


def _accepted(a, v):
    try:
        resolve(a.kwarg, a.condition(v))
    except ValueError:
        return False
    return True


def test_declared_ranges_are_what_the_parsers_accept():
    for a in AXES.values():
        for v in [x for x in (a.lo, a.hi) if x is not None]:
            assert a.valid(v) is None and _accepted(a, v), f"{a.name}: the declared limit {v} is refused"
        if a.lo is not None:
            below = a.lo - 1
            assert a.valid(below) and not _accepted(a, below), f"{a.name}: {below} is below lo and accepted"
        if a.hi is not None:
            above = a.hi + 1
            assert a.valid(above) and not _accepted(a, above), f"{a.name}: {above} is above hi and accepted"
        for v in a.grid:
            assert a.valid(v) is None, f"{a.name}: grid value {v} is outside its own range"


def test_axes_module_needs_no_heavy_imports():
    import subprocess, sys
    code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('axes', %r); "
            "m = importlib.util.module_from_spec(s); sys.modules['axes'] = m; s.loader.exec_module(m); "
            "assert not {'torch', 'mani_skill', 'custom_maniskill_tasks'} & set(sys.modules), sorted(sys.modules)"
            % axes_module.__file__)
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_ungraded_conditions_parse_differ_and_are_complete():
    for category in CATEGORIES.values():
        default = resolve(category.kwarg, "default")
        configs = []
        for condition in category.conditions:
            config = resolve(category.kwarg, condition)  # raises on anything the parser would refuse
            assert config != default, f"{category.name}: {condition!r} is the default scene"
            configs.append(repr(config))
        assert len(set(configs)) == len(configs), f"{category.name}: two conditions build the same scene"
    by_name = {c.name: set(c.conditions) for c in CATEGORIES.values()}
    assert by_name["table_material"] == set(MATERIALS)
    assert not any(c.kwarg == "lighting" for c in CATEGORIES.values()), "no lighting preset is in the ungraded comparison"
    assert set(PRESETS_LEFT_OUT) == set(LIGHTING_PRESETS) - {"default"}, \
        "a lighting preset is not accounted for in PRESETS_LEFT_OUT"
    assert all(v in AXES or isinstance(v, str) for v in PRESETS_LEFT_OUT.values())
    for pattern in PATTERNS:
        assert f"table-{pattern}" in by_name["texture.table"] and f"object-{pattern}" in by_name["texture.object"]
    assert {c.kwarg for c in CATEGORIES.values()} <= {"lighting", "texture", "table_material"}


def test_readme_table_is_the_registry():
    text = README.read_text()
    start, end = (text.index(m) for m in TABLE_MARKERS)
    shown = text[start + len(TABLE_MARKERS[0]):end].strip()
    assert shown == markdown_table(), (
        "README's axis table is stale: run `python -m custom_maniskill_tasks.axes` and paste it "
        f"between {TABLE_MARKERS}"
    )


if __name__ == "__main__":
    tests = [
        test_every_mixin_kwarg_is_graded_or_explained,
        test_every_parser_keyword_is_on_an_axis,
        test_grids_parse_and_name_their_keywords,
        test_neutral_value_is_the_default_scene,
        test_non_neutral_values_change_the_scene_and_are_distinct,
        test_signed_axes_are_two_keywords_and_mirror_each_other,
        test_a_count_axis_refuses_a_fraction_rather_than_rounding,
        test_declared_ranges_are_what_the_parsers_accept,
        test_axes_module_needs_no_heavy_imports,
        test_ungraded_conditions_parse_differ_and_are_complete,
        test_readme_table_is_the_registry,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
