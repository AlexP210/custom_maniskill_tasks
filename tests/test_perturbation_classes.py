"""`perturbation_classes`: `python tests/test_perturbation_classes.py`. No simulator needed, except
for the one test that checks the material list against `table_material`."""

import numpy as np

from custom_maniskill_tasks.perturbation_classes import (
    CLASSES,
    EVALUATION_FRACTION,
    EVALUATION_TABLE_MATERIALS,
    TABLE_MATERIALS,
    TRAIN_TABLE_MATERIALS,
    Randomized,
    registry,
    resolve_perturbations,
)


def _raises(error, call, *args, **kwargs):
    try:
        call(*args, **kwargs)
    except error as e:
        return str(e)
    raise AssertionError(f"{call.__name__}{args} did not raise {error.__name__}")


def test_default_and_none_and_empty_are_the_stock_scene():
    for config in (None, {}, {c: "default" for c in CLASSES}, {"lighting": {"brightness": "default"}}):
        p = resolve_perturbations(config)
        assert p.is_default and not p.is_randomized and p.active == () and p.to_config() == {}
        assert p["lighting"] == {"brightness": 1.0, "direction": 0.0, "temperature": 0.0}


def test_a_dict_sets_the_named_aspects_and_leaves_the_rest_neutral():
    p = resolve_perturbations({"lighting": {"temperature": "default", "brightness": 1.5}})
    assert p["lighting"] == {"brightness": 1.5, "direction": 0.0, "temperature": 0.0}
    assert p.active == ("lighting",) and not p.is_randomized
    assert p.to_config() == {"lighting": {"brightness": 1.5}}


def test_a_bare_class_string_applies_to_every_aspect():
    p = resolve_perturbations({"table_colour": "randomize-train"})
    assert set(p.randomized()) == {("table_colour", a) for a in ("r", "g", "b", "brightness")}
    assert set(p.randomized().values()) == {"train"}
    assert p.is_randomized and p.active == ("table_colour",)


def test_mixed_fixed_and_randomized_aspects():
    p = resolve_perturbations({"lighting": {"temperature": "randomize-train", "brightness": 1.5}})
    assert p.randomized() == {("lighting", "temperature"): "train"}
    drawn = p.draw("lighting", seed=3)
    assert drawn["brightness"] == 1.5 and drawn["direction"] == 0.0
    assert registry()["lighting"]["temperature"].range.contains(drawn["temperature"], "train")


def test_config_round_trips():
    config = {
        "lighting": {"temperature": "randomize-evaluation", "brightness": 1.5},
        "table_material": "randomize-full",
        "viewpoint": {"pitch": -5, "yaw": "randomize-train"},
    }
    p = resolve_perturbations(config)
    assert resolve_perturbations(p.to_config()) == p
    assert list(p.to_config()) == [c for c in CLASSES if c in config]


def test_unknown_names_and_malformed_values_are_rejected():
    assert "Unknown perturbation class 'corruption'" in _raises(ValueError, resolve_perturbations, {"corruption": "default"})
    assert "no aspect 'tint'" in _raises(ValueError, resolve_perturbations, {"lighting": {"tint": 1}})
    assert "randomize-" in _raises(ValueError, resolve_perturbations, {"lighting": "randomize-heldout"})
    assert "expected" in _raises(ValueError, resolve_perturbations, {"lighting": {"brightness": "bright"}})
    assert "expected" in _raises(ValueError, resolve_perturbations, {"lighting": {"brightness": True}})
    assert "mapping" in _raises(ValueError, resolve_perturbations, {"lighting": 1.5})
    assert "marble" in _raises(ValueError, resolve_perturbations, {"table_material": {"material": "velvet"}})


def test_fixed_values_are_checked_against_the_full_range_strictly():
    for config in (
        {"lighting": {"temperature": 1.5}}, {"lighting": {"brightness": 0.1}},
        {"lighting": {"brightness": 9}}, {"lighting": {"direction": -0.1}},
        {"table_colour": {"r": 1.41}}, {"viewpoint": {"pitch": 16}},
    ):
        assert "outside its range" in _raises(ValueError, resolve_perturbations, config)
    # the ends themselves are in, and a fixed value may lie in the evaluation part
    p = resolve_perturbations({"lighting": {"temperature": 1.0, "brightness": 8}, "viewpoint": {"pitch": -15}})
    assert p["lighting"]["temperature"] == 1.0 and p["viewpoint"]["pitch"] == -15.0


def test_splits_partition_every_range():
    rng = np.random.default_rng(0)
    for cls, aspects in registry().items():
        for name, aspect in aspects.items():
            if aspect.range is None:
                continue
            r = aspect.range
            train = [r.sample(rng, "train") for _ in range(300)]
            evaluation = [r.sample(rng, "evaluation") for _ in range(300)]
            full = [r.sample(rng, "full") for _ in range(300)]
            assert all(r.contains(v, "train") and not r.contains(v, "evaluation") for v in train), (cls, name)
            assert all(r.contains(v, "evaluation") and not r.contains(v, "train") for v in evaluation), (cls, name)
            assert all(min(r.low, r.high) <= v <= max(r.low, r.high) for v in train + evaluation + full)
            assert abs(sum(b - a for a, b in r.bands("evaluation")) - EVALUATION_FRACTION) < 1e-9
            assert r.contains(aspect.neutral, "train"), f"{cls}.{name}: the neutral is in training"
            if r.two_sided:  # full reaches into both held-out ends
                assert any(not r.contains(v, "train") for v in full), (cls, name)
                assert min(evaluation) < aspect.neutral < max(evaluation), (cls, name)
            else:
                assert min(evaluation) >= r.low + (1 - EVALUATION_FRACTION) * (r.high - r.low) - 1e-9


def test_material_splits():
    assert not set(EVALUATION_TABLE_MATERIALS) & set(TRAIN_TABLE_MATERIALS)
    assert set(EVALUATION_TABLE_MATERIALS) | set(TRAIN_TABLE_MATERIALS) == set(TABLE_MATERIALS)
    aspect = registry()["table_material"]["material"]
    rng = np.random.default_rng(0)
    for split, pool in (("train", TRAIN_TABLE_MATERIALS), ("evaluation", EVALUATION_TABLE_MATERIALS), ("full", TABLE_MATERIALS)):
        assert {aspect.sample(rng, split) for _ in range(400)} == set(pool), split
    assert resolve_perturbations({"table_material": {"material": "gold"}})["table_material"] == {"material": "gold"}


def test_material_list_matches_table_material():
    from custom_maniskill_tasks.table_material import DEFAULT_TABLE_MATERIAL, MATERIALS

    assert set(TABLE_MATERIALS) == {DEFAULT_TABLE_MATERIAL, *MATERIALS}


def test_draws_are_reproducible_and_independent_per_aspect():
    both = resolve_perturbations({"lighting": "randomize-train"})
    only_t = resolve_perturbations({"lighting": {"temperature": "randomize-train"}})
    a, b = both.draw("lighting", 7), both.draw("lighting", 7)
    assert a == b and a != both.draw("lighting", 8)
    # fixing the other aspects does not change what temperature draws
    assert only_t.draw("lighting", 7)["temperature"] == a["temperature"]
    # nor does another class's draw share a stream with it
    other = resolve_perturbations({"lighting": "randomize-train", "table_colour": "randomize-train"})
    assert other.draw("lighting", 7) == a


def test_fov_neutral_follows_the_rig():
    panda = resolve_perturbations({"viewpoint": {"fov": 100}})
    assert panda["viewpoint"]["fov"] == 100.0
    assert "outside its range" in _raises(ValueError, resolve_perturbations, {"viewpoint": {"fov": 108}}, fov_degrees=52.5)
    zed = resolve_perturbations({"viewpoint": {"fov": 50}}, fov_degrees=52.5)
    assert zed.to_config() == {"viewpoint": {"fov": 50.0}}
    assert resolve_perturbations({"viewpoint": {"fov": 52.5}}, fov_degrees=52.5).is_default


def test_randomized_str_and_eq():
    assert str(Randomized("train")) == "randomize-train"
    assert resolve_perturbations({"lighting": {"brightness": "randomize-full"}})["lighting"]["brightness"] == Randomized("full")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
