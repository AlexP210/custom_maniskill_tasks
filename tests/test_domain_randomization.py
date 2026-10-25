"""`domain_randomization`: `python tests/test_domain_randomization.py`.

The ranges and the split need no simulator; the rest runs against the real one, physx_cpu at
num_envs=1, since sapien can only enable GPU PhysX once per process.
"""

import json

import numpy as np

from custom_maniskill_tasks import (
    AXES_RANDOMIZED,
    DOMAIN_RANDOMIZATION_RANGES as RANGES,
    HELDOUT_TABLE_MATERIALS,
    MATERIALS,
    canonical_domain_randomization,
    canonical_lighting,
    check_lighting,
    make_env,
)
from custom_maniskill_tasks.domain_randomization import (
    HELDOUT_FRACTION,
    TRAIN_TABLE_MATERIALS,
    draw_axis,
    lighting_preset,
    transform_pixels,
)

TASK = "PushCube-v1.1"
POKE = "PokeCube-v1.1"


def _raises(error, call, *args, **kwargs):
    try:
        call(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"{call.__name__}{args} did not raise {error.__name__}")


def _frame(obs):
    return np.asarray(obs["sensor_data"]["hand_camera"]["rgb"].cpu())[0].copy()  # a view of a render buffer


def test_a_range_is_split_into_training_and_held_out_with_no_overlap():
    rng = np.random.default_rng(0)
    for axis, parameters in RANGES.items():
        for name, r in parameters.items():
            train = [r.sample(rng, "train") for _ in range(400)]
            held = [r.sample(rng, "heldout") for _ in range(400)]
            assert all(r.contains(v, "train") and not r.contains(v, "heldout") for v in train), (axis, name)
            assert all(r.contains(v, "heldout") and not r.contains(v, "train") for v in held), (axis, name)
            assert all(min(r.low, r.high) <= v <= max(r.low, r.high) for v in train + held), (axis, name)
            # the held-out part is a fifth of the range, in its own scale
            width = sum(b - a for a, b in r.bands("heldout"))
            assert abs(width - HELDOUT_FRACTION) < 1e-9
            assert abs(sum(b - a for a, b in r.bands("train")) - (1 - HELDOUT_FRACTION)) < 1e-9
            if r.two_sided:  # both ends of a range with its neutral in the middle are held out
                middle = np.sqrt(r.low * r.high) if r.log else 0.5 * (r.low + r.high)
                assert min(held) < middle < max(held), (axis, name)
    direction = RANGES["lighting"]["direction"]
    assert not direction.two_sided and min(direction.sample(rng, "heldout") for _ in range(200)) >= 0.8
    brightness = RANGES["lighting"]["brightness"]
    assert brightness.contains(1.0, "train"), "the default's own brightness is in training"
    values = [brightness.sample(rng, "heldout") for _ in range(400)]
    assert min(values) < 0.4 and max(values) > 5, "both the dim and the bright end are held out"


def test_material_split_holds_out_a_couple_and_keeps_the_rest():
    assert len(HELDOUT_TABLE_MATERIALS) == 2 and set(HELDOUT_TABLE_MATERIALS) <= set(MATERIALS)
    assert not set(HELDOUT_TABLE_MATERIALS) & set(TRAIN_TABLE_MATERIALS)
    assert set(TRAIN_TABLE_MATERIALS) | set(HELDOUT_TABLE_MATERIALS) == {"default", *MATERIALS}
    rng = np.random.default_rng(0)
    train = {draw_axis("table_material", rng, "train")["material"] for _ in range(300)}
    held = {draw_axis("table_material", rng, "heldout")["material"] for _ in range(300)}
    assert train == set(TRAIN_TABLE_MATERIALS) and held == set(HELDOUT_TABLE_MATERIALS)


def test_every_axis_draws_every_parameter_and_the_lighting_stack_parses():
    rng = np.random.default_rng(1)
    for axis in AXES_RANDOMIZED:
        for split in ("train", "heldout"):
            drawn = draw_axis(axis, rng, split)
            assert set(drawn) == ({"material"} if axis == "table_material" else set(RANGES[axis]))
    for _ in range(50):
        condition = draw_axis("lighting", rng, "heldout")
        config = canonical_lighting(lighting_preset(condition))
        assert config.randomization is None and not config.domain_randomization
    # the neutral point is the default scene
    assert canonical_lighting(lighting_preset({"brightness": 1.0, "direction": 0.0, "temperature": 0.0})) \
        == canonical_lighting("default")
    _raises(ValueError, canonical_domain_randomization, ["bogus"])
    assert canonical_domain_randomization(["table_material", "lighting"]) == ("lighting", "table_material")
    assert canonical_domain_randomization(None) == canonical_domain_randomization([]) == ()


def test_texture_transform_is_the_identity_at_neutral_and_moves_the_texture_otherwise():
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)
    neutral = dict(scale=1.0, shear=0.0, rotation=0.0, shift_x=0.0, shift_y=0.0)
    assert np.array_equal(transform_pixels(pixels, **neutral), pixels)
    # a shift of a whole period is no shift, and a half-turn of a texture that is its own half-turn
    assert np.array_equal(transform_pixels(pixels, **{**neutral, "shift_x": 1.0}), pixels)
    for change in ({"scale": 1.5}, {"shear": 0.2}, {"rotation": 20.0}, {"shift_x": 0.25}):
        assert not np.array_equal(transform_pixels(pixels, **{**neutral, **change}), pixels), change
    shifted = transform_pixels(pixels, **{**neutral, "shift_x": 8 / 48})
    assert np.array_equal(shifted[:, 8:], pixels[:, :-8]), "an exact whole-pixel shift moves the columns"


def test_domain_randomization_is_recorded_redrawn_each_reset_and_reproducible():
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist",
                   domain_randomization=list(AXES_RANDOMIZED))
    base = env.unwrapped
    assert base.reconfiguration_freq == 1
    assert base.spec.kwargs["domain_randomization"] == list(AXES_RANDOMIZED)
    json.dumps(base.spec.kwargs)
    seen, frames = [], {}
    for seed in (1, 2, 3, 1):
        obs, _ = env.reset(seed=seed)
        drawn = base.episode_randomization
        assert set(drawn) == set(AXES_RANDOMIZED)
        for axis, parameters in drawn.items():
            if axis == "table_material":
                assert parameters["material"] in TRAIN_TABLE_MATERIALS
                assert base.table_material == parameters["material"]
            else:
                assert all(RANGES[axis][k].contains(v, "train") for k, v in parameters.items()), (axis, parameters)
        check_lighting(env, base.lighting)
        frame = _frame(obs)
        assert frame.dtype == np.uint8 and frame.shape[-1] == 3
        if seed in frames:
            assert np.array_equal(frames[seed], frame), "the same reset seed is the same condition, and nothing leaked"
            assert drawn == seen[seed - 1]
        frames[seed] = frame
        seen.append(drawn)
    assert len({json.dumps(d, sort_keys=True) for d in seen}) == 3
    env.close()


def test_each_axis_changes_the_image_and_heldout_draws_the_held_out_part():
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist")
    obs, _ = env.reset(seed=3)
    default = _frame(obs).astype(float)
    env.close()
    for axis in AXES_RANDOMIZED:
        for split, other in (("train", "heldout"), ("heldout", "train")):
            env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist",
                           domain_randomization=[axis], domain_randomization_split=split)
            obs, _ = env.reset(seed=3)
            drawn = env.unwrapped.episode_randomization
            assert set(drawn) == {axis}
            if axis == "table_material":
                material = drawn[axis]["material"]
                assert (material in HELDOUT_TABLE_MATERIALS) == (split == "heldout")
            else:
                assert all(RANGES[axis][k].contains(v, split) and not RANGES[axis][k].contains(v, other)
                           for k, v in drawn[axis].items()), (axis, split, drawn)
            if not (axis == "table_material" and drawn[axis]["material"] == "default"):
                assert np.abs(_frame(obs).astype(float) - default).mean() > 0.05, (axis, split)
            env.close()


def test_the_object_colour_is_what_changes_not_the_scene():
    from custom_maniskill_tasks.texture import _render_parts

    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist")
    env.reset(seed=3)
    stock = [tuple(p.material.base_color[:3]) for p in _render_parts(env.unwrapped.obj)]
    env.close()
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist",
                   domain_randomization=["object_colour"])
    env.reset(seed=3)
    offsets = [env.unwrapped.episode_randomization["object_colour"][c] for c in "rgb"]
    got = [tuple(p.material.base_color[:3]) for p in _render_parts(env.unwrapped.obj)]
    assert got == [tuple(np.clip(np.asarray(s) + offsets, 0, 1)) for s in stock] or np.allclose(
        got, [np.clip(np.asarray(s) + offsets, 0, 1) for s in stock], atol=1e-6)
    env.close()


def test_the_other_tasks_build_and_it_is_refused_where_it_cannot_work():
    for task in ("PickCube-v1.1", "LiftPegUpright-v1.1"):
        env = make_env(task, num_envs=1, sim_backend="physx_cpu", camera_view="wrist",
                       domain_randomization=list(AXES_RANDOMIZED))
        env.reset(seed=0)
        env.close()
    kwargs = dict(num_envs=1, sim_backend="physx_cpu")
    _raises(ValueError, make_env, TASK, domain_randomization=["bogus"], **kwargs)
    _raises(ValueError, make_env, TASK, domain_randomization_split="test", **kwargs)
    _raises(ValueError, make_env, TASK, domain_randomization=["lighting"], lighting="dim", **kwargs)
    _raises(ValueError, make_env, TASK, domain_randomization=["table_colour"], lighting="dark-table", **kwargs)
    _raises(ValueError, make_env, TASK, domain_randomization=["table_material"], table_material="gold", **kwargs)
    _raises(ValueError, make_env, TASK, domain_randomization=["lighting"], reconfiguration_freq=0, **kwargs)
    _raises(ValueError, make_env, POKE, domain_randomization=["object_colour"], **kwargs)  # no single object
    _raises(ValueError, make_env, "PushCube-v1", domain_randomization=["lighting"], **kwargs)  # a stock id
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist")  # off by default
    assert "domain_randomization" not in env.unwrapped.spec.kwargs
    assert env.unwrapped.reconfiguration_freq == 0 and env.unwrapped.episode_randomization == {}
    env.close()


if __name__ == "__main__":
    tests = [
        test_a_range_is_split_into_training_and_held_out_with_no_overlap,
        test_material_split_holds_out_a_couple_and_keeps_the_rest,
        test_every_axis_draws_every_parameter_and_the_lighting_stack_parses,
        test_texture_transform_is_the_identity_at_neutral_and_moves_the_texture_otherwise,
        test_domain_randomization_is_recorded_redrawn_each_reset_and_reproducible,
        test_each_axis_changes_the_image_and_heldout_draws_the_held_out_part,
        test_the_object_colour_is_what_changes_not_the_scene,
        test_the_other_tasks_build_and_it_is_refused_where_it_cannot_work,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
