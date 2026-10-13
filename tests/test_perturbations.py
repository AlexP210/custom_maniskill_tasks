"""The appearance kwargs of the `-v1.1` tasks other than `lighting` and `distractors`
(`corruption`, `viewpoint`, `texture`): `python tests/test_perturbations.py`.

Against the real simulator, physx_cpu at num_envs=1, since sapien can only enable GPU PhysX once
per process.
"""

import json

import numpy as np
import torch

from custom_maniskill_tasks import (
    CorruptionConfig,
    canonical_corruption,
    canonical_lighting,
    canonical_table_material,
    canonical_texture,
    canonical_viewpoint,
    corrupt_images,
    check_lighting,
    check_viewpoint,
    make_env,
    randomization_pool,
)

TASK = "PushCube-v1.1"
POKE = "PokeCube-v1.1"


def _hand_frame(task=TASK, **kwargs):
    env = make_env(task, num_envs=1, sim_backend="physx_cpu", camera_view="wrist", **kwargs)
    obs, _ = env.reset(seed=7)
    frame = np.asarray(obs["sensor_data"]["hand_camera"]["rgb"].cpu())[0]
    spec_kwargs = dict(env.unwrapped.spec.kwargs)
    env.close()
    return frame, spec_kwargs


def _raises(error, call, *args, **kwargs):
    try:
        call(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"{call.__name__}{args}{kwargs} did not raise {error.__name__}")


# ---- corruption ----------------------------------------------------------------------------

def test_corruption_parses_and_rejects():
    assert canonical_corruption("default").is_default and canonical_corruption(None).is_default
    assert canonical_corruption("noise-0.1+blur-2") == CorruptionConfig(blur=2.0, noise=0.1)
    assert canonical_corruption(["lowres-4"]) == CorruptionConfig(lowres=4.0)
    for bad in ("blur", "blur-", "blur--2", "jpeg-50", "lowres-0.5", "sharpen-2"):
        _raises(ValueError, canonical_corruption, bad)


def test_corrupt_images_keeps_shape_and_dtype_and_does_what_it_says():
    rng = np.random.RandomState(0)
    images = torch.from_numpy(rng.randint(0, 256, size=(2, 32, 32, 3)).astype(np.uint8))

    assert corrupt_images(images, CorruptionConfig()) is images, "no corruption is no copy"
    for config in (CorruptionConfig(blur=2.0), CorruptionConfig(lowres=4.0), CorruptionConfig(noise=0.1)):
        out = corrupt_images(images, config, torch.Generator().manual_seed(0))
        assert out.shape == images.shape and out.dtype == torch.uint8, config

    blurred = corrupt_images(images, CorruptionConfig(blur=2.0)).float()
    assert blurred.std() < 0.5 * images.float().std(), "blur removes high-frequency content"
    small = corrupt_images(images, CorruptionConfig(lowres=8.0)).float()
    assert small.std() < 0.5 * images.float().std()

    flat = torch.full((1, 16, 16, 3), 128, dtype=torch.uint8)
    noisy = corrupt_images(flat, CorruptionConfig(noise=0.1), torch.Generator().manual_seed(1)).float()
    assert abs(noisy.std().item() - 25.5) < 4.0, "noise=0.1 is a std of 25.5 grey levels"
    again = corrupt_images(flat, CorruptionConfig(noise=0.1), torch.Generator().manual_seed(1)).float()
    assert torch.equal(noisy, again), "the same generator seed, the same noise"


def test_corruption_reaches_the_observation_and_is_recorded():
    clean, clean_kwargs = _hand_frame()
    assert "corruption" not in clean_kwargs
    blurred, kwargs = _hand_frame(corruption="blur-3")
    assert kwargs["corruption"] == "blur-3" and blurred.shape == clean.shape
    assert blurred.astype(float).std() < clean.astype(float).std() or not np.array_equal(blurred, clean)
    assert not np.array_equal(blurred, clean)
    first, _ = _hand_frame(corruption="noise-0.1")
    second, _ = _hand_frame(corruption="noise-0.1")
    assert np.array_equal(first, second), "same reset seed, same noise"
    assert not np.array_equal(first, clean)


# ---- viewpoint -----------------------------------------------------------------------------

def test_viewpoint_parses_and_rejects():
    assert canonical_viewpoint("default").is_default and canonical_viewpoint(None).is_default
    both = canonical_viewpoint("left-2+right-2")
    assert np.allclose(both.translation, 0.0), "opposite shifts cancel"
    shifted = canonical_viewpoint("back-3+pitch-up-5+fov-90")
    assert np.allclose(shifted.translation, (-0.03, 0.0, 0.0))
    assert np.isclose(shifted.pitch, -np.deg2rad(5)) and np.isclose(shifted.fov, np.pi / 2)
    for bad in ("left", "left-x", "sideways-3", "fov-5", "fov-200", "pitch-3"):
        _raises(ValueError, canonical_viewpoint, bad)


def test_viewpoint_moves_the_wrist_camera_and_is_recorded():
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist", viewpoint="back-3+fov-70")
    camera = env.unwrapped._sensors["hand_camera"]
    assert np.isclose(camera.config.fov, np.deg2rad(70))
    assert env.unwrapped.spec.kwargs["viewpoint"] == "back-3+fov-70"
    json.dumps(env.unwrapped.spec.kwargs)
    env.close()

    clean, _ = _hand_frame()
    moved, _ = _hand_frame(viewpoint="left-4")
    assert not np.array_equal(clean, moved)
    zero, kwargs = _hand_frame(viewpoint="default")
    assert np.array_equal(clean, zero) and "viewpoint" not in kwargs

    _raises(ValueError, make_env, TASK, num_envs=1, sim_backend="physx_cpu", viewpoint="left-4")  # not wrist


# ---- table_material ------------------------------------------------------------------------

def test_table_material_changes_the_table_only_visually():
    assert canonical_table_material(None) == "default"
    _raises(ValueError, canonical_table_material, "banana")
    clean, clean_kwargs = _hand_frame()
    assert "table_material" not in clean_kwargs
    for material in ("marble", "glass", "metal"):
        frame, kwargs = _hand_frame(table_material=material)
        assert kwargs["table_material"] == material and not np.array_equal(frame, clean), material
    _raises(ValueError, _hand_frame, table_material="banana")


def test_table_material_leaves_contact_alone():
    def friction(**kwargs):
        env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", **kwargs)
        env.reset(seed=7)
        shapes = env.unwrapped.table_scene.table._objs[0].find_component_by_type(
            __import__("sapien").physx.PhysxRigidBaseComponent
        ).collision_shapes
        out = [(m.static_friction, m.dynamic_friction, m.restitution)
               for m in (s.physical_material for s in shapes)]
        env.close()
        return out
    assert friction() == friction(table_material="glass")


# ---- texture -------------------------------------------------------------------------------

def test_texture_parses_and_rejects():
    assert canonical_texture("default") == () and canonical_texture(None) == ()
    assert canonical_texture("table-checker+cube-dots-4") == (
        ("table", "checker", 8), ("cube", "dots", 4),
    )
    for bad in ("checker", "table-plaid", "table-checker-0", "table_checker"):
        _raises(ValueError, canonical_texture, bad)


def test_texture_changes_the_target_and_only_the_target():
    clean, clean_kwargs = _hand_frame()
    assert "texture" not in clean_kwargs
    table, kwargs = _hand_frame(texture="table-checker")
    assert kwargs["texture"] == "table-checker" and not np.array_equal(table, clean)
    obj, _ = _hand_frame(texture="object-stripes")
    assert not np.array_equal(obj, clean)
    # object-stripes touches the cube and the goal target's pixels only where the cube is: the
    # far corners of the image are wood in both
    corner = (slice(0, 20), slice(0, 20))
    assert np.array_equal(obj[corner], clean[corner]), "an object texture leaves the table alone"
    by_name, _ = _hand_frame(texture="cube-stripes")
    assert np.array_equal(by_name, obj), "'object' on PushCube is the actor named cube"


def test_a_named_actor_beats_its_class_and_bad_targets_raise():
    class_only, _ = _hand_frame(texture="object-dots")
    named, _ = _hand_frame(texture="object-dots+cube-checker")
    named_first, _ = _hand_frame(texture="cube-checker+object-dots")
    assert np.array_equal(named, named_first), "specificity, not order"
    assert not np.array_equal(named, class_only)
    _raises(ValueError, _hand_frame, texture="banana-checker")
    _raises(ValueError, _hand_frame, texture="distractors-checker")  # there are none
    _raises(ValueError, _hand_frame, task=POKE, texture="object-checker")  # no single object


def test_perturbations_stack_with_lighting_and_each_other():
    everything = dict(
        lighting="dim+object-hue-30", texture="table-stripes+object-checker", distractors=2,
        viewpoint="up-2+pitch-down-4", corruption="blur-1+noise-0.02",
    )
    frame, kwargs = _hand_frame(**everything)
    assert {k: kwargs[k] for k in everything} == everything
    assert frame.dtype == np.uint8 and frame.shape[-1] == 3
    json.dumps(kwargs)


# ---- randomize -----------------------------------------------------------------------------

EVERY = ("lighting", "texture", "table_material", "distractors", "viewpoint", "corruption")


def _applied(env, kwarg, condition):
    """That `condition` is what the env is now built with."""
    base = env.unwrapped
    if kwarg == "lighting":
        assert base.lighting == canonical_lighting(condition)
        check_lighting(env, base.lighting)
    elif kwarg == "texture":
        assert base.texture == canonical_texture(condition)
    elif kwarg == "table_material":
        assert base.table_material == canonical_table_material(condition)
    elif kwarg == "distractors":
        assert len(base.distractors) == condition
    elif kwarg == "viewpoint":
        assert base.viewpoint == canonical_viewpoint(condition)
        check_viewpoint(env, base.viewpoint)
    else:
        assert base.corruption == canonical_corruption(condition)


def test_randomize_draws_every_kwarg_from_its_pool_on_each_reset_and_applies_it():
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist", **dict.fromkeys(EVERY, "randomize"))
    assert env.unwrapped.reconfiguration_freq == 1
    assert {k: env.unwrapped.spec.kwargs[k] for k in EVERY} == dict.fromkeys(EVERY, "randomize")
    json.dumps(env.unwrapped.spec.kwargs)
    seen = {k: set() for k in EVERY}
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        drawn = env.unwrapped.episode_perturbation
        assert set(drawn) == set(EVERY)
        for kwarg, condition in drawn.items():
            assert condition in randomization_pool(kwarg), (kwarg, condition)
            _applied(env, kwarg, condition)
            seen[kwarg].add(condition)
        frame = np.asarray(obs["sensor_data"]["hand_camera"]["rgb"].cpu())
        assert frame.dtype == np.uint8 and frame.shape[-1] == 3
    assert all(len(v) > 1 for v in seen.values()), f"a kwarg never changed: {seen}"
    env.close()


def test_randomize_is_reproducible_from_the_reset_seed():
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist", **dict.fromkeys(EVERY, "randomize"))
    first, _ = env.reset(seed=4)
    drawn = env.unwrapped.episode_perturbation
    env.reset(seed=5)
    second, _ = env.reset(seed=4)
    assert env.unwrapped.episode_perturbation == drawn
    a, b = (np.asarray(o["sensor_data"]["hand_camera"]["rgb"].cpu()) for o in (first, second))
    assert np.array_equal(a, b)
    env.close()


def test_randomize_one_kwarg_leaves_the_others_alone():
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist",
                   viewpoint="randomize", lighting="dim", corruption="blur-1")
    for seed in range(6):
        env.reset(seed=seed)
        assert set(env.unwrapped.episode_perturbation) == {"viewpoint"}
        _applied(env, "viewpoint", env.unwrapped.episode_perturbation["viewpoint"])
        assert env.unwrapped.lighting == canonical_lighting("dim")
        assert env.unwrapped.corruption == canonical_corruption("blur-1")
    env.close()


def test_randomized_corruption_alone_does_not_reconfigure():
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist", corruption="randomize")
    assert env.unwrapped.reconfiguration_freq == 0
    drawn = set()
    for seed in range(12):
        _, info = env.reset(seed=seed)
        assert not info["reconfigure"] or seed == 0
        condition = env.unwrapped.episode_perturbation["corruption"]
        _applied(env, "corruption", condition)
        drawn.add(condition)
    assert len(drawn) > 1
    env.close()


def test_randomize_is_refused_where_it_cannot_work():
    kwargs = dict(num_envs=1, sim_backend="physx_cpu")
    _raises(ValueError, make_env, TASK, viewpoint="randomize", **kwargs)  # not the wrist camera
    _raises(ValueError, make_env, "PushCube-v1", lighting="randomize", **kwargs)  # a stock id
    _raises(ValueError, make_env, TASK, camera_view="wrist", lighting="randomize", reconfiguration_freq=0, **kwargs)


def test_randomized_textures_skip_object_patterns_where_there_is_no_object():
    env = make_env(POKE, num_envs=1, sim_backend="physx_cpu", camera_view="wrist", texture="randomize")
    for seed in range(20):
        env.reset(seed=seed)
        assert not env.unwrapped.episode_perturbation["texture"].startswith("object-")
    env.close()


if __name__ == "__main__":
    tests = [
        test_corruption_parses_and_rejects,
        test_corrupt_images_keeps_shape_and_dtype_and_does_what_it_says,
        test_corruption_reaches_the_observation_and_is_recorded,
        test_viewpoint_parses_and_rejects,
        test_viewpoint_moves_the_wrist_camera_and_is_recorded,
        test_table_material_changes_the_table_only_visually,
        test_table_material_leaves_contact_alone,
        test_texture_parses_and_rejects,
        test_texture_changes_the_target_and_only_the_target,
        test_a_named_actor_beats_its_class_and_bad_targets_raise,
        test_perturbations_stack_with_lighting_and_each_other,
        test_randomize_draws_every_kwarg_from_its_pool_on_each_reset_and_applies_it,
        test_randomize_is_reproducible_from_the_reset_seed,
        test_randomize_one_kwarg_leaves_the_others_alone,
        test_randomized_corruption_alone_does_not_reconfigure,
        test_randomize_is_refused_where_it_cannot_work,
        test_randomized_textures_skip_object_patterns_where_there_is_no_object,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
