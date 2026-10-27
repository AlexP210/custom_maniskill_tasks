"""`perturbations` on a real env: `python tests/test_perturbation_mixin.py [test_name ...]`.

physx_cpu at num_envs=1 throughout. With no arguments each test runs in a subprocess of its own,
because sapien's Vulkan renderer stops finding its device after about eight scenes have been built
in one process.
"""

import subprocess
import sys

import numpy as np

import custom_maniskill_tasks  # noqa: F401  (registers the ids under test)
from custom_maniskill_tasks import make_env, perturbation_registry

TASK = "PushCube-v1.1"
POKE = "PokeCube-v1.1"


def _build(task=TASK, perturbations=None, **kwargs):
    return make_env(task, camera_view="wrist", sim_backend="physx_cpu", perturbations=perturbations, **kwargs)


def _frame(obs):
    return np.asarray(obs["sensor_data"]["hand_camera"]["rgb"].cpu())[0].astype(np.int32)  # a view of a render buffer


def _error(call, *args, **kwargs):
    try:
        call(*args, **kwargs)
    except ValueError as error:
        return str(error)
    raise AssertionError("did not raise ValueError")


def test_default_is_the_stock_scene_and_passes_nothing():
    stock = _build()
    reference = _frame(stock.reset(seed=0)[0])
    assert "perturbations" not in stock.unwrapped.spec.kwargs and stock.unwrapped.reconfiguration_freq == 0
    stock.close()
    spelled = _build(perturbations={c: "default" for c in perturbation_registry()})
    assert "perturbations" not in spelled.unwrapped.spec.kwargs, "all-default is not recorded"
    assert np.abs(_frame(spelled.reset(seed=0)[0]) - reference).max() == 0, "and renders the stock image"
    spelled.close()


def test_a_neutral_value_inside_a_class_is_the_stock_scene_too():
    stock = _build()
    reference = _frame(stock.reset(seed=0)[0])
    stock.close()
    env = _build(perturbations={"lighting": {"brightness": 1.0, "temperature": "default"}, "table_colour": {"r": 1.0}})
    assert "perturbations" not in env.unwrapped.spec.kwargs
    assert np.abs(_frame(env.reset(seed=0)[0]) - reference).max() == 0
    env.close()


def test_every_class_changes_the_image():
    stock = _build()
    reference = _frame(stock.reset(seed=0)[0])
    stock.close()
    for config in (
        {"lighting": {"brightness": 3.0}},
        {"lighting": {"temperature": -0.8}},
        {"lighting": {"direction": 1.0}},
        {"object_colour": {"r": 0.4, "g": -0.4}},
        {"table_colour": {"brightness": 0.5}},
        {"table_texture": {"rotation": 30, "scale": 2}},
        {"table_material": {"material": "marble"}},
        {"viewpoint": {"pitch": 10}},
        {"viewpoint": {"fov": 120}},
    ):
        env = _build(perturbations=config)
        assert env.unwrapped.reconfiguration_freq == 0, "fixed values build the scene once"
        shifted = _frame(env.reset(seed=0)[0])
        # a few pixels is enough: the cube is about 0.5% of the wrist image
        changed = np.abs(shifted - reference).max(-1)
        assert (changed > 8).sum() > 50, config
        assert env.unwrapped.spec.kwargs["perturbations"] == resolve_config(config), config
        env.close()


def resolve_config(config):
    from custom_maniskill_tasks import resolve_perturbations

    return resolve_perturbations(config).to_config()


def test_fixed_aspects_stay_while_randomized_ones_redraw_each_reset():
    env = _build(perturbations={"lighting": {"temperature": "randomize-train", "brightness": 1.5}})
    assert env.unwrapped.reconfiguration_freq == 1, "randomizing rebuilds the scene every reset"
    draws = []
    for seed in (1, 2, 3):
        env.reset(seed=seed)
        draws.append(env.unwrapped.episode_perturbations["lighting"])
    assert {d["brightness"] for d in draws} == {1.5} and {d["direction"] for d in draws} == {0.0}
    assert len({d["temperature"] for d in draws}) == 3
    temperature = perturbation_registry()["lighting"]["temperature"].range
    assert all(temperature.contains(d["temperature"], "train") for d in draws)
    env.close()


def test_the_same_reset_seed_redraws_the_same_scene():
    env = _build(perturbations={c: "randomize-full" for c in ("lighting", "table_colour", "table_texture", "object_colour")})
    a = _frame(env.reset(seed=4)[0]); drawn = env.unwrapped.episode_perturbations
    b = _frame(env.reset(seed=4)[0])
    assert env.unwrapped.episode_perturbations == drawn and np.abs(a - b).max() == 0
    c = _frame(env.reset(seed=5)[0])
    assert env.unwrapped.episode_perturbations != drawn and np.abs(a - c).mean() > 0.5
    env.close()


def test_splits_draw_from_their_own_part_of_each_range():
    registry = perturbation_registry()
    for split, member in (("train", True), ("evaluation", False)):
        env = _build(perturbations={"table_colour": f"randomize-{split}", "viewpoint": {"yaw": f"randomize-{split}"}})
        for seed in range(4):
            env.reset(seed=seed)
            drawn = env.unwrapped.episode_perturbations
            for name in ("r", "g", "b", "brightness"):
                assert registry["table_colour"][name].range.contains(drawn["table_colour"][name], "train") == member
            assert registry["viewpoint"]["yaw"].range.contains(drawn["viewpoint"]["yaw"], "train") == member
        env.close()


def test_the_wrist_camera_is_posed_each_episode_by_the_drawn_viewpoint():
    from custom_maniskill_tasks.perturbation_mixin import viewpoint_config
    from custom_maniskill_tasks.viewpoint import wrist_camera_local_pose

    env = _build(perturbations={"viewpoint": {"yaw": "randomize-full", "pitch": "randomize-full", "left": 2.0}})
    poses = []
    for seed in (1, 2):
        env.reset(seed=seed)  # `check_perturbations` ran at build; here the pose is read back
        drawn = env.unwrapped.episode_perturbations["viewpoint"]
        built = env.unwrapped._sensors["hand_camera"].camera.local_pose
        expected = wrist_camera_local_pose(viewpoint_config(drawn, env.unwrapped.WRIST_RIG))
        got = [*built.p, *built.q]
        assert np.allclose(got, expected, atol=1e-4) or np.allclose(got, [*expected[:3], *-np.array(expected[3:])], atol=1e-4)
        assert drawn["left"] == 2.0
        poses.append(got)
    assert not np.allclose(poses[0], poses[1]), "the camera moved between episodes"
    env.close()


def test_bad_configs_are_refused_before_anything_is_built():
    assert "outside its range" in _error(_build, perturbations={"lighting": {"temperature": 1.5}})
    assert "Unknown perturbation class" in _error(_build, perturbations={"corruption": "default"})
    assert "wrist" in _error(make_env, TASK, sim_backend="physx_cpu", camera_view="default",
                             perturbations={"viewpoint": {"pitch": 5}})
    assert "reconfiguration_freq" in _error(_build, perturbations={"lighting": "randomize-train"}, reconfiguration_freq=0)
    assert "object of interest" in _error(_build, POKE, perturbations={"object_colour": "randomize-train"})


def test_other_tasks_build_and_it_is_refused_where_it_cannot_work():
    env = _build("LiftPegUpright-v1.1", {"object_colour": "randomize-train", "table_material": "randomize-full"})
    env.reset(seed=0)
    env.close()
    env = _build(POKE, {"lighting": "randomize-train", "table_texture": "randomize-train"})
    env.reset(seed=0)
    env.close()
    assert "-v1.1" in _error(_build, "PickCube-v1", {"lighting": "randomize-train"})


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    by_name = {t.__name__: t for t in TESTS}
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            by_name[name]()
            print(f"ok  {name}")
    else:
        failed = []
        for test in TESTS:
            result = subprocess.run([sys.executable, __file__, test.__name__], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"ok  {test.__name__}")
            else:
                failed.append(test.__name__)
                print(f"FAIL {test.__name__}\n{result.stdout[-1500:]}\n{result.stderr[-3000:]}")
        print(f"\n{len(TESTS) - len(failed)} passed" + (f", {len(failed)} failed: {failed}" if failed else ""))
        sys.exit(1 if failed else 0)
