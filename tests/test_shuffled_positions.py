"""PickCube-MultiArmDistractors-ShuffledPositions: the robots' slots permuted each episode."""

import pytest
import torch

import custom_maniskill_tasks  # noqa: F401  (registers the ids under test)
from custom_maniskill_tasks import make_env
from custom_maniskill_tasks.shuffled_positions import DISTRACTOR_CUBE_COLOURS, SLOT_XY

TASK = "PickCube-MultiArmDistractors-ShuffledPositions"
BASE = "PickCube-MultiArmDistractors"


def _zero(env):
    return torch.zeros(env.action_space.shape)


def _reset(task, seed, **kwargs):
    env = make_env(task, obs_mode="state_dict", **kwargs)
    env.reset(seed=seed)
    return env


def test_slots_are_a_permutation_that_varies_with_the_seed():
    seen = set()
    for seed in range(12):
        env = _reset(TASK, seed)
        u = env.unwrapped
        slots = {k: int(v[0]) for k, v in u.robot_slots().items()}
        assert sorted(slots.values()) == [0, 1, 2]
        torch.testing.assert_close(u.agent.robot.pose.p[0, :2], SLOT_XY[slots["agent"]])
        for name in u.distractor_names:
            torch.testing.assert_close(u.distractors[name].robot.pose.p[0, :2], SLOT_XY[slots[name]])
        seen.add(tuple(slots.values()))
        env.close()
    assert len(seen) >= 4, f"only {seen} in 12 seeds"


@pytest.mark.parametrize("seed", [0, 1, 5])
def test_same_episode_as_the_unshuffled_task_relative_to_each_base(seed):
    s, b = _reset(TASK, seed), _reset(BASE, seed)
    us, ub = s.unwrapped, b.unwrapped
    for actor in ("cube", "goal_site"):
        torch.testing.assert_close(getattr(us, actor).pose.p - us.agent.robot.pose.p,
                                   getattr(ub, actor).pose.p - ub.agent.robot.pose.p)
    for name in us.distractor_names:
        torch.testing.assert_close(us.distractor_cubes[name].pose.p - us.distractors[name].robot.pose.p,
                                   ub.distractor_cubes[name].pose.p - ub.distractors[name].robot.pose.p)
    for _ in range(50):  # the whole episode: the distractors finish by step 22-35
        _, rs, *_ = s.step(_zero(s))
        _, rb, *_ = b.step(_zero(b))
        torch.testing.assert_close(rs, rb, atol=1e-5, rtol=0)
    assert all(bool(v.all()) for v in us.distractor_success().values()), "a distractor failed from a shuffled slot"
    s.close(), b.close()


def test_cube_colours_belong_to_the_robots():
    env = _reset(TASK, 0)
    u = env.unwrapped
    for name, rgba in DISTRACTOR_CUBE_COLOURS.items():
        material = u.distractor_cubes[name]._objs[0].find_component_by_type(
            __import__("sapien").render.RenderBodyComponent).render_shapes[0].material
        assert tuple(material.base_color) == pytest.approx(rgba)
    env.close()


def test_without_distractors_the_agent_still_moves():
    slots = set()
    for seed in range(8):
        env = _reset(TASK, seed, distractor_robots=[])
        slots.add(int(env.unwrapped.robot_slots()["agent"][0]))
        env.step(_zero(env))
        env.close()
    assert len(slots) > 1


def test_pick_place_refused():
    with pytest.raises(ValueError, match="pick_cube"):
        make_env(TASK, distractor_task="pick_place")


def test_distractor_draw_changes_only_the_distractors():
    """The distractor probe's conditions: same seed and actions, other distractor draws, same agent replay."""
    torch.manual_seed(0)
    actions = [torch.rand(4) * 2 - 1 for _ in range(30)]
    runs = []
    for draw in range(3):
        env = _reset(TASK, 4, sim_backend="physx_cpu", distractor_draw=draw)
        u = env.unwrapped
        start = u.distractor_state()
        for a in actions:
            env.step(a)
        runs.append(({k: v.clone() for k, v in u.robot_slots().items()}, start["object_p"],
                     u.agent.tcp.pose.p.clone(), u.cube.pose.p.clone(), u.distractor_state()["qpos"]))
        env.close()
    (slots0, objects0, tcp0, cube0, qpos0) = runs[0]
    for slots, objects, tcp, cube, qpos in runs[1:]:
        assert slots == slots0, "the slot permutation must not depend on the distractor draw"
        torch.testing.assert_close(tcp, tcp0, atol=0, rtol=0)
        torch.testing.assert_close(cube, cube0, atol=0, rtol=0)
        assert (objects - objects0).abs().max() > 0.01, "other draws, same distractor cubes"
        assert (qpos - qpos0).abs().max() > 0.01


def test_distractor_draw_zero_is_the_default_stream():
    a, b = _reset(BASE, 3), _reset(BASE, 3, distractor_draw=0)
    torch.testing.assert_close(a.unwrapped.distractor_state()["object_p"], b.unwrapped.distractor_state()["object_p"])
    a.close(), b.close()
