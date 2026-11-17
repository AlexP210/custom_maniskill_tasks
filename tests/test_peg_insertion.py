"""PegInsertionSide-v1.1 / -v2: `python -m pytest tests/test_peg_insertion.py`.

physx_cpu at num_envs=1 throughout, since sapien can only enable GPU PhysX once per process.
"""

import numpy as np
import torch

import custom_maniskill_tasks  # noqa: F401  (registers the ids under test)
from custom_maniskill_tasks import make_env
from custom_maniskill_tasks.distractor_robots import CUBE_HALF_SIZE, CYCLE_STEPS, STATIONS
from mani_skill.utils.registration import REGISTERED_ENVS


def _zero(env):
    return torch.zeros(env.action_space.shape)


def test_registered_with_stock_horizon_and_control_mode():
    for task in ("PegInsertionSide-v1.1", "PegInsertionSide-v2"):
        spec = REGISTERED_ENVS[task]
        assert spec.max_episode_steps == REGISTERED_ENVS["PegInsertionSide-v1"].max_episode_steps == 100
        assert spec.default_kwargs["control_mode"] == "pd_ee_delta_pos"


def test_scene_camera_only_and_plain_panda():
    env = make_env("PegInsertionSide-v2")
    obs, _ = env.reset(seed=0)
    assert list(obs["sensor_data"]) == ["base_camera"], "no wrist camera"
    assert env.unwrapped.agent.uid == "panda"
    assert env.action_space.shape == (4,), "distractors are not part of the action space"
    env.close()


def test_success_does_not_terminate():
    env = make_env("PegInsertionSide-v2", ignore_terminations=False)
    env.reset(seed=0)
    u = env.unwrapped
    u.peg.set_pose(u.goal_pose)  # head at the hole's centre: inserted
    _, _, terminated, truncated, info = env.step(_zero(env))
    assert bool(info["success"].reshape(-1)[0]), "teleport did not produce a success to suppress"
    assert not bool(terminated.reshape(-1)[0]) and not bool(truncated.reshape(-1)[0])
    env.close()


def test_task_draws_match_v1_1_per_seed():
    """The distractors draw from their own stream: same seed, same peg, box and agent start."""
    def first_state(task):
        env = make_env(task)
        env.reset(seed=3)
        u = env.unwrapped
        out = (u.peg_half_sizes.clone(), u.peg.pose.raw_pose.clone(), u.box.pose.raw_pose.clone(),
               u.agent.robot.get_qpos().clone())
        env.close()
        return out

    for a, b in zip(first_state("PegInsertionSide-v1.1"), first_state("PegInsertionSide-v2")):
        torch.testing.assert_close(a, b)


def test_distractors_carry_their_cubes_along_the_tour():
    """By the end of its first cycle each distractor has carried its cube to the next pad in its tour."""
    for seed in (1, 4):
        env = make_env("PegInsertionSide-v2")
        env.reset(seed=seed)
        u = env.unwrapped
        ends = {n: int(u._distractor_delay[k, 0]) + CYCLE_STEPS for k, n in enumerate(u.distractor_names)}
        peak = {n: 0.0 for n in u.distractor_names}
        for k, n in enumerate(u.distractor_names):
            station = STATIONS[n]
            goal = torch.tensor(station.pads[station.tour[(int(u._distractor_start[k, 0]) + 1) % len(station.tour)]])
            ends[n] = (ends[n], goal)
        for t in range(1, max(e for e, _ in ends.values()) + 1):
            env.step(_zero(env))
            for n, (end, goal) in ends.items():
                peak[n] = max(peak[n], float(u.distractor_cubes[n].pose.p[0, 2]))
                if t == end:
                    at = u.distractor_cubes[n].pose.p[0, :2]
                    assert peak[n] > 0.15, f"{n} never lifted its cube"
                    assert (at - goal).norm() < 0.03, (seed, n, at, goal)
        env.close()


def test_distractor_selection_and_state_dict():
    env = make_env("PegInsertionSide-v2", distractor_robots=["left"])
    env.reset(seed=0)
    u = env.unwrapped
    state = u.get_state_dict()
    assert "panda-agent-distractor-left" in state["articulations"]
    assert "distractor_left_cube" in state["actors"]
    assert not any("right" in k for k in [*state["articulations"], *state["actors"]])
    for _ in range(30):
        env.step(_zero(env))
    moved = u.get_state_dict()
    u.set_state_dict(state)
    torch.testing.assert_close(u.distractor_cubes["left"].pose.p, state["actors"]["distractor_left_cube"][:, :3])
    assert not torch.allclose(moved["articulations"]["panda-agent-distractor-left"],
                              state["articulations"]["panda-agent-distractor-left"])
    env.close()


def test_object_hue_leaves_distractors_alone():
    def colours(perturbations):
        env = make_env("PegInsertionSide-v2", perturbations=perturbations)
        env.reset(seed=0)
        u = env.unwrapped

        def first_colour(actor):
            import sapien
            body = actor._objs[0].find_component_by_type(sapien.render.RenderBodyComponent)
            return np.array(body.render_shapes[0].parts[0].material.base_color)

        out = first_colour(u.distractor_cubes["left"]), first_colour(u.peg)
        env.close()
        return out

    (cube0, peg0), (cube1, peg1) = colours(None), colours({"object_hue": {"degrees": 120}})
    np.testing.assert_allclose(cube0, cube1)
    assert not np.allclose(peg0, peg1), "object_hue did not reach the task's own peg"


def test_scene_camera_frames_all_three_robots():
    """Every link of the agent and both distractors stays inside the scene camera's frame (5%
    margin) through an episode, so the distractors are actually in the observation."""
    env = make_env("PegInsertionSide-v2")
    env.reset(seed=2)
    u = env.unwrapped
    params = u._sensors["base_camera"].get_params()
    K, E = params["intrinsic_cv"][0].numpy(), params["extrinsic_cv"][0].numpy()
    size = u._sensors["base_camera"].config.width
    worst = 1.0
    for t in range(100):
        if t % 5 == 0:
            for robot in [u.agent, *u.distractors.values()]:
                for link in robot.robot.get_links():
                    cam = E @ np.r_[link.pose.p[0].numpy(), 1.0]
                    assert cam[2] > 0, f"{link.name} is behind the camera"
                    uv = (K @ cam[:3])[:2] / cam[2]
                    worst = min(worst, *(uv / size), *(1 - uv / size))
        env.step(_zero(env))
    assert worst > 0.05, f"a robot link came within {worst:.0%} of the frame edge"
    env.close()
