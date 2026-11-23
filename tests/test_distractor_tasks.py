"""LiftPegUpright-MultiArmDistractors (the shared distractor scene on LiftPegUpright): `python -m pytest tests/test_distractor_tasks.py`.

physx_cpu at num_envs=1 throughout, since sapien can only enable GPU PhysX once per process.
"""

import math

import numpy as np
import torch
from transforms3d.euler import euler2quat

import custom_maniskill_tasks  # noqa: F401  (registers the ids under test)
from custom_maniskill_tasks import make_env
from custom_maniskill_tasks.distractor_scene import SCENE_CAMERA_EYE, SceneCameraMixin
from custom_maniskill_tasks.distractor_tasks import LIFT_PEG_MAX_START_DELAY
from mani_skill.utils.registration import REGISTERED_ENVS

TASK = "LiftPegUpright-MultiArmDistractors"


def _zero(env):
    return torch.zeros(env.action_space.shape)


def test_registered_with_stock_horizon_and_control_mode():
    spec = REGISTERED_ENVS[TASK]
    assert spec.max_episode_steps == REGISTERED_ENVS["LiftPegUpright-v1"].max_episode_steps == 50
    assert spec.default_kwargs["control_mode"] == "pd_ee_delta_pos"
    assert issubclass(spec.cls, SceneCameraMixin)


def test_scene_camera_only_and_plain_panda():
    env = make_env(TASK)
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    assert list(obs["sensor_data"]) == ["base_camera"], "no wrist camera"
    assert u.agent.uid == "panda"
    assert env.action_space.shape == (4,), "distractors are not part of the action space"
    eye = u._sensors["base_camera"].camera.global_pose.p[0].numpy()
    np.testing.assert_allclose(eye, SCENE_CAMERA_EYE, atol=1e-5)
    assert u.distractor_names == ("left", "right")
    env.close()


def test_control_scene_has_no_distractors():
    env = make_env(TASK, distractor_robots=[])
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    assert u.distractors == {} and u.distractor_cubes == {}
    assert not any(k.startswith("distractor") for k in u.get_state_dict()["actors"])
    assert list(obs["sensor_data"]) == ["base_camera"]
    env.step(_zero(env))
    env.close()


def test_success_does_not_terminate():
    env = make_env(TASK, ignore_terminations=False)
    env.reset(seed=0)
    u = env.unwrapped
    # stand the peg on its end, clear of the gripper. The stock check reads the XYZ-Euler z angle,
    # which a bare 90-degree turn about y leaves at 0 (gimbal lock); this pose stands the peg the
    # same way with z at pi/2
    state = u.get_state_dict()
    peg = state["actors"]["peg"]
    peg[..., :3] = torch.tensor([0.05, 0.1, u.peg_half_length])
    peg[..., 3:7] = torch.tensor(euler2quat(0, math.pi / 2, math.pi / 2), dtype=torch.float32)
    peg[..., 7:] = 0
    u.set_state_dict(state)
    _, _, terminated, truncated, info = env.step(_zero(env))
    assert bool(info["success"].reshape(-1)[0]), "teleport did not produce a success to suppress"
    assert not bool(terminated.reshape(-1)[0]) and not bool(truncated.reshape(-1)[0])
    env.close()


def test_task_draws_match_v1_1_per_seed():
    """The distractors draw from their own stream: same seed, same peg and agent start as v1.1."""
    def first_state(task):
        env = make_env(task, robot_uids="panda")
        env.reset(seed=3)
        u = env.unwrapped
        out = u.peg.pose.raw_pose.clone(), u.agent.robot.get_qpos().clone()
        env.close()
        return out

    for a, b in zip(first_state("LiftPegUpright-v1.1"), first_state(TASK)):
        torch.testing.assert_close(a, b)


def test_distractors_start_early_and_lift_within_the_episode():
    """The shortened start delay lets each distractor lift its cube before the 50-step horizon."""
    for seed in (0, 1, 2, 3):
        env = make_env(TASK)
        env.reset(seed=seed)
        u = env.unwrapped
        assert int(u._distractor_delay.max()) <= LIFT_PEG_MAX_START_DELAY
        peak = {n: 0.0 for n in u.distractor_names}
        for _ in range(50):
            env.step(_zero(env))
            for n in u.distractor_names:
                peak[n] = max(peak[n], float(u.distractor_cubes[n].pose.p[0, 2]))
        assert all(v > 0.15 for v in peak.values()), (seed, peak)
        assert float(u.peg.pose.p[0, 2]) < 0.05, "the peg moved without the agent touching it"
        env.close()


def test_object_hue_leaves_distractors_alone():
    def colours(perturbations):
        import sapien

        env = make_env(TASK, perturbations=perturbations)
        env.reset(seed=0)
        u = env.unwrapped

        def first_colour(actor):
            body = actor._objs[0].find_component_by_type(sapien.render.RenderBodyComponent)
            return np.array(body.render_shapes[0].parts[0].material.base_color)

        out = first_colour(u.distractor_cubes["left"]), first_colour(u.peg)
        env.close()
        return out

    (cube0, peg0), (cube1, peg1) = colours(None), colours({"object_hue": {"degrees": 120}})
    np.testing.assert_allclose(cube0, cube1)
    assert not np.allclose(peg0, peg1), "object_hue did not reach the task's own peg"


def test_scene_camera_frames_all_three_robots_and_the_peg():
    """Every link of the agent and both distractors, and the peg, stay inside the frame (5% margin)."""
    env = make_env(TASK)
    env.reset(seed=2)
    u = env.unwrapped
    params = u._sensors["base_camera"].get_params()
    K, E = params["intrinsic_cv"][0].numpy(), params["extrinsic_cv"][0].numpy()
    size = u._sensors["base_camera"].config.width
    worst = 1.0
    for t in range(50):
        if t % 5 == 0:
            points = [link.pose.p[0].numpy() for robot in [u.agent, *u.distractors.values()]
                      for link in robot.robot.get_links()] + [u.peg.pose.p[0].numpy()]
            for xyz in points:
                cam = E @ np.r_[xyz, 1.0]
                assert cam[2] > 0
                uv = (K @ cam[:3])[:2] / cam[2]
                worst = min(worst, *(uv / size), *(1 - uv / size))
        env.step(_zero(env))
    assert worst > 0.05, f"a point came within {worst:.0%} of the frame edge"
    env.close()
