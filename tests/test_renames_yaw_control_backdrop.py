"""The 2026-11-23 changes: renamed ids and their aliases, the 5-D `pd_ee_delta_pos_yaw` control mode,
and `background_video` on every `-MultiArmDistractors` task.

    python -m pytest tests/test_renames_yaw_control_backdrop.py

physx_cpu at num_envs=1 throughout, since sapien can only enable GPU PhysX once per process.
"""

import numpy as np
import pytest
import torch

import custom_maniskill_tasks  # noqa: F401  (registers the ids under test)
from custom_maniskill_tasks import DEPRECATED_TASK_IDS, make_env
from custom_maniskill_tasks.backdrop_video import BACKDROP_X, backdrop_centre
from custom_maniskill_tasks.controllers import tcp_yaw
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import REGISTERED_ENVS

RENAMED = {
    "PushCube-v1.2": "PushCube-FR3",
    "PlaceSphere-v1.2": "PlaceSphere-FR3",
    "LiftPegUpright-v1.2": "LiftPegUpright-FR3",
    "PokeCube-v1.2": "PokeCube-FR3",
    "PickCube-v1.2": "PickCube-FR3",
    "PickSingleYCB-v1.2": "PickSingleYCB-FR3",
    "LiftPegUpright-v2": "LiftPegUpright-HarderPerception",
    "LiftPegUpright-v3": "LiftPegUpright-MultiArmDistractors",
    "PickCube-v2": "PickCube-MultiArmDistractors",
    "PegInsertionSide-v2": "PegInsertionSide-MultiArmDistractors",
}
DISTRACTOR_IDS = ("PegInsertionSide-MultiArmDistractors", "LiftPegUpright-MultiArmDistractors", "PickCube-MultiArmDistractors")


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- renames
def test_old_ids_are_aliases_of_the_new_ones():
    assert DEPRECATED_TASK_IDS == RENAMED
    for old, new in RENAMED.items():
        a, b = REGISTERED_ENVS[old], REGISTERED_ENVS[new]
        assert a.cls is b.cls and a.default_kwargs == b.default_kwargs and a.max_episode_steps == b.max_episode_steps
    assert "PegInsertionSide-v3" not in REGISTERED_ENVS


def test_an_alias_builds_the_same_episode():
    def first_state(task):
        env = make_env(task, obs_mode="state")
        obs, _ = env.reset(seed=5)
        env.close()
        return obs

    torch.testing.assert_close(first_state("PickCube-v2"), first_state("PickCube-MultiArmDistractors"))


# --------------------------------------------------------------------------- yaw control
def test_yaw_mode_is_5d_turns_the_hand_and_keeps_it_vertical():
    env = make_env("PegInsertionSide-v1.1", obs_mode="state")
    assert env.unwrapped.control_mode == "pd_ee_delta_pos_yaw" and env.action_space.shape == (5,)
    env.reset(seed=0)
    tcp = env.unwrapped.agent.tcp
    yaw0, p0 = float(tcp_yaw(tcp.pose.q[0])), tcp.pose.p[0].clone()
    for _ in range(10):
        env.step(torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))
    turned = _wrap(float(tcp_yaw(tcp.pose.q[0])) - yaw0)
    assert 0.25 < turned < 1.0 + 1e-3, turned  # +dyaw turns anticlockwise seen from above, ~0.04 rad/step
    assert (tcp.pose.p[0] - p0).norm() < 0.01, "a pure turn should not move the hand"
    for _ in range(20):  # move and turn back down to the table: the hand stays pointing straight down
        env.step(torch.tensor([0.3, -0.3, -0.5, -1.0, 1.0]))
        z = quaternion_to_matrix(tcp.pose.q[0])[:, 2]
        assert float(z[2]) < -0.995, z
    env.close()


def test_other_tasks_and_panda_modes_are_untouched():
    env = make_env("PushCube-v1.1", obs_mode="state")
    assert env.unwrapped.control_mode == "pd_ee_delta_pos" and env.action_space.shape == (4,)
    env.close()
    env = make_env("PegInsertionSide-v1.1", obs_mode="state", control_mode="pd_ee_delta_pose")
    assert env.action_space.shape == (7,)
    env.close()


def _scripted_episode(env, seed, horizon):
    """A closed-loop scripted policy in the 5-D action space: turn the hand along the peg and grasp
    it behind its centre, lift, turn the peg onto the hole's axis and line its head up in front of
    the box, then push in while centred. Returns the first success step, or None."""
    env.reset(seed=seed)
    u = env.unwrapped
    half, radius = float(u.peg_half_sizes[0, 0]), float(u.peg_half_sizes[0, 2])
    phase, hold = "approach", 0
    for t in range(horizon):
        tcp, hand_yaw = u.agent.tcp.pose.p[0].numpy(), float(tcp_yaw(u.agent.tcp.pose.q[0]))
        head, centre, hole = u.peg_head_pose.p[0].numpy(), u.peg.pose.p[0].numpy(), u.box_hole_pose.p[0].numpy()
        axis = quaternion_to_matrix(u.box_hole_pose.q[0])[:, 0].numpy()
        axis[2] = 0
        axis /= np.linalg.norm(axis)
        heading = head - centre
        heading[2] = 0
        heading /= np.linalg.norm(heading)
        peg_yaw, axis_yaw = np.arctan2(heading[1], heading[0]), np.arctan2(axis[1], axis[0])
        grasp, grip = centre - heading * 0.03, 1.0
        if phase in ("approach", "descend", "close"):
            dyaw = _wrap(2 * (peg_yaw - hand_yaw)) / 2  # the hand's x along the peg, either way round
        if phase == "approach":
            target = np.r_[grasp[:2], radius + 0.08]
            if np.linalg.norm(target - tcp) < 0.01 and abs(dyaw) < 0.03:
                phase, hold = "descend", 0
        elif phase == "descend":
            target, hold = np.r_[grasp[:2], radius], hold + 1
            if abs(tcp[2] - radius) < 0.003 or hold > 15:
                phase, hold = "close", 0
        elif phase == "close":
            target, grip, hold = np.r_[grasp[:2], radius], -1.0, hold + 1
            if hold >= 4:
                phase = "lift"
        if phase in ("lift", "align", "insert"):
            grip, dyaw = -1.0, _wrap(axis_yaw - peg_yaw)
            if phase == "lift":
                goal = np.r_[head[:2], hole[2] + 0.03]
                if head[2] > hole[2] - 0.005:
                    phase = "align"
            if phase == "align":
                goal = hole - axis * (half + 0.03)
                if np.linalg.norm(head - goal) < 0.004 and abs(dyaw) < 0.01:
                    phase = "insert"
            if phase == "insert":
                rel = hole - head
                lateral = rel - axis * (rel @ axis)
                advance = 0.012 if np.linalg.norm(lateral) < 0.0015 and abs(dyaw) < 0.01 else 0.0
                goal = head + lateral + axis * min(advance, max(rel @ axis + 0.01, 0))
            target = tcp + (goal - head)
        action = np.r_[np.clip(2.0 * (target - tcp) / 0.1, -1, 1), np.clip(2.5 * dyaw / 0.1, -1, 1), grip]
        env.step(torch.tensor(action, dtype=torch.float32))
        if bool(u.evaluate()["success"][0]):
            return t + 1
    return None


def test_yaw_control_makes_the_stock_geometry_solvable():
    """Position-only control could insert ~1 stock layout in 10 at all; the scripted policy above
    solves 15/20 within 200 steps (11/20 within the 100-step horizon, median 80). Here, half of 6."""
    env = make_env("PegInsertionSide-v1.1", obs_mode="state")
    solved = [_scripted_episode(env, seed, 200) for seed in range(6)]
    env.close()
    assert sum(s is not None for s in solved) >= 3, solved


# --------------------------------------------------------------------------- background video
@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A 30-frame 20 Hz clip whose frame k is a flat grey of level 8 k, so frames are told apart."""
    import cv2

    path = str(tmp_path_factory.mktemp("clip") / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 36))
    for k in range(30):
        writer.write(np.full((36, 64, 3), 8 * k, np.uint8))
    writer.release()
    return path


@pytest.mark.parametrize("task", DISTRACTOR_IDS)
def test_background_video_is_off_by_default(task):
    env = make_env(task)
    env.reset(seed=0)
    assert env.unwrapped._backdrop is None
    env.close()


def test_background_video_plays_and_stays_out_of_the_state(clip):
    env = make_env("PickCube-MultiArmDistractors", background_video=clip)
    env.reset(seed=0)
    u = env.unwrapped
    actor, count = u._backdrop
    assert count == 30
    assert not any(k.startswith("backdrop_video") for k in u.get_state_dict()["actors"])

    def shown():
        z = actor.pose.p[:, 2].cpu().numpy()
        on = np.flatnonzero(np.isclose(z, backdrop_centre()[2]))
        assert len(on) == 1 and np.allclose(actor.pose.p[:, 0].cpu().numpy(), BACKDROP_X)
        return int(on[0])

    first = shown()
    for t in range(1, 35):  # past the end of the loop
        env.step(torch.zeros(env.action_space.shape))
        assert shown() == (first + t) % count
    env.close()


def test_background_video_reaches_the_scene_camera_and_not_the_task(clip):
    """The scene camera's top rows look at the wall and change as the clip plays; the task's draws
    are the same with and without it."""
    def run(**kwargs):
        env = make_env("LiftPegUpright-MultiArmDistractors", distractor_robots=[], **kwargs)
        obs, _ = env.reset(seed=0)
        peg = env.unwrapped.peg.pose.raw_pose.clone()
        top = lambda o: o["sensor_data"]["base_camera"]["rgb"][0, :20].float().mean().item()
        a = top(obs)
        for _ in range(10):
            obs, *_ = env.step(torch.zeros(env.action_space.shape))
        env.close()
        return peg, abs(top(obs) - a)

    (peg_on, change), (peg_off, _) = run(background_video=clip), run()
    assert change > 5, "the wall should brighten as the grey clip does"
    torch.testing.assert_close(peg_on, peg_off)
