"""Peg-insertion distractors (`distractor_task="peg_insertion"`, `peg_distractor_robots`): `python -m pytest tests/test_peg_distractors.py`.

physx_cpu at num_envs=1 throughout, since sapien can only enable GPU PhysX once per process.
"""

import numpy as np
import pytest
import torch

import custom_maniskill_tasks  # noqa: F401  (registers the ids under test)
from custom_maniskill_tasks import YAW_CONTROL_MODE, make_env
from custom_maniskill_tasks.peg_distractor_robots import PEG_CYCLE_STEPS

TASK = "PegInsertionSide-MultiArmDistractors"


def _idle(env):
    a = torch.zeros(env.action_space.shape)
    a[-1] = 1.0
    return a


def test_bad_distractor_task_is_refused():
    with pytest.raises(ValueError, match="distractor_task"):
        make_env(TASK, distractor_task="stacking")


def test_distractors_insert_their_pegs_and_put_them_back():
    env = make_env(TASK, distractor_task="peg_insertion")
    for seed in (0, 1):
        env.reset(seed=seed)
        u = env.unwrapped
        assert env.action_space.shape == (5,), "distractors are not part of the action space"
        assert {r.control_mode for r in u.distractors.values()} == {YAW_CONTROL_MODE}
        assert u.distractor_cubes == {}
        delays = u._distractor_delay[:, 0].tolist()
        inserted = {n: None for n in u.distractor_names}
        for t in range(PEG_CYCLE_STEPS + max(delays)):
            env.step(_idle(env))
            for n, flag in u.distractor_inserted().items():
                if bool(flag[0]) and inserted[n] is None:
                    inserted[n] = t + 1
        # each goes in within the episode, and a full cycle later its peg is back where it lay
        assert all(step is not None and step <= 100 for step in inserted.values()), inserted
        for k, n in enumerate(u.distractor_names):
            peg = u.distractor_pegs[n].pose.p[0].numpy()
            np.testing.assert_allclose(peg[:2], u._distractor_rest[k, 0, :2].numpy(), atol=0.01)
            assert peg[2] < 0.03, "lying on the table"
    env.close()


def test_task_draws_and_agent_are_untouched():
    """The agent's peg and box, its proprioception and its action space don't depend on what the distractors do."""
    poses = {}
    for task in ("pick_place", "peg_insertion"):
        env = make_env(TASK, distractor_task=task)
        env.reset(seed=3)
        u = env.unwrapped
        poses[task] = (u.peg.pose.raw_pose.numpy(), u.box.pose.raw_pose.numpy(), u.agent.robot.get_qpos().numpy())
        env.close()
    for a, b in zip(*poses.values()):
        np.testing.assert_array_equal(a, b)


def test_state_dict_replays_the_distractors():
    env = make_env(TASK, distractor_task="peg_insertion")
    env.reset(seed=0)
    u = env.unwrapped
    for _ in range(40):
        env.step(_idle(env))
    state = u.get_state_dict()
    assert {"distractor_left_peg", "distractor_right_peg"} <= set(state["actors"])
    elapsed = u.elapsed_steps.clone()
    for _ in range(10):
        env.step(_idle(env))
    after = {n: p.pose.p.clone() for n, p in u.distractor_pegs.items()}
    u.set_state_dict(state)
    u._elapsed_steps[:] = elapsed  # the script is timed by elapsed_steps, which the state dict leaves out
    for _ in range(10):
        env.step(_idle(env))
    for n, p in u.distractor_pegs.items():
        torch.testing.assert_close(p.pose.p, after[n], atol=2e-3, rtol=0)
    env.close()
