"""PlugCharger-v1.1: `python -m pytest tests/test_plug_charger.py`.

physx_cpu at num_envs=1 throughout, since sapien can only enable GPU PhysX once per process.
"""

import numpy as np
import torch

import custom_maniskill_tasks  # noqa: F401  (registers the ids under test)
from custom_maniskill_tasks import make_env
from mani_skill.utils.registration import REGISTERED_ENVS

STOCK_START_QPOS = np.array([0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04])


def _zero(env):
    return torch.zeros(env.action_space.shape)


def test_registered_with_stock_horizon_control_mode_and_a_real_reward():
    spec = REGISTERED_ENVS["PlugCharger-v1.1"]
    assert spec.max_episode_steps == REGISTERED_ENVS["PlugCharger-v1"].max_episode_steps == 200
    assert spec.default_kwargs["control_mode"] == "pd_ee_delta_pos"
    env = make_env("PlugCharger-v1.1")
    assert env.unwrapped.reward_mode == "sparse", "the stock fallback, 'none', is a reward of 0 on every step"
    env.close()


def test_stock_scene_camera_only_plain_panda_and_stock_start_pose():
    env = make_env("PlugCharger-v1.1")
    obs, _ = env.reset(seed=0)
    assert list(obs["sensor_data"]) == ["base_camera"], "no wrist camera"
    u = env.unwrapped
    assert u.agent.uid == "panda" and env.action_space.shape == (4,)
    pose = u._sensors["base_camera"].camera.get_global_pose()
    assert np.allclose(pose.p, [0.3, 0.0, 0.6], atol=1e-5), "the stock task's own base_camera"
    for seed in range(3):
        env.reset(seed=seed)
        qpos = u.agent.robot.get_qpos()[0].cpu().numpy()
        assert np.abs(qpos[:7] - STOCK_START_QPOS[:7]).max() < 0.1  # the stock joint vector, plus 0.02 noise
    env.close()


def test_success_does_not_terminate_and_is_rewarded():
    env = make_env("PlugCharger-v1.1", ignore_terminations=False)
    env.reset(seed=0)
    u = env.unwrapped
    u.charger.set_pose(u.goal_pose)  # plugged in
    _, reward, terminated, truncated, info = env.step(_zero(env))
    assert bool(info["success"].reshape(-1)[0]), "teleport did not produce a success to suppress"
    assert not bool(terminated.reshape(-1)[0]) and not bool(truncated.reshape(-1)[0])
    assert float(reward.reshape(-1)[0]) == 1.0
    env.reset(seed=0)
    _, reward, *_ = env.step(_zero(env))
    assert float(reward.reshape(-1)[0]) == 0.0
    env.close()


def test_table_perturbations_find_the_table():
    """The stock task names its table scene `scene_builder`; the table perturbations need `table_scene`."""
    env = make_env("PlugCharger-v1.1", perturbations={"table_material": {"material": "marble"}})
    env.reset(seed=0)
    assert env.unwrapped.table_scene is env.unwrapped.scene_builder
    env.close()
