"""The registered `-v1.1` task ids: `python tests/test_tasks.py`.

physx_cpu at num_envs=1 throughout, since sapien can only enable GPU PhysX once per process.
"""

import gymnasium as gym
import numpy as np
import torch

import custom_maniskill_tasks  # noqa: F401  (registers the ids under test)
from custom_maniskill_tasks import FULL_HORIZON_TASKS, make_env
from mani_skill.utils.registration import REGISTERED_ENVS

ACTION = np.array([0.4, -0.3, 0.2, 0.0], dtype=np.float32)


def _scalar(x):
    return np.asarray(x.cpu() if isinstance(x, torch.Tensor) else x).reshape(-1)[0]


def test_ids_are_registered_with_both_registries():
    for task, base in FULL_HORIZON_TASKS.items():
        assert task in REGISTERED_ENVS, task
        assert task in gym.registry, f"{task} is not usable through plain gym.make"
        spec = REGISTERED_ENVS[task]
        assert spec.max_episode_steps == REGISTERED_ENVS[base].max_episode_steps
        assert issubclass(spec.cls, REGISTERED_ENVS[base].cls), "must be the same task underneath"
        assert spec.default_kwargs["control_mode"] == "pd_ee_delta_pos"


def _push_cube_into_the_goal(env):
    """Teleport the cube inside the goal region, so `info["success"]` is actually True.

    Random actions essentially never solve the task, so without this the "does not terminate"
    assertion would pass on an env that terminates perfectly happily.
    """
    base = env.unwrapped
    state = base.get_state_dict()
    cube, goal = state["actors"]["cube"], state["actors"]["goal_region"]
    cube[..., :2] = goal[..., :2]  # same xy; z, quaternion and velocities stay as they are
    cube[..., 2] = base.cube_half_size
    base.set_state_dict(state)


def test_plain_gym_make_needs_no_extra_flags():
    """The point of the ids: the collection convention survives a bare gym.make."""
    env = gym.make("PushCube-v1.1", num_envs=1)
    assert env.unwrapped.control_mode == "pd_ee_delta_pos", "the registered default should apply"
    assert env.unwrapped.reward_mode == "normalized_dense"
    env.reset(seed=0)
    _push_cube_into_the_goal(env)

    successes = 0
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        assert not bool(_scalar(terminated)), "the task itself must not terminate"
        successes += int(bool(_scalar(info["success"])))
    assert successes > 0, "the goal was never reached, so nothing was proven about terminated"
    assert bool(_scalar(truncated)), "the 50-step time limit still truncates"
    print(f"      (goal held for {successes}/50 steps, terminated stayed False throughout)")
    env.close()

    # the stock task, in the same situation, does end the episode -- this is the only difference
    stock = gym.make("PushCube-v1", num_envs=1, control_mode="pd_ee_delta_pos")
    stock.reset(seed=0)
    _push_cube_into_the_goal(stock)
    _, _, terminated, _, info = stock.step(np.zeros(4, dtype=np.float32))
    assert bool(_scalar(info["success"])) and bool(_scalar(terminated)), (
        "PushCube-v1 was expected to terminate on success; if it no longer does, the -v1.1 "
        "variants have nothing left to suppress"
    )
    stock.close()

    # an explicit control_mode still wins over the registration default
    env = gym.make("PushCube-v1.1", num_envs=1, control_mode="pd_joint_delta_pos")
    assert env.unwrapped.control_mode == "pd_joint_delta_pos"
    env.close()


def test_dynamics_reward_and_success_are_unchanged():
    """The variant differs from its stock task in `terminated` and nothing else."""
    variant = gym.make("PushCube-v1.1", num_envs=1)
    stock = gym.make("PushCube-v1", num_envs=1, control_mode="pd_ee_delta_pos")
    obs_variant, _ = variant.reset(seed=3)
    obs_stock, _ = stock.reset(seed=3)
    assert torch.allclose(obs_variant, obs_stock), "same scene and reset distribution"
    for _ in range(10):
        obs_variant, reward_variant, terminated, _, info_variant = variant.step(ACTION)
        obs_stock, reward_stock, _, _, info_stock = stock.step(ACTION)
        assert torch.allclose(obs_variant, obs_stock), "same dynamics"
        assert torch.allclose(reward_variant, reward_stock), "same reward"
        assert torch.equal(info_variant["success"], info_stock["success"]), "same predicate"
        assert not bool(_scalar(terminated))
    variant.close()
    stock.close()


def test_every_id_works_through_make_env():
    for task in FULL_HORIZON_TASKS:
        env = make_env(task, obs_mode="rgb", camera_view="focused", camera_resolution=64,
                       frame_skip=2, n_frames=2)
        obs, _ = env.reset(seed=0)
        rgb = obs["sensor_data"]["base_camera"]["rgb"]
        assert rgb.shape == (1, 2, 64, 64, 3), (task, rgb.shape)
        assert env.action_space.shape == (2 * env.unwrapped.single_action_space.shape[0],)
        obs, reward, terminated, truncated, info = env.step(env.rand_act())
        assert not bool(_scalar(terminated))
        assert int(_scalar(env.unwrapped.elapsed_steps)) == 2
        print(f"      ({task}: horizon {env.get_wrapper_attr('_max_episode_steps')}, "
              f"action {tuple(env.action_space.shape)})")
        env.close()


if __name__ == "__main__":
    tests = [
        test_ids_are_registered_with_both_registries,
        test_plain_gym_make_needs_no_extra_flags,
        test_dynamics_reward_and_success_are_unchanged,
        test_every_id_works_through_make_env,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
