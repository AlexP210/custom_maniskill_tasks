"""The registered `-v1.1` task ids: `python tests/test_tasks.py`.

physx_cpu at num_envs=1 throughout, since sapien can only enable GPU PhysX once per process.
"""

import gymnasium as gym
import numpy as np
import torch

import custom_maniskill_tasks  # noqa: F401  (registers the ids under test)
from custom_maniskill_tasks import FULL_HORIZON_TASKS, distractors, make_env
from custom_maniskill_tasks.viewpoint import in_wrist_view
from mani_skill.utils.assets import is_data_source_downloaded
from mani_skill.utils.registration import REGISTERED_ENVS

ACTION = np.array([0.4, -0.3, 0.2, 0.0], dtype=np.float32)


def _scalar(x):
    return np.asarray(x.cpu() if isinstance(x, torch.Tensor) else x).reshape(-1)[0]


def _missing_assets(task):
    """The asset ids `task` declares that are not on this machine.

    Only PickSingleYCB-v1.1 declares any. Building it without them would drop into ManiSkill's
    interactive "download now? (y|n)" prompt and hang the run, so the test says what to fetch and
    moves on instead -- a missing 26MB download is a machine that is not set up, not a defect in
    the registration, which the rest of these tests cover either way.
    """
    return [
        asset_id
        for asset_id in REGISTERED_ENVS[task].asset_download_ids or []
        if not is_data_source_downloaded(asset_id)
    ]


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
        missing = _missing_assets(task)
        if missing:
            print(f"      (skipped {task}: run `python -m mani_skill.utils.download_asset "
                  f"{' '.join(missing)}` to include it)")
            continue
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


def _distractor_positions(env):
    return torch.stack([a.pose.p[0].cpu() for a in env.distractors.values()])


def _distractors_are_clear_and_visible(env) -> bool:
    """Every distractor is inside the wrist camera's view and clear of the others and of the
    boxes its task says to keep clear."""
    footprints = distractors.ycb_footprints()
    radii = [footprints[m]["radius"] for m in env.distractor_models]
    positions = _distractor_positions(env).numpy()
    for xy, r in zip(positions[:, :2], radii):
        if not in_wrist_view(np.array([[*xy, 0.0], [*xy, 0.02]]), env.viewpoint).all():
            return False
    for i in range(len(positions)):
        for j in range(i):
            gap = np.abs(positions[i, :2] - positions[j, :2]).max() - radii[i] - radii[j]
            if gap < distractors.DISTRACTOR_MARGIN - 1e-3:
                return False
    for attr, offset, half in env.distractor_avoid:
        centre = getattr(env, attr).pose.p[0, :2].cpu().numpy() + np.array(offset)
        for xy, r in zip(positions[:, :2], radii):
            if (np.abs(xy - centre) - (np.array(half) + r)).max() < distractors.DISTRACTOR_MARGIN - 1e-3:
                return False
    return True


def test_distractors_are_opt_in_and_change_nothing_else():
    """`distractors=N` adds N YCB objects to the scene and nothing to the task."""
    for task in ("LiftPegUpright-v1.1", "PlaceSphere-v1.1", "PushCube-v1.1", "PickCube-v1.1", "PokeCube-v1.1"):
        stock = gym.make(task, num_envs=1, obs_mode="state")
        with_them = gym.make(task, num_envs=1, obs_mode="state", distractors=3)
        assert not stock.unwrapped.distractors, "must be off by default"
        assert "distractors" not in stock.spec.kwargs
        assert len(with_them.unwrapped.distractors) == 3
        assert with_them.spec.kwargs["distractors"] == 3, "recorded, so a replay rebuilds it"

        obs_stock, _ = stock.reset(seed=2)
        obs_with, _ = with_them.reset(seed=2)
        assert torch.allclose(obs_stock, obs_with), (task, "same object draw and state obs")
        env = with_them.unwrapped
        assert _distractors_are_clear_and_visible(env), task
        first = _distractor_positions(env)
        with_them.reset(seed=2)
        assert torch.equal(first, _distractor_positions(env)), "same seed, same places"
        with_them.reset(seed=3)
        assert not torch.equal(first, _distractor_positions(env)), "a different seed moves them"
        assert _distractors_are_clear_and_visible(env), task
        with_them.reset(seed=2)
        stock.reset(seed=2)
        for _ in range(10):
            obs_stock, reward_stock, _, _, info_stock = stock.step(ACTION)
            obs_with, reward_with, _, _, info_with = with_them.step(ACTION)
            assert torch.allclose(obs_stock, obs_with, atol=1e-5)
            assert torch.allclose(reward_stock, reward_with, atol=1e-5)
            assert torch.equal(info_stock["success"], info_with["success"])
        stock.close()
        with_them.close()


def test_distractors_take_a_count_and_nothing_else():
    """A flag from the old copies-of-the-object distractors, or a count out of range, raises."""
    for bad, error in ((True, TypeError), (1.5, TypeError), (-1, ValueError), (99, ValueError)):
        try:
            make_env("PickCube-v1.1", num_envs=1, sim_backend="physx_cpu", distractors=bad)
        except error:
            pass
        else:
            raise AssertionError(f"distractors={bad!r} was accepted")
    try:
        make_env("PickCube-v1", num_envs=1, sim_backend="physx_cpu", distractors=1)
    except ValueError:
        pass
    else:
        raise AssertionError("a stock -v1 id accepted distractors")
    env = make_env("PickCube-v1.1", num_envs=1, sim_backend="physx_cpu", distractors=0)
    assert "distractors" not in env.unwrapped.spec.kwargs, "a default is not passed on"
    env.close()


if __name__ == "__main__":
    tests = [
        test_ids_are_registered_with_both_registries,
        test_plain_gym_make_needs_no_extra_flags,
        test_dynamics_reward_and_success_are_unchanged,
        test_every_id_works_through_make_env,
        test_distractors_are_opt_in_and_change_nothing_else,
        test_distractors_take_a_count_and_nothing_else,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
