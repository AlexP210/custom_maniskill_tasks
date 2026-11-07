"""`LiftPegUpright-v2`, the marked peg: `python tests/test_marked_peg.py`.

physx_cpu at num_envs=1 throughout, since sapien can only enable GPU PhysX once per process.
"""

import gymnasium as gym
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import custom_maniskill_tasks  # noqa: F401  (registers the ids under test)
from custom_maniskill_tasks import BASE_MARKINGS, MARKINGS, TOP_MARKINGS, marking_pairs
from mani_skill.utils.geometry import rotation_conversions

TASK = "LiftPegUpright-v2"


def _make(**kwargs):
    return gym.make(TASK, num_envs=1, obs_mode="state", sim_backend="physx_cpu", **kwargs)


def _base_x_direction(task) -> float:
    """The world x-component of the peg's base-end axis (its local +x)."""
    return float(rotation_conversions.quaternion_to_matrix(task.peg.pose.q)[0, 0, 0])


def _stand(task, base_down: bool):
    """Teleport the peg upright on its base end (or on its decoy end), clear of the gripper."""
    state = task.get_state_dict()
    peg = state["actors"]["peg"]
    peg[..., :3] = torch.tensor([0.05, 0.1, task.peg_half_length])
    peg[..., 3:7] = torch.tensor(euler2quat(0, np.pi / 2 if base_down else -np.pi / 2, 0), dtype=torch.float32)
    peg[..., 7:] = 0
    task.set_state_dict(state)


def test_marking_splits():
    """Base and decoy markings never overlap, training never sees a held-out marking, and every
    evaluation pair holds out exactly one, so it is solvable by elimination."""
    base = set(BASE_MARKINGS["train"] + BASE_MARKINGS["evaluation"])
    top = set(TOP_MARKINGS["train"] + TOP_MARKINGS["evaluation"])
    assert not base & top and base | top == set(MARKINGS)
    held_out = set(BASE_MARKINGS["evaluation"] + TOP_MARKINGS["evaluation"])
    train, evaluation, full = (set(marking_pairs(s)) for s in ("train", "evaluation", "full"))
    assert all(not set(pair) & held_out for pair in train)
    assert all(len(set(pair) & held_out) == 1 for pair in evaluation)
    assert train | evaluation < full and not train & evaluation
    assert all(t == "none" for _, t in marking_pairs("train", decoy=False))
    assert {b for b, _ in marking_pairs("evaluation", decoy=False)} == set(BASE_MARKINGS["evaluation"])


def test_bad_kwargs_are_rejected():
    for bad in ({"markings": "test"}, {"yaw": "half"}, {"colours": "rgb"}, {"decoy": "no"}):
        try:
            _make(**bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} was accepted")


def test_dynamics_are_v1_1s():
    """The markings are visual only: one collision box, of v1.1's size and mass."""
    def body(task):
        return task.peg._objs[0].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)

    v1, v2 = gym.make("LiftPegUpright-v1.1", num_envs=1, obs_mode="state", sim_backend="physx_cpu"), _make()
    v1.reset(seed=0)
    v2.reset(seed=0)
    a, b = body(v1.unwrapped), body(v2.unwrapped)
    assert len(b.collision_shapes) == 1, "a marking has a collision shape"
    assert np.allclose(a.collision_shapes[0].half_size, b.collision_shapes[0].half_size)
    assert np.isclose(a.mass, b.mass)
    assert len(v2.unwrapped.peg._objs[0].find_component_by_type(sapien.render.RenderBodyComponent).render_shapes) > 1
    v1.close()
    v2.close()


def test_reset_is_seeded_and_places_the_peg_as_v1_1_does():
    v1, v2 = gym.make("LiftPegUpright-v1.1", num_envs=1, obs_mode="state", sim_backend="physx_cpu"), _make()
    task = v2.unwrapped
    for seed in (0, 5):
        v1.reset(seed=seed)
        v2.reset(seed=seed)
        assert torch.allclose(v1.unwrapped.peg.pose.p, task.peg.pose.p), "same xy as v1.1 at the same seed"
    v2.reset(seed=3)
    drawn, q = dict(task.peg_markings[0]), task.peg.pose.q.clone()
    v2.reset(seed=3)
    assert drawn == task.peg_markings[0] and torch.allclose(q, task.peg.pose.q)
    v1.close()
    v2.close()


def test_flip_yaw_turns_the_base_end_both_ways():
    env = _make(reconfiguration_freq=0)
    xs = []
    for seed in range(100):
        env.reset(seed=seed)
        xs.append(_base_x_direction(env.unwrapped))
    xs = np.array(xs)
    assert np.allclose(np.abs(xs), 1, atol=1e-5), "flip keeps the peg along the x-axis, as in v1.1"
    assert 0.3 < (xs > 0).mean() < 0.7, f"base end towards +x in {(xs > 0).mean():.0%} of resets"
    env.close()


def test_success_needs_the_base_end_down():
    env = _make()
    task = env.unwrapped
    env.reset(seed=0)
    flat = float(task.compute_normalized_dense_reward(None, None, task.evaluate()))
    _stand(task, base_down=True)
    info = task.evaluate()
    assert bool(info["success"])
    assert np.isclose(float(task.compute_normalized_dense_reward(None, None, info)), 1.0)
    _stand(task, base_down=False)
    info = task.evaluate()
    assert not bool(info["success"]), "standing on the decoy end must not count"
    wrong = float(task.compute_normalized_dense_reward(None, None, info))
    assert wrong < flat + 0.05, f"the decoy end ({wrong:.3f}) should earn no more than lying flat ({flat:.3f})"
    print(f"      (normalized reward: lying flat {flat:.3f}, on the decoy end {wrong:.3f}, on the base 1.0)")
    env.close()


def test_decoy_false_leaves_the_top_plain():
    env = _make(decoy=False)
    env.reset(seed=0)
    assert env.unwrapped.peg_markings[0]["top"] == "none"
    env.close()


def test_perturbations_still_apply():
    env = _make(perturbations={"table_material": "randomize-train", "object_colour": "randomize-train"})
    env.reset(seed=0)
    task = env.unwrapped
    assert task.reconfiguration_freq == 1
    assert set(task.episode_perturbations) == {"object_colour", "table_material"}
    env.close()


if __name__ == "__main__":
    tests = [
        test_marking_splits,
        test_bad_kwargs_are_rejected,
        test_dynamics_are_v1_1s,
        test_reset_is_seeded_and_places_the_peg_as_v1_1_does,
        test_flip_yaw_turns_the_base_end_both_ways,
        test_success_needs_the_base_end_down,
        test_decoy_false_leaves_the_top_plain,
        test_perturbations_still_apply,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
