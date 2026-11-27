"""PickCube-MultiArmDistractors (the shared distractor scene on PickCube): `python -m pytest tests/test_pick_cube_distractors.py`.

physx_cpu at num_envs=1 throughout, since sapien can only enable GPU PhysX once per process.
"""

import numpy as np
import torch

import custom_maniskill_tasks  # noqa: F401  (registers the ids under test)
from custom_maniskill_tasks import make_env
from custom_maniskill_tasks.distractor_scene import SCENE_CAMERA_EYE, SceneCameraMixin
from custom_maniskill_tasks.distractor_tasks import LIFT_PEG_MAX_START_DELAY
from mani_skill.utils.registration import REGISTERED_ENVS

TASK = "PickCube-MultiArmDistractors"


def _zero(env):
    return torch.zeros(env.action_space.shape)


def test_registered_with_stock_horizon_and_control_mode():
    spec = REGISTERED_ENVS[TASK]
    assert spec.max_episode_steps == REGISTERED_ENVS["PickCube-v1"].max_episode_steps == 50
    assert spec.default_kwargs["control_mode"] == "pd_ee_delta_pos"
    assert issubclass(spec.cls, SceneCameraMixin)


def test_scene_camera_goal_pos_and_plain_panda():
    env = make_env(TASK)
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    assert list(obs["sensor_data"]) == ["base_camera"], "no wrist camera"
    assert "goal_pos" in obs["extra"], "the goal marker is hidden, so goal_pos is the only route to the target"
    assert u.agent.uid == "panda" and env.action_space.shape == (4,)
    eye = u._sensors["base_camera"].camera.global_pose.p[0].numpy()
    np.testing.assert_allclose(eye, SCENE_CAMERA_EYE, atol=1e-5)
    assert u.distractor_names == ("left", "right")
    env.close()


def test_control_scene_has_no_distractors():
    env = make_env(TASK, distractor_robots=[])
    env.reset(seed=0)
    u = env.unwrapped
    assert u.distractors == {} and u.distractor_cubes == {}
    assert not any(k.startswith("distractor") for k in u.get_state_dict()["actors"])
    env.step(_zero(env))
    env.close()


def test_still_distractors_hold_the_same_scene():
    """`distractors_still` keeps the seed's scene and removes only the motion."""
    states = {}
    for still in (False, True):
        env = make_env(TASK, distractors_still=still)
        env.reset(seed=3)
        u = env.unwrapped
        start = u.distractor_state()
        for _ in range(40):
            env.step(torch.as_tensor(env.action_space.sample()))
        states[still] = (start, u.distractor_state())
        env.close()
    (moving0, moving1), (still0, still1) = states[False], states[True]
    for key in still0:
        torch.testing.assert_close(still0[key], moving0[key])
    assert (moving1["qpos"] - moving0["qpos"]).abs().max() > 0.1, "the moving distractors never moved"
    torch.testing.assert_close(still1["qpos"], still0["qpos"], atol=1e-4, rtol=0)
    torch.testing.assert_close(still1["object_p"], still0["object_p"], atol=1e-3, rtol=0)


def test_success_does_not_terminate():
    env = make_env(TASK, ignore_terminations=False)
    env.reset(seed=0)
    u = env.unwrapped
    # the goal moved onto the resting cube (a cube teleported up to the goal falls during the step)
    u.goal_site.set_pose(u.cube.pose)
    _, _, terminated, truncated, info = env.step(_zero(env))
    assert bool(info["success"].reshape(-1)[0]), "no success to suppress"
    assert not bool(terminated.reshape(-1)[0]) and not bool(truncated.reshape(-1)[0])
    env.close()


def test_task_draws_match_v1_1_per_seed():
    """The distractors draw from their own stream: same seed, same cube, goal and agent start as v1.1."""
    def first_state(task):
        env = make_env(task, robot_uids="panda")
        env.reset(seed=3)
        u = env.unwrapped
        out = u.cube.pose.raw_pose.clone(), u.goal_site.pose.p.clone(), u.agent.robot.get_qpos().clone()
        env.close()
        return out

    for a, b in zip(first_state("PickCube-v1.1"), first_state(TASK)):
        torch.testing.assert_close(a, b)


def test_tour_distractors_start_early_and_lift_within_the_episode():
    """The pre-2026-11-23 cube tour (`distractor_task="pick_place"`)."""
    for seed in (0, 1, 2, 3):
        env = make_env(TASK, distractor_task="pick_place")
        env.reset(seed=seed)
        u = env.unwrapped
        assert int(u._distractor_delay.max()) <= LIFT_PEG_MAX_START_DELAY
        peak = {n: 0.0 for n in u.distractor_names}
        for _ in range(50):
            env.step(_zero(env))
            for n in u.distractor_names:
                peak[n] = max(peak[n], float(u.distractor_cubes[n].pose.p[0, 2]))
        assert all(v > 0.15 for v in peak.values()), (seed, peak)
        assert float(u.cube.pose.p[0, 2]) < 0.03, "the task's cube moved without the agent touching it"
        env.close()


def test_object_hue_leaves_distractor_cubes_alone():
    def colours(perturbations):
        import sapien

        env = make_env(TASK, perturbations=perturbations)
        env.reset(seed=0)
        u = env.unwrapped

        def first_colour(actor):
            body = actor._objs[0].find_component_by_type(sapien.render.RenderBodyComponent)
            return np.array(body.render_shapes[0].parts[0].material.base_color)

        out = first_colour(u.distractor_cubes["left"]), first_colour(u.cube)
        env.close()
        return out

    (theirs0, ours0), (theirs1, ours1) = colours(None), colours({"object_hue": {"degrees": 120}})
    np.testing.assert_allclose(theirs0, theirs1)
    assert not np.allclose(ours0, ours1), "object_hue did not reach the task's own cube"


def test_distractor_cubes_look_like_the_task_cube_and_no_pads():
    import sapien

    env = make_env(TASK)
    env.reset(seed=0)
    u = env.unwrapped

    def material(actor):
        body = actor._objs[0].find_component_by_type(sapien.render.RenderBodyComponent)
        m = body.render_shapes[0].parts[0].material
        return np.array([*m.base_color, m.roughness, m.specular, m.metallic])

    for cube in u.distractor_cubes.values():
        np.testing.assert_allclose(material(cube), material(u.cube))
    names = [e.name for e in u.scene.sub_scenes[0].entities]
    assert not any("_pad_" in n for n in names), [n for n in names if "_pad_" in n]
    env.close()

    # the other distractor tasks keep their coloured cubes and pads
    env = make_env("LiftPegUpright-MultiArmDistractors")
    env.reset(seed=0)
    assert any("_pad_" in e.name for e in env.unwrapped.scene.sub_scenes[0].entities)
    env.close()


def test_scene_camera_frames_all_three_robots_the_cube_and_the_goal():
    """Every robot link, the cube and the (hidden) goal stay inside the frame (5% margin)."""
    env = make_env(TASK)
    u = env.unwrapped
    worst = 1.0
    for seed in (0, 2):
        env.reset(seed=seed)
        params = u._sensors["base_camera"].get_params()
        K, E = params["intrinsic_cv"][0].numpy(), params["extrinsic_cv"][0].numpy()
        size = u._sensors["base_camera"].config.width
        for t in range(50):
            if t % 5 == 0:
                points = [link.pose.p[0].numpy() for robot in [u.agent, *u.distractors.values()]
                          for link in robot.robot.get_links()]
                points += [u.cube.pose.p[0].numpy(), u.goal_site.pose.p[0].numpy()]
                for xyz in points:
                    cam = E @ np.r_[xyz, 1.0]
                    assert cam[2] > 0
                    uv = (K @ cam[:3])[:2] / cam[2]
                    worst = min(worst, *(uv / size), *(1 - uv / size))
            env.step(_zero(env))
    assert worst > 0.05, f"a point came within {worst:.0%} of the frame edge"
    env.close()


def test_tour_start_pins_the_distractors_and_nothing_else():
    """`distractor_tour_start` changes where the distractors' cubes start; the task's draws, the
    per-episode delays and the cube jitter stay what the seed gives (the distractor probe's conditions)."""
    import pytest
    from custom_maniskill_tasks.distractor_robots import STATIONS

    states = {}
    for start in (0, 2):
        env = make_env("PickCube-MultiArmDistractors", distractor_task="pick_place", distractor_tour_start=start)
        env.reset(seed=5)
        u = env.unwrapped
        assert (u._distractor_start == start).all()
        states[start] = dict(cube=u.cube.pose.raw_pose.clone(), goal=u.goal_site.pose.p.clone(),
                             delay=u._distractor_delay.clone(), state=u.distractor_state())
        env.close()
    torch.testing.assert_close(states[0]["cube"], states[2]["cube"], rtol=0, atol=0)
    torch.testing.assert_close(states[0]["goal"], states[2]["goal"], rtol=0, atol=0)
    torch.testing.assert_close(states[0]["delay"], states[2]["delay"], rtol=0, atol=0)
    s0, s2 = states[0]["state"], states[2]["state"]
    assert s0["qpos"].shape == (1, 2, 9) and s0["object_p"].shape == (1, 2, 3) and s0["object_q"].shape == (1, 2, 4)
    # the cubes sit on different pads of the tour, jittered the same way
    for k, name in enumerate(("left", "right")):
        tour = STATIONS[name].tour
        pads = np.array([STATIONS[name].pads[i] for i in tour])
        offset0 = s0["object_p"][0, k, :2].numpy() - pads[0]
        offset2 = s2["object_p"][0, k, :2].numpy() - pads[2]
        np.testing.assert_allclose(offset0, offset2, atol=1e-6)
    with pytest.raises(ValueError, match="distractor_tour_start"):
        make_env("PickCube-MultiArmDistractors", distractor_task="pick_place", distractor_tour_start=4)
    with pytest.raises(ValueError, match="distractor_tour_start"):  # the pick_cube distractors have no tour
        make_env("PickCube-MultiArmDistractors", distractor_tour_start=0)


def _idle(env):
    a = _zero(env)
    a[-1] = 1.0
    return a


def test_pick_cube_distractors_do_the_task_and_hold_the_goal():
    """The default distractors (`distractor_task="pick_cube"`) each pick their own cube up and hold it at their goal."""
    from custom_maniskill_tasks import YAW_CONTROL_MODE

    env = make_env(TASK)
    for seed in (0, 1, 2):
        env.reset(seed=seed)
        u = env.unwrapped
        assert u.distractor_task == "pick_cube" and u.task_alignment_probability == 0.5
        assert {r.control_mode for r in u.distractors.values()} == {YAW_CONTROL_MODE}
        assert env.action_space.shape == (4,), "the agent keeps PickCube's own action space"
        for _ in range(50):
            env.step(_idle(env))
        assert all(bool(ok[0]) for ok in u.distractor_success().values()), (seed, u.distractor_success())
        assert float(u.cube.pose.p[0, 2]) < 0.03, "the task's cube moved without the agent touching it"
    env.close()


def test_task_alignment_copies_the_agents_set_up_relative_to_each_base():
    import pytest
    from custom_maniskill_tasks.distractor_robots import STATIONS
    from custom_maniskill_tasks.pick_cube_distractor_robots import CUBE_SPAWN_HALF_SIZE, MAX_GOAL_HEIGHT, TASK_CENTRE_AHEAD

    for p, expect_aligned in ((1.0, True), (0.0, False)):
        env = make_env(TASK, task_alignment_probability=p)
        for seed in range(4):
            env.reset(seed=seed)
            u = env.unwrapped
            assert bool(u._distractor_aligned[:, 0].all()) == expect_aligned and bool(u._distractor_aligned[:, 0].any()) == expect_aligned
            agent_xy, agent_goal = u.cube.pose.p[0, :2].numpy(), u.goal_site.pose.p[0].numpy()
            for k, name in enumerate(u.distractor_names):
                centre = np.array([STATIONS[name].base[0] + TASK_CENTRE_AHEAD, STATIONS[name].base[1]])
                xy = u.distractor_cubes[name].pose.p[0, :2].numpy() - centre
                goal = u._distractor_goal[k, 0].numpy() - np.r_[centre, 0.0]
                assert np.all(np.abs(xy) <= CUBE_SPAWN_HALF_SIZE + 1e-6) and np.all(np.abs(goal[:2]) <= CUBE_SPAWN_HALF_SIZE + 1e-6)
                assert 0.02 - 1e-6 <= goal[2] <= 0.02 + MAX_GOAL_HEIGHT + 1e-6
                same = np.allclose(xy, agent_xy, atol=1e-6) and np.allclose(goal, agent_goal, atol=1e-6) and \
                    torch.allclose(u.distractor_cubes[name].pose.q[0], u.cube.pose.q[0], atol=1e-6)
                assert same == expect_aligned, (p, seed, name)
        env.close()
    # about half the stations align at the default, and the task's own draws are PickCube-v1.1's either way
    env = make_env(TASK)
    aligned = []
    for seed in range(40):
        env.reset(seed=seed)
        aligned += env.unwrapped._distractor_aligned[:, 0].tolist()
    env.close()
    assert 0.3 < np.mean(aligned) < 0.7, np.mean(aligned)
    for bad in (-0.1, 1.5, True):
        with pytest.raises(ValueError, match="task_alignment_probability"):
            make_env(TASK, task_alignment_probability=bad)
    with pytest.raises(ValueError, match="distractor_task"):
        make_env(TASK, distractor_task="peg_insertion")


def test_distractor_mode_leaves_the_task_draws_alone():
    poses = {}
    for task in ("pick_cube", "pick_place"):
        env = make_env(TASK, distractor_task=task)
        env.reset(seed=11)
        u = env.unwrapped
        poses[task] = (u.cube.pose.raw_pose.clone(), u.goal_site.pose.p.clone(), u.agent.robot.get_qpos().clone())
        env.close()
    for a, b in zip(*poses.values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
