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


def test_distractors_start_early_and_lift_within_the_episode():
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
