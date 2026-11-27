"""The `cameras` list (`custom_maniskill_tasks.cameras`): `python -m pytest tests/test_cameras.py`.

physx_cpu at num_envs=1 throughout, since sapien can only enable GPU PhysX once per process.
"""

import pytest

import custom_maniskill_tasks  # noqa: F401  (registers the ids under test)
from custom_maniskill_tasks import CameraSpec, cameras_config, make_env, resolve_cameras


def test_specs_validate_and_canonicalize():
    specs = resolve_cameras([{"name": "wrist", "preset": "wrist"}, CameraSpec("scene", "task", 64, fov=40)])
    assert [s.uid for s in specs] == ["hand_camera", "base_camera"]
    assert [s.observation_key for s in specs] == ["wrist_rgb", "scene_rgb"]
    assert cameras_config(specs) == [
        {"name": "wrist", "preset": "wrist", "resolution": 224},
        {"name": "scene", "preset": "task", "resolution": 64, "fov": 40.0},
    ]
    assert resolve_cameras(None) is None


@pytest.mark.parametrize("cameras, message", [
    ([], "empty"),
    ([{"name": "a", "preset": "task"}, {"name": "b", "preset": "focused"}], "share a uid"),
    ([{"name": "a", "preset": "task"}, {"name": "a", "preset": "wrist"}], "share a name"),
    ([{"name": "Wrist", "preset": "wrist"}], "lowercase"),
    ([{"name": "w", "preset": "side"}], "unknown preset"),
    ([{"name": "w", "preset": "wrist", "resolution": 0}], "positive integer"),
    ([{"name": "w", "preset": "wrist", "fov": 1.9}], None),  # degrees: 1.9 is legal, if odd
    ([{"name": "w", "preset": "wrist", "fov": 200}], "degrees"),
    ([{"name": "w", "preset": "wrist", "res": 64}], "unknown keys"),
])
def test_bad_specs_are_refused(cameras, message):
    if message is None:
        resolve_cameras(cameras)
        return
    with pytest.raises(ValueError, match=message):
        resolve_cameras(cameras)


def test_fov_override_reaches_the_camera():
    import numpy as np

    env = make_env("PushCube-v1.1", cameras=[{"name": "wrist", "preset": "wrist", "fov": 60}])
    assert np.isclose(env.unwrapped._sensors["hand_camera"].config.fov, np.deg2rad(60))
    env.close()


def test_a_robot_camera_is_sized_or_refused():
    # the stock PegInsertionSide-v1 ships only panda_wristcam, whose hand_camera the wrist preset adopts
    env = make_env("PegInsertionSide-v1", cameras=[{"name": "wrist", "preset": "wrist", "resolution": 80}])
    obs, _ = env.reset(seed=0)
    assert list(obs["sensor_data"]) == ["hand_camera"] and obs["sensor_data"]["hand_camera"]["rgb"].shape == (1, 80, 80, 3)
    env.close()
    # a robot camera the list does not ask for cannot be removed, so it is refused
    with pytest.raises(ValueError, match="carries its own camera"):
        make_env("PegInsertionSide-v1.1", cameras=[{"name": "scene", "preset": "task"}], robot_uids="panda_wristcam")


def test_distractor_tasks_task_preset_is_the_scene_camera():
    from custom_maniskill_tasks.distractor_scene import SCENE_CAMERA_EYE
    import numpy as np

    env = make_env("PickCube-MultiArmDistractors", cameras=[{"name": "scene", "preset": "task", "resolution": 96}])
    env.reset(seed=0)
    camera = env.unwrapped._sensors["base_camera"]
    assert np.allclose(camera.camera.global_pose.p[0].cpu().numpy(), SCENE_CAMERA_EYE, atol=1e-5)
    assert camera.config.width == 96
    env.close()


def test_wrist_pitch_tilts_the_camera_towards_the_top_of_its_image():
    """A point near the top edge of the untilted view moves towards the centre when the camera pitches up."""
    import numpy as np

    def top_point_row(pitch):
        env = make_env("PushCube-v1.1", sim_backend="physx_cpu",
                       cameras=[{"name": "wrist", "preset": "wrist", **({"pitch": pitch} if pitch else {})}])
        env.reset(seed=0)
        camera = env.unwrapped._sensors["hand_camera"]
        params = camera.get_params()
        K, E = params["intrinsic_cv"][0].cpu().numpy(), params["extrinsic_cv"][0].cpu().numpy()
        # a world point 0.3 m ahead of the hand along +x (the image's top at pitch 0) on the table
        hand = env.unwrapped.agent.robot.links_map["panda_hand"].pose.p[0].cpu().numpy()
        c = E @ np.r_[hand[0] + 0.3, hand[1], 0.0, 1.0]
        env.close()
        return float((K @ c[:3])[1] / c[2])

    assert top_point_row(30) > top_point_row(0) + 40
    with pytest.raises(ValueError, match="wrist preset"):
        resolve_cameras([{"name": "s", "preset": "task", "pitch": 10}])
    with pytest.raises(ValueError, match="undo the camera's `pitch`"):
        make_env("PushCube-v1.1", cameras=[{"name": "w", "preset": "wrist", "pitch": 10}],
                 perturbations={"viewpoint": {"pitch": 5}})
    assert cameras_config([{"name": "w", "preset": "wrist", "pitch": 30}]) == [
        {"name": "w", "preset": "wrist", "resolution": 224, "pitch": 30.0}]


def test_wrist_offset_moves_the_camera_on_its_mount():
    """`offset` is in the untilted camera frame: -x lifts the camera straight back along its view."""
    import numpy as np

    def position(**mount):
        env = make_env("PushCube-v1.1", sim_backend="physx_cpu", cameras=[{"name": "wrist", "preset": "wrist", **mount}])
        env.reset(seed=0)
        p = env.unwrapped._sensors["hand_camera"].camera.global_pose.p[0].cpu().numpy()
        env.close()
        return p

    base, lifted = position(), position(offset=[-0.04, 0, 0], pitch=30)
    # the stock mount looks straight down, so backing off along the view raises the camera 4 cm
    assert np.allclose(lifted - base, [0, 0, 0.04], atol=2e-3)
    for bad in ([0.0, 0.0], [0.3, 0, 0], ["a", 0, 0]):
        with pytest.raises(ValueError, match="offset is"):
            resolve_cameras([{"name": "w", "preset": "wrist", "offset": bad}])
    with pytest.raises(ValueError, match="wrist preset"):
        resolve_cameras([{"name": "s", "preset": "task", "offset": [0, 0, 0.01]}])
    with pytest.raises(ValueError, match="undo the camera"):
        make_env("PushCube-v1.1", cameras=[{"name": "w", "preset": "wrist", "offset": [-0.04, 0, 0]}],
                 perturbations={"viewpoint": {"pitch": 5}})
    assert cameras_config([{"name": "w", "preset": "wrist", "offset": (-0.04, 0, 0)}]) == [
        {"name": "w", "preset": "wrist", "resolution": 224, "offset": [-0.04, 0.0, 0.0]}]
