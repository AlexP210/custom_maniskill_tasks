"""The `-v1.2` ids, the Franka Research 3 and its ZED mini: `python tests/test_fr3.py [test_name ...]`.

physx_cpu at num_envs=1 throughout. With no arguments each test runs in a subprocess of its own,
because sapien's Vulkan renderer stops finding its device after about eight scenes have been built
in one process.
"""

import subprocess
import sys

import gymnasium as gym
import numpy as np
import sapien
import torch
from transforms3d.quaternions import mat2quat, quat2mat

import custom_maniskill_tasks  # noqa: F401  (registers the ids under test)
from custom_maniskill_tasks import FR3_TASKS, FR3_ZED_RIG, PANDA_RIG, axes, make_env
from custom_maniskill_tasks import fr3
from custom_maniskill_tasks.viewpoint import canonical_viewpoint, wrist_camera_world_pose
from mani_skill.utils.assets import is_data_source_downloaded
from mani_skill.utils.registration import REGISTERED_ENVS

ZERO_ACTION = np.zeros(4, dtype=np.float32)


def _scalar(x):
    return np.asarray(x.cpu() if isinstance(x, torch.Tensor) else x).reshape(-1)[0]


def _has_assets(task):
    return all(is_data_source_downloaded(a) for a in REGISTERED_ENVS[task].asset_download_ids or [])


def _rotation(pose: sapien.Pose) -> np.ndarray:
    return quat2mat(np.asarray(pose.q))


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #
def test_ids_are_registered_and_derive_from_v1_1():
    for task, (base, base_cls) in FR3_TASKS.items():
        assert task.endswith("-v1.2") and base.endswith("-v1.1")
        assert task in REGISTERED_ENVS and task in gym.registry, task
        spec, base_spec = REGISTERED_ENVS[task], REGISTERED_ENVS[base]
        assert issubclass(spec.cls, base_cls), "same task underneath"
        assert spec.max_episode_steps == base_spec.max_episode_steps
        assert spec.default_kwargs["control_mode"] == "pd_ee_delta_pos"
        assert spec.asset_download_ids == base_spec.asset_download_ids
        assert set(fr3.FR3_UIDS) <= set(spec.cls.SUPPORTED_ROBOTS)
        assert spec.cls.WRIST_RIG is FR3_ZED_RIG
        assert base_spec.cls.WRIST_RIG is PANDA_RIG, "the v1.1 tasks keep the Panda's camera"


def test_a_v1_2_task_cannot_be_built_on_another_robot():
    try:
        gym.make("PushCube-v1.2", robot_uids="panda")
    except ValueError as error:
        assert "FR3 variant" in str(error)
    else:
        raise AssertionError("PushCube-v1.2 built on a panda")


# --------------------------------------------------------------------------- #
# the robot
# --------------------------------------------------------------------------- #
def _reset_state(task, **kwargs):
    env = make_env(task, obs_mode="state", robot_init_qpos_noise=0.0, **kwargs)
    env.reset(seed=3)
    base = env.unwrapped
    out = (
        base.agent.tcp.pose.p[0].cpu().numpy(),
        base.agent.tcp.pose.q[0].cpu().numpy(),
        base.agent.robot.get_qpos()[0].cpu().numpy(),
        base.agent.robot.pose.p[0].cpu().numpy(),
    )
    env.close()
    return out


def _reset_pose_is_the_pandas(task):
    """The task logic that picks the start pose by robot uid must land the FR3 where the Panda is:
    same base, same joints, same TCP (kinematically, the FR3 is the Panda)."""
    base = FR3_TASKS[task][0]
    if not _has_assets(task):
        print(f"      (skipped {task}: YCB assets missing)")
        return
    panda, fr = _reset_state(base), _reset_state(task)
    assert np.allclose(panda[0], fr[0], atol=1e-4), (task, panda[0], fr[0])
    # a quaternion and its negative are the same rotation
    assert abs(abs(float(panda[1] @ fr[1])) - 1) < 1e-5, task
    assert np.allclose(panda[2], fr[2], atol=1e-5), task
    assert np.allclose(panda[3], fr[3]), task


def _reset_pose_test(task):
    def test():
        _reset_pose_is_the_pandas(task)

    test.__name__ = "test_reset_pose_is_the_pandas_" + task.split("-")[0]
    return test


RESET_POSE_TESTS = [_reset_pose_test(task) for task in FR3_TASKS]


def test_the_robot_is_the_fr3():
    env = make_env("PushCube-v1.2", obs_mode="state")
    env.reset(seed=0)
    agent = env.unwrapped.agent
    assert env.unwrapped.robot_uids == "fr3" and type(agent) is fr3.FR3
    names = [j.name for j in agent.robot.get_active_joints()]
    assert names == [f"fr3_joint{i}" for i in range(1, 8)] + ["fr3_finger_joint1", "fr3_finger_joint2"]
    # the FR3's own limits, not the Panda's (joint 4 stops at -0.117 rather than -0.07, joint 6 at 0.44 rather than -0.0175)
    limits = agent.robot.get_qlimits()[0].cpu().numpy()
    assert np.isclose(limits[3, 1], -0.1169, atol=1e-3) and np.isclose(limits[5, 0], 0.4398, atol=1e-3)
    assert agent.tcp.name == "fr3_hand_tcp" and agent.finger1pad_link.name == "fr3_leftfinger_pad"
    hand = next(l for l in agent.robot.get_links() if l.name == "fr3_hand")
    assert np.allclose((hand.pose.inv() * agent.tcp.pose).p[0].cpu().numpy(), [0, 0, 0.1034], atol=1e-5)
    # the gripper mirrors one finger onto the other
    gripper = env.unwrapped.agent.controller.controllers["gripper"]
    assert gripper.config.mimic == {"fr3_finger_joint2": {"joint": "fr3_finger_joint1"}}
    env.step(np.array([0, 0, 0, -1.0], dtype=np.float32))  # close it: nothing in the way, fingers meet
    for _ in range(10):
        env.step(np.array([0, 0, 0, -1.0], dtype=np.float32))
    q = env.unwrapped.agent.robot.get_qpos()[0, -2:].cpu().numpy()
    assert q.max() < 0.02 and abs(q[0] - q[1]) < 1e-3, q
    env.close()


def test_pick_cube_reward_is_computed_as_for_a_panda():
    """`PickCube`'s dense reward drops the finger velocities only for a panda. The FR3 has to go
    down that branch, or its static reward would count the fingers."""
    from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv

    seen = []
    original = PickCubeEnv.compute_dense_reward

    def spy(self, *args, **kwargs):
        seen.append(self.robot_uids)
        return original(self, *args, **kwargs)

    PickCubeEnv.compute_dense_reward = spy
    try:
        rewards = {}
        for task in ("PickCube-v1.1", "PickCube-v1.2"):
            env = make_env(task, obs_mode="state", robot_init_qpos_noise=0.0, reward_wrappers=())
            env.reset(seed=1)
            rewards[task] = float(_scalar(env.step(ZERO_ACTION)[1]))
            expected = "fr3" if task.endswith("1.2") else "panda"
            assert env.unwrapped.robot_uids == expected, "the swap must not outlive the hook"
            env.close()
    finally:
        PickCubeEnv.compute_dense_reward = original
    assert set(seen) == {"panda"}, seen
    assert abs(rewards["PickCube-v1.1"] - rewards["PickCube-v1.2"]) < 5e-3, rewards


# --------------------------------------------------------------------------- #
# the camera
# --------------------------------------------------------------------------- #
def test_extrinsics_are_the_real_mounts():
    pose = FR3_ZED_RIG.local_pose
    assert np.allclose(pose.p, [0.0642, -0.0315, -0.0640])
    forward, left, up = (_rotation(pose)[:, i] for i in range(3))
    # the real optical axis in the TCP frame (ZED_CAMERA_SIM_CONFIG.md)
    assert np.allclose(forward, [-0.342, 0, 0.940], atol=1e-3), forward
    # and the real mount's quaternion (ROS convention), converted by hand: sapien's forward/left/up are ROS z/-x/-y
    ros = quat2mat([0.69636, -0.12279, -0.12279, 0.69636])
    assert np.allclose(forward, ros[:, 2], atol=1e-3)
    assert np.allclose(left, -ros[:, 0], atol=1e-3) and np.allclose(up, -ros[:, 1], atol=1e-3)
    assert np.isclose(np.linalg.det(_rotation(pose)), 1)


def test_intrinsics_are_the_real_cameras_at_224():
    assert np.isclose(np.degrees(fr3.ZED_HFOV), 82.504, atol=1e-3), "the SDK calibration's HFOV"
    assert np.isclose(np.degrees(FR3_ZED_RIG.fov), 52.517, atol=1e-3)
    env = make_env("PushCube-v1.2", camera_view="wrist")
    obs, _ = env.reset(seed=0)
    assert set(obs["sensor_data"]) == {"hand_camera"}, "only the wrist camera under wrist_only"
    assert obs["sensor_data"]["hand_camera"]["rgb"].shape == (1, 224, 224, 3)
    camera = env.unwrapped._sensors["hand_camera"]
    K = camera.camera.get_intrinsic_matrix()[0].cpu().numpy()
    # the real camera's fx at 1280 x 720, scaled to the 720 -> 224 crop-and-resize
    real_fx = fr3.ZED_FX_HD720 * 224 / 720
    assert np.isclose(K[0, 0], real_fx, rtol=1e-3) and np.isclose(K[1, 1], real_fx, rtol=1e-3), (K, real_fx)
    assert np.allclose([K[0, 2], K[1, 2]], [112, 112]), "principal point at the centre"
    assert np.isclose(camera.config.near, 0.002)
    built = camera.camera.local_pose
    assert np.allclose(built.p, FR3_ZED_RIG.local_pose.p, atol=1e-5)
    assert abs(abs(float(np.asarray(built.q) @ np.asarray(FR3_ZED_RIG.local_pose.q))) - 1) < 1e-5
    env.close()


def test_the_nominal_pose_is_where_the_camera_is_at_reset():
    env = make_env("PushCube-v1.2", camera_view="wrist", robot_init_qpos_noise=0.0)
    env.reset(seed=5)
    pose = env.unwrapped._sensors["hand_camera"].camera.global_pose
    position, rotation = wrist_camera_world_pose(canonical_viewpoint("default", FR3_ZED_RIG))
    assert np.allclose(pose.p[0].cpu().numpy(), position, atol=2e-3), (pose.p, position)
    built = quat2mat(pose.q[0].cpu().numpy())
    assert np.allclose(built, rotation, atol=1e-2), (built, rotation)
    env.close()


def test_the_wrist_robot_is_chosen_by_the_view():
    default = make_env("PushCube-v1.2", obs_mode="rgb", camera_view="default")
    assert default.unwrapped.robot_uids == "fr3"
    assert "hand_camera" not in default.unwrapped._sensors, "only the wrist view pays for the camera"
    default.close()
    wrist = make_env("PushCube-v1.2", obs_mode="rgb", camera_view="wrist")
    assert wrist.unwrapped.robot_uids == "fr3_zed"
    assert wrist.unwrapped.spec.kwargs["robot_uids"] == "fr3_zed", "recorded, so a replay builds the same robot"
    wrist.close()


# --------------------------------------------------------------------------- #
# viewpoint, distractors, axes
# --------------------------------------------------------------------------- #
def test_viewpoint_shifts_the_zed_and_not_the_panda():
    env = make_env("PushCube-v1.2", camera_view="wrist", viewpoint="back-2+pitch-up-5+fov-60")
    env.reset(seed=0)  # make_env already checked the built pose and fov against the config
    config = env.unwrapped.viewpoint
    assert config.rig is FR3_ZED_RIG and np.isclose(np.degrees(config.field_of_view), 60)
    assert np.isclose(np.degrees(canonical_viewpoint("default", FR3_ZED_RIG).field_of_view), 52.517, atol=1e-3)
    assert np.isclose(np.degrees(canonical_viewpoint("default").field_of_view), 108), "the Panda's is untouched"
    assert canonical_viewpoint("default", FR3_ZED_RIG).is_default
    env.close()


def test_distractors_are_in_the_zeds_view():
    """Every distractor is inside the view. What does *not* hold on this camera is the clearance
    `test_tasks` checks on the Panda's: the ZED sees about 16 x 21 cm of table, so the objects
    cannot all be kept clear of the task's own and each other (one distractor overlaps in about 4
    of 10 resets), and placement falls back to the roomiest visible spot, as it is documented to."""
    task = "PickCube-v1.2"
    if not _has_assets("PickCube-v1.2") or not is_data_source_downloaded("ycb"):
        print("      (skipped: YCB assets missing)")
        return
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from custom_maniskill_tasks.viewpoint import in_wrist_view

    env = make_env(task, camera_view="wrist", distractors=3)
    for seed in range(3):
        env.reset(seed=seed)
        for actor in env.unwrapped.distractors.values():
            xy = actor.pose.p[0, :2].cpu().numpy()
            assert in_wrist_view(np.array([[*xy, 0.0], [*xy, 0.02]]), env.unwrapped.viewpoint).all(), seed
    env.close()


def test_the_fov_axis_is_the_zeds():
    fov = axes.axis("viewpoint.fov", "fr3_zed")
    assert fov.neutral == 52.517 and fov is not axes.axis("viewpoint.fov")
    assert axes.axis("viewpoint.fov").neutral == 108, "the Panda's axis is unchanged"
    # the neutral value is the default view, to the precision the keyword has
    neutral = canonical_viewpoint(fov.condition(fov.neutral), FR3_ZED_RIG)
    assert np.isclose(neutral.field_of_view, FR3_ZED_RIG.fov, atol=1e-5)


TESTS = [
    test_ids_are_registered_and_derive_from_v1_1,
    test_a_v1_2_task_cannot_be_built_on_another_robot,
    *RESET_POSE_TESTS,
    test_the_robot_is_the_fr3,
    test_pick_cube_reward_is_computed_as_for_a_panda,
    test_extrinsics_are_the_real_mounts,
    test_intrinsics_are_the_real_cameras_at_224,
    test_the_nominal_pose_is_where_the_camera_is_at_reset,
    test_the_wrist_robot_is_chosen_by_the_view,
    test_viewpoint_shifts_the_zed_and_not_the_panda,
    test_distractors_are_in_the_zeds_view,
    test_the_fov_axis_is_the_zeds,
]

if __name__ == "__main__":
    by_name = {t.__name__: t for t in TESTS}
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            by_name[name]()
            print(f"ok  {name}")
    else:
        failed = []
        for test in TESTS:
            result = subprocess.run([sys.executable, __file__, test.__name__], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"ok  {test.__name__}")
            else:
                failed.append(test.__name__)
                print(f"FAIL {test.__name__}\n{result.stdout[-1500:]}\n{result.stderr[-3000:]}")
        print(f"\n{len(TESTS) - len(failed)} passed" + (f", {len(failed)} failed: {failed}" if failed else ""))
        sys.exit(1 if failed else 0)
