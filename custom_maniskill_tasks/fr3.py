"""The Franka Research 3 and the ZED mini on it, as `-v1.2` tasks see them.

ManiSkill has no FR3, so this builds one from Franka's own `franka_description` (the URDF and
meshes under `assets/fr3`, see `assets/fr3/build_fr3_assets.py`). It has the same seven joints,
the same 0.1034 m TCP on the same Franka Hand and the same finger joints as the Panda, so
`Panda`'s controllers, grasp test and static test are reused as they are; what differs is the
joint limits and the link and joint names (`fr3_*`).

Two agents, so that only the wrist view pays for a camera:

- `fr3`: the arm and hand alone;
- `fr3_zed`: the same with the real robot's wrist camera, a ZED mini mounted on the hand. The
  numbers are those of the real mount and of the real camera, as measured on the real unit (see
  `ZED_CAMERA_SIM_CONFIG.md`): they are data, not tunables.

The camera is mounted on `fr3_hand_tcp`, which is the real `O_T_EE`, so it moves exactly with the
TCP. The mount bracket is not modelled, and no collision shape protects the lens.

## What the images are

The real camera's frame is the ZED SDK's rectified left image at HD720, 1280 x 720, whose
horizontal field of view is 82.504 degrees (from the SDK calibration `fx` = 729.734). The policy
is given 224 x 224, so the real side takes the **centre 720 x 720 crop and resizes it to 224 x
224**. Cropping the width to the height leaves a square pinhole image with the vertical field of
view the 720 rows always had, 52.517 degrees, on both axes, which is what is rendered here
(fx = fy = 227.1 px at 224). A real pipeline that does anything else (a squash, a different crop,
the raw instead of the rectified image) is not what this simulates.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import sapien
from transforms3d.quaternions import mat2quat

from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from custom_maniskill_tasks.cameras import (
    DEFAULT_CAMERA_RESOLUTION,
    WRIST_CAMERA_UID,
    WristRig,
)

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "fr3"

FR3_UID = "fr3"
FR3_ZED_UID = "fr3_zed"
FR3_UIDS = (FR3_UID, FR3_ZED_UID)

# --------------------------------------------------------------------------- #
# The real camera
# --------------------------------------------------------------------------- #
ZED_STREAM = (1280, 720)
"""(width, height) of the stream the calibration below was read from: HD720."""
ZED_FX_HD720 = 729.734
"""`left_cam.fx` of the SDK calibration, at `ZED_STREAM`. Only valid together with that width."""

ZED_HFOV = 2 * np.arctan(ZED_STREAM[0] / (2 * ZED_FX_HD720))
"""82.504 degrees: what the full rectified frame sees across. Not what is rendered, see below."""
ZED_CROPPED_FOV = 2 * np.arctan(ZED_STREAM[1] / (2 * ZED_FX_HD720))
"""52.517 degrees: the field of view of the centre `ZED_STREAM[1]` x `ZED_STREAM[1]` crop, on both
axes. This is the camera the policy sees (crop, then resize to `DEFAULT_CAMERA_RESOLUTION`)."""
ZED_NEAR = 0.002
"""A 2 mm near clip, so the lens never sees through close objects."""

# the real mount, in the `fr3_hand_tcp` frame
ZED_POSITION = (0.0642, -0.0315, -0.0640)
ZED_RPY_DEG = (0.0, -20.0, 90.0)
"""R = Ry(pitch) Rz(yaw) Rx(roll), in the ROS camera convention (+z forward, +x right, +y down)."""


def _ros_to_sapien_rotation() -> np.ndarray:
    """The rotation of the camera's frame in the TCP's, in sapien's camera convention (+x forward,
    +y left, +z up) rather than the ROS one the real mount is measured in."""
    roll, pitch, yaw = np.deg2rad(ZED_RPY_DEG)
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    ros = ry @ rz @ rx
    # sapien's forward / left / up are the ROS camera's +z / -x / -y
    return np.column_stack([ros[:, 2], -ros[:, 0], -ros[:, 1]])


ZED_LOCAL_POSE = sapien.Pose(p=list(ZED_POSITION), q=mat2quat(_ros_to_sapien_rotation()))
"""The ZED's pose in `fr3_hand_tcp`, in sapien's camera frame."""

FR3_ZED_RIG = WristRig(
    name="fr3_zed",
    fov=float(ZED_CROPPED_FOV),
    local_pose=ZED_LOCAL_POSE,
    # measured off the built env (noise-free reset, seed-independent); `tests/test_fr3.py` fails if
    # they drift from it. 0.23 m up, looking forward-down at 110 degrees about world y
    nominal_position=np.array([0.0642, 0.0315, 0.2338]),
    nominal_quaternion=np.array([0.573576, 0.0, 0.819152, 0.0]),  # wxyz
    # the fingertips first show 83% of the way down the image (found from the segmentation mask)
    finger_line=0.80,
    wrist_robot_uid=FR3_ZED_UID,
)
"""The ZED mini on the FR3: see the module docstring."""


# --------------------------------------------------------------------------- #
# The robot
# --------------------------------------------------------------------------- #
@register_agent()
class FR3(Panda):
    """Franka Research 3 with the Franka Hand: a Panda in everything but its limits and names."""

    uid = FR3_UID
    urdf_path = str(ASSET_DIR / "fr3.urdf")
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            fr3_leftfinger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            fr3_rightfinger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    # the Panda's rest pose is inside the FR3's limits (joint 4 stops at -0.117, joint 6 at 0.44)
    arm_joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
    gripper_joint_names = ["fr3_finger_joint1", "fr3_finger_joint2"]
    ee_link_name = "fr3_hand_tcp"

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        for controller in configs.values():
            controller["gripper"].mimic = {"fr3_finger_joint2": {"joint": "fr3_finger_joint1"}}
        return configs

    def _after_init(self):
        links = {link.name: link for link in self.robot.get_links()}
        self.finger1_link = links["fr3_leftfinger"]
        self.finger2_link = links["fr3_rightfinger"]
        self.finger1pad_link = links["fr3_leftfinger_pad"]
        self.finger2pad_link = links["fr3_rightfinger_pad"]
        self.tcp = links[self.ee_link_name]


@register_agent()
class FR3ZED(FR3):
    """`FR3` with the real robot's ZED mini on the hand, as `hand_camera`."""

    uid = FR3_ZED_UID

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid=WRIST_CAMERA_UID,
                pose=FR3_ZED_RIG.local_pose,
                width=DEFAULT_CAMERA_RESOLUTION,
                height=DEFAULT_CAMERA_RESOLUTION,
                fov=FR3_ZED_RIG.fov,
                near=ZED_NEAR,
                far=100,
                mount=self.robot.links_map[self.ee_link_name],
            )
        ]


# --------------------------------------------------------------------------- #
# The tasks
# --------------------------------------------------------------------------- #
class FR3Mixin:
    """Builds a tabletop task on the FR3 instead of the Panda.

    The stock tasks pick their starting pose, and in two places their reward, by comparing
    `robot_uids` to `"panda"` (`TableSceneBuilder.initialize`, `PickSingleYCB._initialize_episode`,
    `PickCube.compute_dense_reward`), and raise or silently do something else for any other robot.
    The FR3 is a Panda for all of what those branches use -- the same nine degrees of freedom in
    the same order, the same rest pose, the same TCP -- so while one of those hooks runs the task
    is told it has a Panda. Nothing else sees the swap: the agent is built, and the cameras and
    observations are made, with the real uid. `tests/test_fr3.py` checks that the reset TCP and
    the dense reward come out the same as the Panda's.
    """

    WRIST_RIG = FR3_ZED_RIG
    ROBOT_UID = FR3_UID

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        supported = list(getattr(cls, "SUPPORTED_ROBOTS", None) or [])
        cls.SUPPORTED_ROBOTS = supported + [u for u in FR3_UIDS if u not in supported]

    def __init__(self, *args, robot_uids=FR3_UID, **kwargs):
        if robot_uids not in FR3_UIDS:
            raise ValueError(f"{type(self).__name__} is the FR3 variant; robot_uids must be one of {FR3_UIDS}, got {robot_uids!r}.")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @contextmanager
    def _task_sees_a_panda(self):
        real = self.robot_uids
        self.robot_uids = "panda"
        try:
            yield
        finally:
            self.robot_uids = real

    def _initialize_episode(self, env_idx, options):
        with self._task_sees_a_panda():
            super()._initialize_episode(env_idx, options)

    def compute_dense_reward(self, obs, action, info):
        with self._task_sees_a_panda():
            return super().compute_dense_reward(obs, action, info)
