"""The wrist camera's viewpoint: where it is, as a shift from where the robot mounts it.

A `ViewpointConfig` is a shift in the camera's own frame -- a translation (m), yaw, pitch and roll
(rad) and an absolute fov -- and `wrist_camera_local_pose` is the camera's pose in its link's frame
once shifted, which is what `sensor_configs` takes. The `viewpoint` perturbation class
(`perturbation_classes`) draws the shift; `PerturbationClassesMixin` poses the camera with it, and
`check_viewpoint` asserts it got there. Only the wrist camera (`camera_view="wrist"`) can be moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from transforms3d.quaternions import qmult, quat2mat

from custom_maniskill_tasks.cameras import (
    PANDA_RIG,
    WRIST_CAMERA_UID,
    WristRig,
)

NOMINAL_WRIST_CAMERA_POSITION = PANDA_RIG.nominal_position
NOMINAL_WRIST_CAMERA_QUATERNION = PANDA_RIG.nominal_quaternion  # wxyz
"""Where the Panda's wrist camera is in the world at reset, before any qpos noise
(`robot_init_qpos_noise`, which moves it a few centimetres either way): looking straight down from
0.24 m, its image "up" being world +x and its image "left" world +y. Measured off the built env
rather than derived. Each `WristRig` carries its own; these are `PANDA_RIG`'s."""

@dataclass(frozen=True)
class ViewpointConfig:
    """A wrist camera's shift, in its own frame: metres, radians, and an absolute fov (or None)."""

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    fov: float | None = None
    rig: WristRig = field(default=PANDA_RIG, compare=False)
    """The camera this shift is a shift of; not part of what makes two shifts equal."""

    @property
    def is_default(self) -> bool:
        return self == ViewpointConfig()

    @property
    def field_of_view(self) -> float:
        return self.rig.fov if self.fov is None else self.fov

    def delta(self) -> tuple[np.ndarray, np.ndarray]:
        """(position, wxyz quaternion) of the shifted camera relative to the nominal one."""
        def about(axis: str, angle: float) -> np.ndarray:
            v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
            return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * np.array(v))])
        q = qmult(qmult(about("z", self.yaw), about("y", self.pitch)), about("x", self.roll))
        return np.array(self.translation), q


DEFAULT_VIEWPOINT_CONFIG = ViewpointConfig()


def wrist_camera_local_pose(config: ViewpointConfig) -> list[float]:
    """[px, py, pz, qw, qx, qy, qz]: the camera's pose in the frame of the link it is mounted on
    (`panda_hand`, `fr3_hand_tcp`), shifted. A plain list, as `sensor_configs` takes and as json
    records."""
    import sapien

    p, q = config.delta()
    pose = config.rig.local_pose * sapien.Pose(p=p, q=q)
    return [float(x) for x in (*pose.p, *pose.q)]


def wrist_camera_world_pose(config: ViewpointConfig) -> tuple[np.ndarray, np.ndarray]:
    """(position, rotation matrix) of the shifted camera in the world at reset, before noise."""
    p, q = config.delta()
    rotation = quat2mat(config.rig.nominal_quaternion)
    return config.rig.nominal_position + rotation @ p, rotation @ quat2mat(q)


def rig_for_task(task_name: str) -> WristRig:
    """The wrist camera rig the robot of `task_name` carries: `PANDA_RIG` unless the task says
    otherwise (`WRIST_RIG`, which the FR3 tasks set)."""
    from mani_skill.utils.registration import REGISTERED_ENVS

    spec = REGISTERED_ENVS.get(task_name)
    return getattr(spec.cls, "WRIST_RIG", PANDA_RIG) if spec is not None else PANDA_RIG


def check_viewpoint(env, config: ViewpointConfig) -> None:
    """Assert the wrist camera in the built env is where `config` put it. The failure it catches
    is a shift that silently did not happen: no wrist camera to move (a different `camera_view`),
    or a camera mounted somewhere other than `WRIST_CAMERA_LOCAL_POSE`."""
    if config.is_default:
        return
    sensors = env.unwrapped._sensors
    if WRIST_CAMERA_UID not in sensors:
        raise ValueError(
            f"a viewpoint was asked for, but this env has no {WRIST_CAMERA_UID!r} to move; use "
            "camera_view='wrist'."
        )
    camera = sensors[WRIST_CAMERA_UID]
    built = camera.camera.local_pose
    expected = wrist_camera_local_pose(config)
    got = [*built.p, *built.q]
    if not np.allclose(got, expected, atol=1e-4) and not np.allclose(got, [*expected[:3], *(-np.array(expected[3:]))], atol=1e-4):
        raise ValueError(
            f"the wrist camera is at {np.round(got, 4).tolist()}, not the {np.round(expected, 4).tolist()} "
            "the viewpoint asked for: something other than the viewpoint posed it."
        )
    if not np.isclose(camera.config.fov, config.field_of_view, atol=1e-4):
        raise ValueError(f"the wrist camera's fov is {camera.config.fov}, not {config.field_of_view}.")
