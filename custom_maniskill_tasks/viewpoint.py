"""`viewpoint`: move the wrist camera, or change its field of view.

A viewpoint is a keyword or several joined with "+" (`"back-3+pitch-up-5"`), each a name and a
non-negative number -- there are no signs, since "+" joins keywords and "-" is part of the names:

| keyword | meaning |
| --- | --- |
| `left|right|up|down-<cm>` | move the camera that far across its own image plane |
| `forward|back-<cm>` | move it that far along its optical axis |
| `pitch-up|pitch-down-<deg>` | turn it, so the view tilts up / down |
| `yaw-left|yaw-right-<deg>` | turn it, so the view swings left / right |
| `roll-left|roll-right-<deg>` | roll it, left side down / right side down |
| `fov-<deg>` | set the field of view outright (the default is `WRIST_CAMERA_FOV`, 108) |

The same axis named twice adds up, so `left-2+right-2` is no shift. `"default"` (or an empty
list) leaves the camera where the robot mounts it.

It is an ordinary env kwarg (`ViewpointMixin`): the pose and fov are merged into the env's
`sensor_configs` for the wrist camera, and the keyword string is what gets recorded. Only the wrist
camera (`camera_view="wrist"`) can be moved. The geometry helpers here are also what the
distractors use to keep their objects in view.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from transforms3d.quaternions import mat2quat, qmult, quat2mat

from custom_maniskill_tasks.cameras import (
    DEFAULT_CAMERA_RESOLUTION,
    WRIST_CAMERA_FOV,
    WRIST_CAMERA_LOCAL_POSE,
    WRIST_CAMERA_UID,
)

DEFAULT_VIEWPOINT = "default"

NOMINAL_WRIST_CAMERA_POSITION = np.array([0.0465, 0.02, 0.2372])
NOMINAL_WRIST_CAMERA_QUATERNION = np.array([0.70710678, 0.0, 0.70710678, 0.0])  # wxyz
"""Where the wrist camera is in the world at reset, before any qpos noise (`robot_init_qpos_noise`,
which moves it a few centimetres either way): looking straight down from 0.24 m, its image "up"
being world +x and its image "left" world +y. Measured off the built env rather than derived."""

_AXIS = re.compile(
    r"^(?P<name>left|right|up|down|forward|back|pitch-up|pitch-down|yaw-left|yaw-right|"
    r"roll-left|roll-right|fov)-(?P<value>\d+(?:\.\d+)?)$"
)

_TRANSLATION_AXES = {  # unit vectors in the camera frame (x forward, y left, z up)
    "left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0), "down": (0.0, 0.0, -1.0),
    "forward": (1.0, 0.0, 0.0), "back": (-1.0, 0.0, 0.0),
}
_ROTATION_AXES = {  # (axis, sign) of the right-handed rotation, in the camera frame
    "pitch-up": ("y", -1.0), "pitch-down": ("y", 1.0),
    "yaw-left": ("z", 1.0), "yaw-right": ("z", -1.0),
    "roll-left": ("x", -1.0), "roll-right": ("x", 1.0),
}


@dataclass(frozen=True)
class ViewpointConfig:
    """A wrist camera's shift, in its own frame: metres, radians, and an absolute fov (or None)."""

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    fov: float | None = None

    @property
    def is_default(self) -> bool:
        return self == ViewpointConfig()

    @property
    def field_of_view(self) -> float:
        return WRIST_CAMERA_FOV if self.fov is None else self.fov

    def delta(self) -> tuple[np.ndarray, np.ndarray]:
        """(position, wxyz quaternion) of the shifted camera relative to the nominal one."""
        def about(axis: str, angle: float) -> np.ndarray:
            v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
            return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * np.array(v))])
        q = qmult(qmult(about("z", self.yaw), about("y", self.pitch)), about("x", self.roll))
        return np.array(self.translation), q


DEFAULT_VIEWPOINT_CONFIG = ViewpointConfig()


def canonical_viewpoint(viewpoint: str | Sequence[str] | ViewpointConfig | None) -> ViewpointConfig:
    """Resolve a `viewpoint` into a `ViewpointConfig`, rejecting unknown keywords early -- a
    misspelled one that fell through would build a working env from the unshifted viewpoint."""
    if isinstance(viewpoint, ViewpointConfig):
        return viewpoint
    if viewpoint is None:
        return DEFAULT_VIEWPOINT_CONFIG
    parts = viewpoint.split("+") if isinstance(viewpoint, str) else list(viewpoint)
    translation = np.zeros(3)
    angles = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    fov = None
    for part in parts:
        if part in (DEFAULT_VIEWPOINT, ""):
            continue
        match = _AXIS.match(part)
        if match is None:
            raise ValueError(
                f"Unknown viewpoint {part!r}, expected 'default' or a \"+\"-joined stack of "
                "'left|right|up|down|forward|back-<cm>' (e.g. 'back-3'), "
                "'pitch-up|pitch-down|yaw-left|yaw-right|roll-left|roll-right-<degrees>' "
                "(e.g. 'pitch-up-5') and 'fov-<degrees>' (e.g. 'fov-90')."
            )
        name, value = match.group("name"), float(match.group("value"))
        if name == "fov":
            if not 10.0 <= value <= 170.0:
                raise ValueError(f"fov-{value:g} is outside 10 to 170 degrees.")
            fov = float(np.deg2rad(value))
        elif name in _TRANSLATION_AXES:
            translation += np.array(_TRANSLATION_AXES[name]) * value / 100.0
        else:
            axis, sign = _ROTATION_AXES[name]
            angles[{"z": "yaw", "y": "pitch", "x": "roll"}[axis]] += float(sign * np.deg2rad(value))
    return ViewpointConfig(
        translation=tuple(float(t) for t in translation),
        yaw=angles["yaw"], pitch=angles["pitch"], roll=angles["roll"], fov=fov,
    )


def wrist_camera_local_pose(config: ViewpointConfig) -> list[float]:
    """[px, py, pz, qw, qx, qy, qz]: the camera's pose in the frame of `panda_hand`, shifted. A
    plain list, as `sensor_configs` takes and as json records."""
    import sapien

    p, q = config.delta()
    pose = WRIST_CAMERA_LOCAL_POSE * sapien.Pose(p=p, q=q)
    return [float(x) for x in (*pose.p, *pose.q)]


def wrist_camera_world_pose(config: ViewpointConfig) -> tuple[np.ndarray, np.ndarray]:
    """(position, rotation matrix) of the shifted camera in the world at reset, before noise."""
    p, q = config.delta()
    rotation = quat2mat(NOMINAL_WRIST_CAMERA_QUATERNION)
    return NOMINAL_WRIST_CAMERA_POSITION + rotation @ p, rotation @ quat2mat(q)


def project_to_wrist_image(
    points: np.ndarray, config: ViewpointConfig = DEFAULT_VIEWPOINT_CONFIG,
    resolution: int = DEFAULT_CAMERA_RESOLUTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, v, depth) of world `points` (..., 3) in the wrist camera's image, u across from the left
    and v down from the top, in pixels of a square `resolution` image. Depth is along the optical
    axis, so a point behind the camera has depth <= 0."""
    position, rotation = wrist_camera_world_pose(config)
    in_camera = (np.asarray(points) - position) @ rotation
    depth = in_camera[..., 0]
    focal = (resolution / 2) / np.tan(config.field_of_view / 2)
    safe = np.where(depth > 1e-6, depth, 1e-6)
    return (
        resolution / 2 - focal * in_camera[..., 1] / safe,
        resolution / 2 - focal * in_camera[..., 2] / safe,
        depth,
    )


FINGER_LINE = 0.62
"""How far down the image (as a fraction of its height) the gripper's fingers begin to get in the
way; anything the distractors put below it could be hidden."""


def in_wrist_view(
    points: np.ndarray, config: ViewpointConfig = DEFAULT_VIEWPOINT_CONFIG, border: float = 0.06
) -> np.ndarray:
    """Whether each world point (..., 3) is inside the wrist image, `border` (a fraction of the
    image) in from every side and above the gripper's fingers."""
    u, v, depth = project_to_wrist_image(points, config, resolution=1)
    return (
        (depth > 0)
        & (u > border) & (u < 1 - border)
        & (v > border) & (v < FINGER_LINE)
    )


def visible_table_bounds(
    config: ViewpointConfig = DEFAULT_VIEWPOINT_CONFIG, height: float = 0.02
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """((x_min, x_max), (y_min, y_max)) of the smallest rectangle round the part of the table, at
    `height`, that `in_wrist_view` accepts; None if none of it is in view."""
    xs, ys = np.meshgrid(np.arange(-0.6, 1.0, 0.01), np.arange(-0.9, 0.9, 0.01))
    points = np.stack([xs, ys, np.full_like(xs, height)], axis=-1)
    inside = in_wrist_view(points, config)
    if not inside.any():
        return None
    return (
        (float(xs[inside].min()), float(xs[inside].max())),
        (float(ys[inside].min()), float(ys[inside].max())),
    )


class ViewpointMixin:
    """Gives a task the `viewpoint` kwarg: the wrist camera moved, turned or zoomed as
    `ViewpointConfig` says. Merges the pose and fov into `sensor_configs`, where ManiSkill applies
    them over the camera's own, so `camera_view="wrist"` is still what puts the camera there."""

    def __init__(self, *args, viewpoint=DEFAULT_VIEWPOINT, sensor_configs=None, **kwargs):
        self._viewpoint_spec = viewpoint
        self._viewpoint = canonical_viewpoint(viewpoint)
        if not self._viewpoint.is_default:
            sensor_configs = dict(sensor_configs or {})
            camera = dict(sensor_configs.get(WRIST_CAMERA_UID, {}))
            camera["pose"] = wrist_camera_local_pose(self._viewpoint)
            if self._viewpoint.fov is not None:
                camera["fov"] = self._viewpoint.fov
            sensor_configs[WRIST_CAMERA_UID] = camera
        super().__init__(*args, sensor_configs=sensor_configs, **kwargs)

    @property
    def viewpoint(self) -> ViewpointConfig:
        """The shift this env was built with; `DEFAULT_VIEWPOINT_CONFIG` unless asked otherwise."""
        return self._viewpoint


def supports_viewpoint(task_name: str) -> bool:
    from mani_skill.utils.registration import REGISTERED_ENVS

    spec = REGISTERED_ENVS.get(task_name)
    return spec is not None and issubclass(spec.cls, ViewpointMixin)


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
