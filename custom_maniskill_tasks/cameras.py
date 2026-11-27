"""The cameras a task is observed through: an explicit list, built per env by `CamerasMixin`.

A task's observation cameras are given as a list of camera specs, the `cameras` env kwarg (and
`make_env`'s argument of the same name):

    cameras=[
        {"name": "wrist", "preset": "wrist", "resolution": 224},
        {"name": "scene", "preset": "task", "resolution": 128},
    ]

Each spec is one camera:

- `name`: what the camera is called downstream. Agents expose its frames as `<name>_rgb`
  (`CameraSpec.observation_key`); the env itself reports them under the sensor's uid, as ManiSkill
  always does (`obs["sensor_data"][uid]["rgb"]`).
- `preset`: what the camera is and where it sits (`CAMERA_PRESETS`):
  - `task`: the task's own external camera (`base_camera`), at its registered pose and fov. For
    the distractor tasks that is the shared front scene camera (`distractor_scene`).
  - `focused`: the same camera re-posed onto the tabletop workspace through a narrow 36 degree
    fov. Same uid as `task`, so the two cannot be combined.
  - `wrist`: a camera on the robot's hand (`hand_camera`): the robot's `WristRig` decides the
    mount, pose and fov -- the realsense pose the `-v1.1` datasets were rendered with on the
    Panda (at the wide fisheye `WRIST_CAMERA_FOV`), the real ZED mini on the FR3.
- `resolution`: square, in pixels (default 224, what the DINO encoders here consume).
- `fov`: optional, degrees; replaces the preset's.
- `pitch`: optional, degrees, `wrist` only: the camera turned about its own horizontal axis on
  its mount, positive towards the top of its image (on the Panda, forward along the hand, where a
  grasped peg's head points). 0 is the straight-down realsense pose every dataset was rendered with.
- `offset`: optional, metres, `wrist` only: `[x, y, z]`, the camera moved on its mount (as on a
  bracket) before it is tilted, in its own untilted frame: x along its view (negative lifts it off
  the hand), y to the image's left, z to the image's top (on the Panda, forward along the hand;
  negative moves it back, away from the fingertips). `offset: [-0.04, 0, 0]` with `pitch: 30` (4 cm
  off the hand, tilted 30 degrees forward) frames the fingertips, a grasped peg and PegInsertionSide's
  hole together (the two-camera PegInsertion task); moving it back instead hides the view behind the
  hand housing.

The env renders exactly the listed cameras, in that order, and nothing else: a robot that carries
cameras of its own (`panda_wristcam`, `fr3_zed`) is refused unless the list asks for exactly that
camera, which then takes the spec's resolution and fov.

This replaces the earlier `camera_view` ("default" | "focused" | "wrist"), which could only ever
give one camera, and whose wrist view registered a camera-carrying Panda for the whole process
(every later env in it gained a hand camera). Here a camera is a per-env config, recorded in
`env.spec.kwargs` like any other kwarg, so a replay of a recorded trajectory rebuilds the same
cameras. The pose and fov numbers are the ones every recorded dataset in this project was rendered
with, so treat them as data, not tunables.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

DEFAULT_CAMERA_RESOLUTION = 224
"""Square resolution a camera renders at unless its spec says otherwise. 224 because that is what
the DINO(v2/v3) encoders consuming these frames want; ManiSkill's own task defaults are 128."""

WRIST_CAMERA_FOV = 0.6 * np.pi
"""The Panda wrist camera's fov: deliberately much wider than the realsense ManiSkill models, for
the fisheye look every `-v1.1` wrist dataset was rendered with."""

FOCUSED_CAMERA_EYE = [0.3, 0.0, 0.9]
FOCUSED_CAMERA_TARGET = [-0.1, 0.0, -0.3]
FOCUSED_CAMERA_FOV = np.pi / 5
"""The `focused` preset: a narrow (36 degree) view from above and in front, on the tabletop workspace."""

TASK_CAMERA_UID = "base_camera"
"""The uid of the task's own external camera (`task` and `focused` presets). Every tabletop task in
this project registers exactly one external camera under it."""

WRIST_CAMERA_UID = "hand_camera"
"""The uid of the wrist camera (`wrist` preset), as on ManiSkill's `panda_wristcam`."""

CAMERA_PRESETS = {"task": TASK_CAMERA_UID, "focused": TASK_CAMERA_UID, "wrist": WRIST_CAMERA_UID}
"""Each preset and the sensor uid it renders under."""

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


WRIST_CAMERA_LOCAL_POSE = sapien.Pose(
    p=[0.035, 0, 0.036], q=euler2quat(0, -1.5707, 3.1415)
) * sapien.Pose(p=[0, 0.02, 0.0115])
"""Where the Panda's wrist camera sits in the frame of `panda_hand` (x forward, y left, z up in the
camera's own frame, as sapien has it): where `panda_wristcam` mounts its realsense."""


@dataclass(frozen=True, eq=False)
class WristRig:
    """One robot's wrist camera: where it is mounted, and what the viewpoint perturbation needs.

    A rig is a property of the robot a task is built on, not of the task: `PANDA_RIG` is the
    realsense every `-v1.1` dataset was rendered with, `fr3.FR3_ZED_RIG` the ZED mini on the
    real FR3. The task class says which (`WRIST_RIG`).
    """

    name: str
    fov: float
    """Vertical field of view (rad), sapien's convention; the images are square, so it is also the horizontal one."""
    local_pose: sapien.Pose
    """The camera's pose in the frame of `mount_link`, in sapien's camera frame (x forward, y left, z up)."""
    nominal_position: np.ndarray
    nominal_quaternion: np.ndarray
    """Where the camera is in the world at reset, before qpos noise (`robot_init_qpos_noise`), wxyz.
    Measured off the built env, and checked against it by `tests/test_fr3.py`."""
    finger_line: float
    """How far down the image (fraction of its height) the gripper's fingers begin to get in the way."""
    mount_link: str = "panda_hand"
    """The robot link the camera is mounted on."""
    near: float = 0.01


PANDA_RIG = WristRig(
    name="panda",
    fov=WRIST_CAMERA_FOV,
    local_pose=WRIST_CAMERA_LOCAL_POSE,
    nominal_position=np.array([0.0465, 0.02, 0.2372]),
    nominal_quaternion=np.array([0.70710678, 0.0, 0.70710678, 0.0]),  # looking straight down from 0.24 m
    finger_line=0.62,
)
"""The realsense on the Panda's hand, at the 108 degree fov the `-v1.1` datasets use."""


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CameraSpec:
    """One observation camera (module doc)."""

    name: str
    preset: str
    resolution: int = DEFAULT_CAMERA_RESOLUTION
    fov: float | None = None
    """Degrees, replacing the preset's; None keeps it."""
    pitch: float | None = None
    """Degrees, `wrist` only: the camera tilted on its mount, positive towards the top of its image."""
    offset: tuple[float, float, float] | None = None
    """Metres, `wrist` only: the camera moved on its mount before the tilt, in its untilted frame (module doc)."""

    def __post_init__(self):
        if not isinstance(self.name, str) or not _NAME.match(self.name):
            raise ValueError(f"camera name {self.name!r} must be lowercase letters, digits and _, starting with a letter")
        if self.preset not in CAMERA_PRESETS:
            raise ValueError(f"camera {self.name!r}: unknown preset {self.preset!r}, expected one of {sorted(CAMERA_PRESETS)}")
        if isinstance(self.resolution, bool) or not isinstance(self.resolution, (int, np.integer)) or self.resolution <= 0:
            raise ValueError(f"camera {self.name!r}: resolution must be a positive integer, got {self.resolution!r}")
        if self.fov is not None and not 0 < float(self.fov) < 180:
            raise ValueError(f"camera {self.name!r}: fov is in degrees, in (0, 180), got {self.fov!r}")
        if self.pitch is not None:
            if self.preset != "wrist":
                raise ValueError(f"camera {self.name!r}: pitch tilts a camera on its mount, so it needs the wrist preset")
            if not -90 <= float(self.pitch) <= 90:
                raise ValueError(f"camera {self.name!r}: pitch is in degrees, in [-90, 90], got {self.pitch!r}")
        if self.offset is not None:
            if self.preset != "wrist":
                raise ValueError(f"camera {self.name!r}: offset moves a camera on its mount, so it needs the wrist preset")
            try:
                offset = tuple(float(v) for v in self.offset)
            except (TypeError, ValueError):
                offset = ()
            if len(offset) != 3 or not all(np.isfinite(offset)) or max(map(abs, offset)) > 0.2:
                raise ValueError(f"camera {self.name!r}: offset is [x, y, z] in metres, each within 0.2, got {self.offset!r}")
            object.__setattr__(self, "offset", offset)  # a hydra ListConfig / list becomes a hashable tuple

    @property
    def remounted(self) -> bool:
        """Whether `pitch` or `offset` moves the camera off its rig's mount."""
        return bool(self.pitch) or (self.offset is not None and any(self.offset))

    @property
    def uid(self) -> str:
        """The sensor uid the env renders this camera under: `obs["sensor_data"][uid]`."""
        return CAMERA_PRESETS[self.preset]

    @property
    def observation_key(self) -> str:
        """The key agents expose this camera's frames under."""
        return f"{self.name}_rgb"

    def to_config(self) -> dict:
        """A plain dict (json, `env.spec.kwargs`), with `fov` only when set."""
        out = {"name": self.name, "preset": self.preset, "resolution": int(self.resolution)}
        if self.fov is not None:
            out["fov"] = float(self.fov)
        if self.pitch is not None:
            out["pitch"] = float(self.pitch)
        if self.offset is not None:
            out["offset"] = list(self.offset)
        return out

    def mount_pose(self, rig: "WristRig") -> sapien.Pose:
        """The wrist camera's pose in its mount link's frame: the rig's, moved by `offset`, then tilted by `pitch`.

        Both in the camera's own untilted frame. The tilt is about its y (left) axis, as
        `viewpoint.ViewpointConfig` rotates it: a rotation of -pitch there turns the view towards
        the top of the image.
        """
        pose = rig.local_pose
        if self.offset is not None and any(self.offset):
            pose = pose * sapien.Pose(p=list(self.offset))
        if self.pitch:
            half = -np.deg2rad(float(self.pitch)) / 2
            pose = pose * sapien.Pose(q=[np.cos(half), 0.0, np.sin(half), 0.0])
        return pose


def resolve_cameras(cameras: Sequence[Mapping[str, Any] | CameraSpec] | None) -> tuple[CameraSpec, ...] | None:
    """Validate a `cameras` list into `CameraSpec`s. None stays None (the task's own sensors, untouched).

    Raises on an empty list, an unknown key, a repeated name, or two cameras on one sensor uid
    (`task` and `focused`, or two `wrist`s): each would otherwise build a different env than asked.
    """
    if cameras is None:
        return None
    if isinstance(cameras, (str, Mapping)) or not isinstance(cameras, Sequence) and not hasattr(cameras, "__iter__"):
        raise TypeError(f"cameras must be a list of camera specs, got {cameras!r}")
    specs = []
    for camera in cameras:
        if isinstance(camera, CameraSpec):
            specs.append(camera)
            continue
        camera = dict(camera)  # also a hydra DictConfig
        unknown = set(camera) - {f.name for f in dataclasses.fields(CameraSpec)}
        if unknown:
            raise ValueError(f"camera spec {camera!r} has unknown keys {sorted(unknown)}")
        specs.append(CameraSpec(**camera))
    if not specs:
        raise ValueError("cameras is empty; a task needs at least one observation camera (or None for its own)")
    for attribute in ("name", "uid"):
        values = [getattr(s, attribute) for s in specs]
        repeated = sorted({v for v in values if values.count(v) > 1})
        if repeated:
            raise ValueError(f"cameras {[s.to_config() for s in specs]} share a {attribute}: {repeated}")
    return tuple(specs)


def cameras_config(cameras) -> list[dict] | None:
    """The canonical, json-able form of a `cameras` list (what `env.spec.kwargs` records)."""
    specs = resolve_cameras(cameras)
    return None if specs is None else [s.to_config() for s in specs]


# --------------------------------------------------------------------------- #
# The mixin
# --------------------------------------------------------------------------- #
class CamerasMixin:
    """Gives a task the `cameras` kwarg (module doc): the env renders exactly those cameras.

    The task's own camera(s) come from `_task_sensor_configs`, which defaults to the stock
    `_default_sensor_configs`; a mixin that replaces the task's camera (`SceneCameraMixin`)
    overrides `_task_sensor_configs` instead, so the `task` preset finds it.
    """

    WRIST_RIG: WristRig = PANDA_RIG
    """The wrist camera this task's robot carries; the FR3 tasks override it."""

    def __init__(self, *args, cameras: Sequence[Mapping[str, Any]] | None = None, **kwargs):
        self.camera_specs = resolve_cameras(cameras)
        super().__init__(*args, **kwargs)

    def _task_sensor_configs(self):
        return super()._default_sensor_configs

    @property
    def _default_sensor_configs(self):
        own = self._task_sensor_configs()
        if self.camera_specs is None:
            return own
        own = {c.uid: c for c in (own if isinstance(own, (list, tuple)) else [own])}
        agent_cameras = self._agent_camera_uids()
        configs = []
        for spec in self.camera_specs:
            if spec.uid in agent_cameras:
                continue  # the robot brings it; sized in _setup_sensors
            configs.append(self._camera_config(spec, own))
        return configs

    def _agent_camera_uids(self) -> set[str]:
        agent = getattr(self, "agent", None)
        if agent is None:
            return set()
        return {c.uid for c in (agent._sensor_configs or [])}

    def _camera_config(self, spec: CameraSpec, own: dict) -> CameraConfig:
        res = int(spec.resolution)
        fov = None if spec.fov is None else float(np.deg2rad(spec.fov))
        if spec.preset == "task":
            if TASK_CAMERA_UID not in own:
                raise ValueError(
                    f"camera {spec.name!r} (preset 'task') wants the task's own {TASK_CAMERA_UID!r}, but "
                    f"{type(self).__name__} registers {sorted(own)}."
                )
            base = own[TASK_CAMERA_UID]
            return dataclasses.replace(base, width=res, height=res, fov=base.fov if fov is None else fov)
        if spec.preset == "focused":
            pose = sapien_utils.look_at(eye=FOCUSED_CAMERA_EYE, target=FOCUSED_CAMERA_TARGET)
            return CameraConfig(TASK_CAMERA_UID, pose, res, res, FOCUSED_CAMERA_FOV if fov is None else fov, 0.01, 100)
        rig = self.WRIST_RIG
        return CameraConfig(
            WRIST_CAMERA_UID, spec.mount_pose(rig), res, res, rig.fov if fov is None else fov, rig.near, 100,
            mount=self.agent.robot.links_map[rig.mount_link],
        )

    def _setup_sensors(self, options: dict):
        if self.camera_specs is not None:
            agent_cameras = self._agent_camera_uids()
            unwanted = agent_cameras - {s.uid for s in self.camera_specs}
            if unwanted:
                raise ValueError(
                    f"robot {self.agent.uid!r} carries its own camera(s) {sorted(unwanted)} that `cameras` "
                    f"does not list, and a robot's cameras cannot be removed; use a robot without them "
                    f"(e.g. robot_uids='panda') or list them."
                )
            # a camera the robot brings takes the spec's size and fov; the viewpoint perturbation
            # (PerturbationClassesMixin) merges its own pose over this afterwards
            custom = dict(self._custom_sensor_configs or {})
            for spec in self.camera_specs:
                if spec.uid in agent_cameras:
                    entry = dict(custom.get(spec.uid, {}), width=int(spec.resolution), height=int(spec.resolution))
                    if spec.fov is not None:
                        entry["fov"] = float(np.deg2rad(spec.fov))
                    if spec.remounted:
                        raise ValueError(
                            f"camera {spec.name!r}: pitch/offset need the wrist camera mounted by `cameras`, but "
                            f"robot {self.agent.uid!r} brings its own; use a robot without one (robot_uids='panda')."
                        )
                    custom[spec.uid] = entry
            self._custom_sensor_configs = custom
        super()._setup_sensors(options)
        if self.camera_specs is not None:
            # keep the listed order (BaseEnv puts the agent's cameras last)
            self._sensors = {s.uid: self._sensors[s.uid] for s in self.camera_specs}


def supports_cameras(task_cls: type) -> bool:
    return issubclass(task_cls, CamerasMixin)


def check_cameras(env, cameras) -> None:
    """Assert the built env renders exactly the listed cameras, at their resolutions.

    Catches what ManiSkill would otherwise let through silently: a `sensor_configs` override for a
    uid nothing has (ignored), a camera the robot added on its own, a size that did not apply.
    """
    specs = resolve_cameras(cameras)
    if specs is None:
        return
    sensors = getattr(env.unwrapped, "_sensors", {})
    if list(sensors) != [s.uid for s in specs]:
        raise ValueError(f"cameras {[s.to_config() for s in specs]} built sensors {list(sensors)}")
    for spec in specs:
        config = sensors[spec.uid].config
        if (config.width, config.height) != (spec.resolution, spec.resolution):
            raise ValueError(f"camera {spec.name!r} renders {config.width}x{config.height}, not {spec.resolution}")
