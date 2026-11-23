"""The shared scene for tabletop tasks with distractor robots: the camera, the robot and the factory.

A task gets distractors from two mixins and one helper:

- `SceneCameraMixin` puts the agent on the plain `panda` (no wrist camera rendered for nothing) and
  observes the task through a `base_camera` in front of the table (`SCENE_CAMERA_EYE`), framed
  so the agent and both distractor stations (`distractor_robots.STATIONS`) stay in view. On its
  own, with no distractors, it gives the matching control scene.
- `DistractorRobotsMixin` (`distractor_robots`) adds the scripted Pandas and their cubes and pads.
- `BackdropVideoMixin` (`backdrop_video`) adds the `background_video` kwarg: off by default, or a
  looping video wall behind the scene (e.g. the factory clip).
- `distractor_task(base_cls, ...)` stacks all three onto a task's `-v1.1` class (no early
  termination, `perturbations` kwarg) and returns the new class, ready for `register_env`. Every
  `-MultiArmDistractors` id is built this way, so each takes `distractor_robots` and
  `background_video`, independently.

What makes a stock tabletop task fit: its objects stay near the table centre (the distractors' pads
are at |y| >= 0.64, their robots 0.7 m to either side), and the agent stands at the table scene's
usual (-0.615, 0, 0). Both hold for PegInsertionSide and LiftPegUpright. Nothing filters collisions,
so an agent that swings 0.7 m sideways can still hit a distractor.

Tasks with a horizon shorter than one pick-and-place (`distractor_robots.CYCLE_STEPS`, 64 steps)
should shorten `distractor_max_start_delay`, so the distractors are moving for most of every episode
rather than idling through a quarter of it.
"""

from __future__ import annotations

import numpy as np

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from custom_maniskill_tasks.backdrop_video import BackdropVideoMixin
from custom_maniskill_tasks.distractor_robots import DistractorRobotsMixin

SCENE_CAMERA_EYE = [2.095, 0.0, 1.425]
SCENE_CAMERA_TARGET = [-0.2, 0.0, 0.1]
SCENE_CAMERA_FOV = np.deg2rad(50)
"""The `base_camera` distractor scenes are observed through: in front of the table, looking back
along -x at 30 degrees from horizontal, so the agent and both distractor stations are in frame.
Chosen by projecting every link of all three robots, the peg, the distractor cubes and the pads
over seeded PegInsertionSide and LiftPegUpright rollouts (agent arm driven randomly), and taking
the tightest pose that keeps them all at least 6% of the image in from the edges: the distractors'
reach sideways is what sets the zoom. The narrow fov from further back flattens the perspective.
Replaced 2026-11-20 a higher view (eye (1.1, 0, 1.6), target (-0.2, 0, 0), fov 70, ~51 degrees
down) that read as bird's-eye; the peg is ~14% larger in this one. Runs launched before then used
the old view. It depends on the stations, not the task, so every distractor task shares it."""


class SceneCameraMixin:
    """The agent on the plain `panda`, observed through the front `base_camera` (module doc)."""

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(SCENE_CAMERA_EYE, SCENE_CAMERA_TARGET)
        return [CameraConfig("base_camera", pose, 128, 128, SCENE_CAMERA_FOV, 0.01, 100)]


def distractor_task(base_cls: type, name: str, doc: str, **class_attrs) -> type:
    """`base_cls` (a `-v1.1` task class) with the scene camera, the plain panda, distractor robots
    and the (default off) background video.

    `class_attrs` override per-task tuning, e.g. `distractor_max_start_delay`. A `base_cls` that
    already has `SceneCameraMixin` in its bases is not given it twice.
    """
    scene = () if issubclass(base_cls, SceneCameraMixin) else (SceneCameraMixin,)
    bases = (DistractorRobotsMixin, BackdropVideoMixin, *scene, base_cls)
    if "panda" not in getattr(base_cls, "SUPPORTED_ROBOTS", ["panda"]):
        raise ValueError(f"{base_cls.__name__} does not support the plain panda the distractor scene uses")
    return type(name, bases, {"__doc__": doc, "__module__": __name__, **class_attrs})
