"""PegInsertionSide with no early termination, and with distractor robots beside the agent.

- `PegInsertionSide-v1.1`: `PegInsertionSide-v1` with no early termination and the `perturbations`
  kwarg, like every other `-v1.1` id. Two differences from the stock task besides: the scene
  camera (see below), and the robot is the plain `panda`, not `panda_wristcam`. The stock task only ships the wrist-camera Panda, which
  renders a `hand_camera` into every observation; these tasks are observed through the scene camera
  (`base_camera`), so the wrist camera is dropped rather than rendered for nothing. The arm, the
  start pose (set by the task itself, not the table scene) and the dynamics are the same.
  `camera_view="wrist"` still works through `make_env` (as on PushCube-v1.1) if it is ever wanted.
- `PegInsertionSide-v2`: v1.1 with scripted Panda distractors to the agent's left and/or right
  (`distractor_robots`, default both), each pick-and-placing a cube of its own between two pads
  (see `distractor_robots`). Task, reward, success, action and agent observations are v1.1's;
  only the scene around them is busier. The same seed gives the same peg, box and start pose in
  v1.1 and v2.

Both ids move the scene camera (`base_camera`) from the stock task's low, close view along +y to a
high view from in front of the table (`SCENE_CAMERA_EYE`), so all three robots are in frame; the
peg and box are correspondingly smaller in it (the peg is ~12 px long at 224 x 224).

The stock task's horizon (100 steps) and reconfiguration default (a new peg and box every reset at
`num_envs=1`, one draw held for the run above it) are carried over.
"""

from __future__ import annotations

import numpy as np

from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

from custom_maniskill_tasks.distractor_robots import DistractorRobotsMixin
from custom_maniskill_tasks.lighting import LightingMixin
from custom_maniskill_tasks.perturbation_mixin import PerturbationClassesMixin
from custom_maniskill_tasks.table_material import TableMaterialMixin
from custom_maniskill_tasks.tasks import CONTROL_MODE, FullHorizonMixin, _horizon

SCENE_CAMERA_EYE = [1.1, 0.0, 1.6]
SCENE_CAMERA_TARGET = [-0.2, 0.0, 0.0]
SCENE_CAMERA_FOV = np.deg2rad(70)
"""The `base_camera` both ids are observed through: high in front of the table, looking back along
-x at about 50 degrees, so the agent and both distractor stations are in frame. Chosen by projecting
every link of all three robots over seeded rollouts and taking the tightest pose that keeps them
clear of the edges (closest link origin ~8% of the image in from the sides, with room to spare
above and below: the distractors' reach sideways is what sets the zoom). The stock camera, (0, -0.3, 0.2) looking along +y, sees the box head-on but has
the agent's right (-y) behind it. Used for v1.1 too, so v1.1 and v2 differ only by the distractors."""

PEG_INSERTION_TASKS = {
    "PegInsertionSide-v1.1": "PegInsertionSide-v1",
    "PegInsertionSide-v2": "PegInsertionSide-v1.1",
}
"""Each registered id and the id it derives from."""


@register_env(
    "PegInsertionSide-v1.1", max_episode_steps=_horizon("PegInsertionSide-v1"), control_mode=CONTROL_MODE
)
class PegInsertionSideFullHorizonEnv(
    FullHorizonMixin, PerturbationClassesMixin, LightingMixin, TableMaterialMixin, PegInsertionSideEnv
):
    """PegInsertionSide-v1 on the plain `panda`, with no early termination and the `perturbations` kwarg."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    object_of_interest = "peg"

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(SCENE_CAMERA_EYE, SCENE_CAMERA_TARGET)
        return [CameraConfig("base_camera", pose, 128, 128, SCENE_CAMERA_FOV, 0.01, 100)]


@register_env(
    "PegInsertionSide-v2", max_episode_steps=_horizon("PegInsertionSide-v1"), control_mode=CONTROL_MODE
)
class PegInsertionSideDistractorEnv(DistractorRobotsMixin, PegInsertionSideFullHorizonEnv):
    """PegInsertionSide-v1.1 with scripted distractor robots beside the agent (`distractor_robots`)."""
