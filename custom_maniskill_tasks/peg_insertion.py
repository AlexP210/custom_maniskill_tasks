"""PegInsertionSide with no early termination, and with distractor robots beside the agent.

- `PegInsertionSide-v1.1`: `PegInsertionSide-v1` with no early termination and the `perturbations`
  kwarg, like every other `-v1.1` id, observed through the stock task's own `base_camera` (low and
  close, looking along +y from beside the box). One other difference: the robot is the plain
  `panda`, not `panda_wristcam`. The stock task only ships the wrist-camera Panda, which renders a
  `hand_camera` into every observation; this id is observed through the scene camera, so the wrist
  camera is dropped rather than rendered for nothing. The arm, the start pose (set by the task
  itself, not the table scene) and the dynamics are the same. A `wrist` camera can still be listed in
  `cameras` if it is ever wanted. Stock geometry: the box turned up
  to 22.5 deg, 3 mm clearance.

  Its default control mode is `pd_ee_delta_pos_yaw` (`PEG_INSERTION_CONTROL_MODE`, 5-D: position
  delta, yaw delta, gripper; see `controllers`), not the 4-D `pd_ee_delta_pos` of every other id
  here. With position-only control the hand never turns, a grasped peg always lies along y, and
  only ~1 stock layout in 10 is insertable at all (the two position-only distractor runs scored
  0). Yaw is the one rotation a top-down grasp needs: a closed-loop scripted policy in this action
  space solves 15/20 stock layouts within 200 steps, 11/20 within the stock 100-step horizon
  (median 80 steps; measured 2026-11-23 with the tracking lag of ~0.04 rad/step). Until then the
  default was the 7-D `pd_ee_delta_pose`.

- `PegInsertionSide-MultiArmDistractors` (formerly `PegInsertionSide-v2`, still registered as an
  alias): v1.1 with scripted Panda distractors to the agent's left and/or right
  (`distractor_robots`, default both), each pick-and-placing a cube of its own between two pads
  (see `distractor_robots`), in the shared distractor scene (`distractor_scene`): the scene camera
  moves from the stock view to one in front of the table, 30° down (`SCENE_CAMERA_EYE`), so all
  three robots are in frame; the peg and box are correspondingly smaller in it (the peg is ~14 px
  long at 224 x 224). `background_video` (default None) adds a looping video wall behind the
  scene (`backdrop_video`). Task, reward, success, agent observations and control mode are
  v1.1's; the same seed gives the same peg, box and start pose in both. (Until 2026-11-23 it
  used the 4-D `pd_ee_delta_pos`, and runs on it are not comparable with ones after.)

`PegInsertionSide-v3` (solvable geometry for position-only control, wrist camera, randomized
layout) was removed on 2026-11-23: yaw control makes the stock geometry solvable, and the video
backdrop it introduced is now on every distractor task.

The stock task's horizon (100 steps) and reconfiguration default (a new peg and box every reset at
`num_envs=1`, one draw held for the run above it) are carried over.
"""

from __future__ import annotations

from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils.registration import register_env

from custom_maniskill_tasks.controllers import YAW_CONTROL_MODE
from custom_maniskill_tasks.peg_distractor_robots import PegInsertionDistractorsMixin
from custom_maniskill_tasks.distractor_scene import (  # noqa: F401  (camera constants re-exported)
    SCENE_CAMERA_EYE,
    SCENE_CAMERA_FOV,
    SCENE_CAMERA_TARGET,
    SceneCameraMixin,
    distractor_task,
)
from custom_maniskill_tasks.cameras import CamerasMixin
from custom_maniskill_tasks.lighting import LightingMixin
from custom_maniskill_tasks.perturbation_mixin import PerturbationClassesMixin
from custom_maniskill_tasks.table_material import TableMaterialMixin
from custom_maniskill_tasks.tasks import FullHorizonMixin, _horizon, register_alias

PEG_INSERTION_TASKS = {
    "PegInsertionSide-v1.1": "PegInsertionSide-v1",
    "PegInsertionSide-MultiArmDistractors": "PegInsertionSide-v1.1",
}
"""Each registered id and the id it derives from."""

PEG_INSERTION_CONTROL_MODE = YAW_CONTROL_MODE
"""Both ids' default control mode, 5-D (module doc); an explicit `control_mode=` still wins."""


@register_env(
    "PegInsertionSide-v1.1", max_episode_steps=_horizon("PegInsertionSide-v1"), control_mode=PEG_INSERTION_CONTROL_MODE
)
class PegInsertionSideFullHorizonEnv(
    FullHorizonMixin, PerturbationClassesMixin, CamerasMixin, LightingMixin, TableMaterialMixin, PegInsertionSideEnv
):
    """PegInsertionSide-v1 on the plain `panda` and its stock `base_camera`, with no early
    termination and the `perturbations` kwarg."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    object_of_interest = "peg"

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


PegInsertionSideDistractorEnv = register_env(
    "PegInsertionSide-MultiArmDistractors",
    max_episode_steps=_horizon("PegInsertionSide-v1"),
    control_mode=PEG_INSERTION_CONTROL_MODE,
)(
    distractor_task(
        PegInsertionSideFullHorizonEnv,
        "PegInsertionSideDistractorEnv",
        "PegInsertionSide-v1.1 with scripted distractor robots beside the agent (`distractor_robots`)"
        " and an optional video backdrop (`background_video`). `distractor_task` picks what the"
        " distractors do: 'pick_place' (default, cubes between pads) or 'peg_insertion' (a peg and"
        " box each, `peg_distractor_robots`).",
        __module__=__name__,
        distractors_mixin=PegInsertionDistractorsMixin,
    )
)
register_alias("PegInsertionSide-v2", "PegInsertionSide-MultiArmDistractors")
