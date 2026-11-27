"""Distractor-robot versions of the `-v1.1` tabletop tasks, built on the shared scene (`distractor_scene`).

PegInsertionSide's (`PegInsertionSide-MultiArmDistractors`) lives in `peg_insertion`, which predates
this module. Each one here is its `-v1.1` task plus `distractor_scene.distractor_task`, so each also
takes `background_video` (default off). Renamed 2026-11-23; the old ids (`LiftPegUpright-v3`,
`PickCube-v2`) stay registered as aliases (`tasks.DEPRECATED_TASK_IDS`):

- `LiftPegUpright-MultiArmDistractors`: `LiftPegUpright-v1.1` (peg stood upright; reward, success, horizon and
  `peg_yaw` kwarg unchanged) on the plain `panda`, observed through the shared front
  `base_camera`, with scripted Panda distractors to the agent's left and/or right
  (`distractor_robots`, default both). `distractor_robots=[]` gives the matching control scene: same
  camera and robot, no distractors. The same seed gives the same peg and agent start as v1.1.

  Its horizon is 50 steps, shorter than one 64-step pick-and-place, so the distractors start within
  `LIFT_PEG_MAX_START_DELAY` steps instead of the default 20. With that, each distractor approaches,
  grasps and lifts its cube and is mid-carry when the episode ends: there is motion in every episode,
  but no completed carry. The stock LiftPegUpright `base_camera` is replaced, as on PegInsertionSide;
  a `wrist` camera can be listed in `cameras` alongside (or instead of) it.

- `PickCube-MultiArmDistractors`: `PickCube-v1.1` (reward, success, the 50-step horizon and the hidden goal marker
  unchanged, so the goal still reaches the agent only through `goal_pos`) in the same scene, made
  harder to tell apart: the distractors' cubes are the task's own (same 4 cm size, same red, same
  plain material, `PICK_CUBE_COLOUR`), and their pads are not drawn, so nothing in the image marks
  which red cube is the agent's except its position. 50 steps, like LiftPegUpright, so the distractors
  start within `LIFT_PEG_MAX_START_DELAY` steps. `object_hue` recolours only the task's cube.
  Since 2026-11-23 each distractor does PickCube itself (`distractor_task="pick_cube"`, the default:
  its own cube and goal from the task's distributions, or with `task_alignment_probability`, default
  0.5, the agent's own set-up; `pick_cube_distractor_robots`). `distractor_task="pick_place"` is the
  earlier cube tour, which every run trained before that date saw.

- `PickCube-MultiArmDistractors-ShuffledPositions` (2026-11-24): PickCube-MultiArmDistractors (pick_cube
  distractors only) with the three robots' positions permuted each episode, the agent moving with
  its cube and goal, and the cubes coloured by robot: the agent's red, the distractors' plain blue and
  green (`shuffled_positions`).

To add a task: one `DISTRACTOR_TASKS` entry naming its `-v1.1` class (the factory checks that it
supports the plain panda), plus any tuning it needs.
"""

from __future__ import annotations

from mani_skill.utils.registration import REGISTERED_ENVS, register_env

from custom_maniskill_tasks.distractor_robots import DistractorRobotsMixin
from custom_maniskill_tasks.distractor_scene import distractor_task
from custom_maniskill_tasks.pick_cube_distractor_robots import PickCubeDistractorsMixin
from custom_maniskill_tasks.shuffled_positions import DISTRACTOR_CUBE_COLOURS, ShuffledPositionsMixin
from custom_maniskill_tasks.tasks import (
    CONTROL_MODE,
    LiftPegUprightFullHorizonEnv,
    PickCubeFullHorizonEnv,
    register_alias,
)

LIFT_PEG_MAX_START_DELAY = 5
"""The start delay for 50-step tasks (LiftPegUpright, PickCube): one pick-and-place is 64 steps."""
PICK_CUBE_COLOUR = (1.0, 0.0, 0.0, 1.0)
"""PickCube-v1's own cube colour (`pick_cube.py`: `build_cube(..., color=[1, 0, 0, 1])`)."""

DISTRACTOR_TASKS = {
    "LiftPegUpright-MultiArmDistractors": dict(
        old_id="LiftPegUpright-v3",
        base_id="LiftPegUpright-v1.1",
        base_cls=LiftPegUprightFullHorizonEnv,
        class_name="LiftPegUprightDistractorEnv",
        class_attrs=dict(distractor_max_start_delay=LIFT_PEG_MAX_START_DELAY),
    ),
    "PickCube-MultiArmDistractors": dict(
        old_id="PickCube-v2",
        base_id="PickCube-v1.1",
        base_cls=PickCubeFullHorizonEnv,
        class_name="PickCubeDistractorEnv",
        distractors_mixin=PickCubeDistractorsMixin,
        class_attrs=dict(
            distractor_max_start_delay=LIFT_PEG_MAX_START_DELAY,
            distractor_cube_colour=PICK_CUBE_COLOUR,
            distractor_pads=False,
        ),
    ),
    "PickCube-MultiArmDistractors-ShuffledPositions": dict(
        old_id=None,
        base_id="PickCube-v1.1",
        base_cls=PickCubeFullHorizonEnv,
        class_name="PickCubeShuffledPositionsDistractorEnv",
        distractors_mixin=ShuffledPositionsMixin,
        class_attrs=dict(
            distractor_max_start_delay=LIFT_PEG_MAX_START_DELAY,
            distractor_cube_colours=DISTRACTOR_CUBE_COLOURS,
            distractor_pads=False,
        ),
    ),
}
"""Each distractor id, its old id, the `-v1.1` id and class it derives from, its distractors' mixin
(default `DistractorRobotsMixin`, the cube tour), and its per-task tuning."""

DISTRACTOR_TASK_CLASSES: dict[str, type] = {}

for _id, _spec in DISTRACTOR_TASKS.items():
    _base = REGISTERED_ENVS[_spec["base_id"]]
    DISTRACTOR_TASK_CLASSES[_id] = register_env(
        _id,
        max_episode_steps=_base.max_episode_steps,
        asset_download_ids=_base.asset_download_ids,
        control_mode=CONTROL_MODE,
    )(
        distractor_task(
            _spec["base_cls"],
            _spec["class_name"],
            f"{_spec['base_id']} with the shared distractor scene (see `distractor_tasks`).",
            __module__=__name__,
            distractors_mixin=_spec.get("distractors_mixin", DistractorRobotsMixin),
            **_spec["class_attrs"],
        )
    )
    if _spec["old_id"] is not None:
        register_alias(_spec["old_id"], _id)

LiftPegUprightDistractorEnv = DISTRACTOR_TASK_CLASSES["LiftPegUpright-MultiArmDistractors"]
PickCubeDistractorEnv = DISTRACTOR_TASK_CLASSES["PickCube-MultiArmDistractors"]
PickCubeShuffledPositionsDistractorEnv = DISTRACTOR_TASK_CLASSES["PickCube-MultiArmDistractors-ShuffledPositions"]
