"""Central definitions of the ManiSkill tasks this project trains and plans on.

Importing this package registers the task ids this project's datasets were collected with --
`PushCube-v1.1`, `PlaceSphere-v1.1`, `LiftPegUpright-v1.1`, `PokeCube-v1.1`, `PickCube-v1.1` and
`PickSingleYCB-v1.1`, the stock tasks with early termination removed (see `tasks`), the `-FR3` ids
of the same six on the Franka Research 3 with the real robot's ZED mini wrist camera (see `fr3`),
and the `-HarderPerception` and `-MultiArmDistractors` variants. Old ids renamed on 2026-11-23
(`-v1.2`, `LiftPegUpright-v2`/`-v3`, `PickCube-v2`, `PegInsertionSide-v2`) stay registered as
aliases, listed in `DEPRECATED_TASK_IDS`. They are then available to plain `gym.make`, not only to
`make_env`.

`make_env` is the entry point; everything else is exported for the callers that build part of a
stack themselves (a dataset replay that needs `sensor_configs` for `RecordEpisode`, an agent that
inserts its own observation adapter between `FrameSkip` and `FrameStack`).
"""

from custom_maniskill_tasks.perturbation_classes import (
    CLASSES as PERTURBATION_CLASSES,
    EVALUATION_TABLE_MATERIALS,
    SPLITS as PERTURBATION_SPLITS,
    Perturbations,
    registry as perturbation_registry,
    resolve_perturbations,
)
from custom_maniskill_tasks.perturbation_mixin import (
    PerturbationClassesMixin,
    check_perturbations,
    supports_perturbations,
)
from custom_maniskill_tasks.backends import backend_kwargs, resolve_sim_backend
from custom_maniskill_tasks.cameras import (
    CAMERA_PRESETS,
    DEFAULT_CAMERA_RESOLUTION,
    FOCUSED_CAMERA_EYE,
    FOCUSED_CAMERA_FOV,
    FOCUSED_CAMERA_TARGET,
    TASK_CAMERA_UID,
    WRIST_CAMERA_FOV,
    PANDA_RIG,
    WRIST_CAMERA_UID,
    CameraSpec,
    CamerasMixin,
    WristRig,
    cameras_config,
    check_cameras,
    resolve_cameras,
)
from custom_maniskill_tasks.fr3 import FR3, FR3_UIDS, FR3_ZED_RIG, FR3ZED
from custom_maniskill_tasks.lighting import (
    DEFAULT_LIGHTING,
    DirectionalLight,
    LightingConfig,
    LightingMixin,
    apply_lighting,
    check_lighting,
    lighting_config,
    supports_lighting,
)
from custom_maniskill_tasks.make_env import TASKS_IN_USE, make_env, with_cameras_kwarg
from custom_maniskill_tasks.table_material import (
    DEFAULT_TABLE_MATERIAL,
    MATERIALS,
    TableMaterialMixin,
    canonical_table_material,
)
from custom_maniskill_tasks.viewpoint import ViewpointConfig, check_viewpoint

# imported for its side effect: adding the 5-D pd_ee_delta_pos_yaw control mode to the Panda
from custom_maniskill_tasks.controllers import YAW_CONTROL_MODE
# imported for its side effect: registering the -v1.1 and -FR3 task ids with ManiSkill and gymnasium
from custom_maniskill_tasks.tasks import (
    CONTROL_MODE,
    DEPRECATED_TASK_IDS,
    FR3_TASKS,
    FULL_HORIZON_TASKS,
    FullHorizonMixin,
    LiftPegUprightFullHorizonEnv,
    PickCubeFullHorizonEnv,
    PickSingleYCBFullHorizonEnv,
    PlugChargerFullHorizonEnv,
    PlaceSphereFullHorizonEnv,
    PokeCubeFullHorizonEnv,
    PushCubeFullHorizonEnv,
)
# LiftPegUpright-HarderPerception: stand the peg on its marked end (see `marked_peg`)
from custom_maniskill_tasks.marked_peg import (
    BASE_MARKINGS,
    MARKED_PEG_TASKS,
    MARKINGS,
    TOP_MARKINGS,
    LiftPegUprightMarkedEnv,
    marking_pairs,
)
# PegInsertionSide-v1.1 / -MultiArmDistractors (scripted distractor robots beside the agent); see
# `peg_insertion`. Every -MultiArmDistractors id also takes `background_video` (see `backdrop_video`)
from custom_maniskill_tasks.distractor_robots import STATIONS as DISTRACTOR_STATIONS, DistractorRobotsMixin
from custom_maniskill_tasks.peg_distractor_robots import DISTRACTOR_TASKS, PegInsertionDistractorsMixin
from custom_maniskill_tasks.pick_cube_distractor_robots import PICK_CUBE_DISTRACTOR_TASKS, PickCubeDistractorsMixin
from custom_maniskill_tasks.peg_insertion import (
    PEG_INSERTION_TASKS,
    PegInsertionSideDistractorEnv,
    PEG_INSERTION_CONTROL_MODE,
    PegInsertionSideFullHorizonEnv,
)
from custom_maniskill_tasks.backdrop_video import BackdropVideoMixin
# the shared distractor scene, and distractor versions of other -v1.1 tasks (LiftPegUpright-, PickCube-MultiArmDistractors)
from custom_maniskill_tasks.distractor_scene import (
    SCENE_CAMERA_EYE,
    SCENE_CAMERA_FOV,
    SCENE_CAMERA_TARGET,
    SceneCameraMixin,
    distractor_task,
)
from custom_maniskill_tasks.distractor_tasks import (
    DISTRACTOR_TASK_CLASSES,
    DISTRACTOR_TASKS,
    LiftPegUprightDistractorEnv,
    PickCubeDistractorEnv,
    PickCubeShuffledPositionsDistractorEnv,
)
from custom_maniskill_tasks.wrappers import (
    DINORewardWrapper,
    FrameSkip,
    FrameStack,
    IgnoreTerminations,
    clone_observations,
    stack_observations,
    stack_space,
)

__all__ = [
    "DISTRACTOR_STATIONS",
    "DISTRACTOR_TASKS",
    "DISTRACTOR_TASK_CLASSES",
    "LiftPegUprightDistractorEnv",
    "PickCubeDistractorEnv",
    "PickCubeShuffledPositionsDistractorEnv",
    "SCENE_CAMERA_EYE",
    "SCENE_CAMERA_FOV",
    "SCENE_CAMERA_TARGET",
    "SceneCameraMixin",
    "distractor_task",
    "DistractorRobotsMixin",
    "PegInsertionDistractorsMixin",
    "PickCubeDistractorsMixin",
    "PICK_CUBE_DISTRACTOR_TASKS",
    "DISTRACTOR_TASKS",
    "PEG_INSERTION_TASKS",
    "PegInsertionSideDistractorEnv",
    "PEG_INSERTION_CONTROL_MODE",
    "DEPRECATED_TASK_IDS",
    "YAW_CONTROL_MODE",
    "BackdropVideoMixin",
    "PegInsertionSideFullHorizonEnv",
    "BASE_MARKINGS",
    "MARKED_PEG_TASKS",
    "MARKINGS",
    "TOP_MARKINGS",
    "LiftPegUprightMarkedEnv",
    "marking_pairs",
    "EVALUATION_TABLE_MATERIALS",
    "PERTURBATION_CLASSES",
    "PERTURBATION_SPLITS",
    "Perturbations",
    "PerturbationClassesMixin",
    "check_perturbations",
    "perturbation_registry",
    "resolve_perturbations",
    "supports_perturbations",
    "CAMERA_PRESETS",
    "CONTROL_MODE",
    "DEFAULT_CAMERA_RESOLUTION",
    "DEFAULT_LIGHTING",
    "DINORewardWrapper",
    "DirectionalLight",
    "DEFAULT_TABLE_MATERIAL",
    "MATERIALS",
    "TableMaterialMixin",
    "ViewpointConfig",
    "canonical_table_material",
    "check_viewpoint",
    "FULL_HORIZON_TASKS",
    "FOCUSED_CAMERA_EYE",
    "FOCUSED_CAMERA_FOV",
    "FOCUSED_CAMERA_TARGET",
    "TASK_CAMERA_UID",
    "FrameSkip",
    "FR3",
    "FR3_TASKS",
    "FR3_UIDS",
    "FR3_ZED_RIG",
    "FR3ZED",
    "FrameStack",
    "FullHorizonMixin",
    "IgnoreTerminations",
    "LightingConfig",
    "LightingMixin",
    "LiftPegUprightFullHorizonEnv",
    "PANDA_RIG",
    "CameraSpec",
    "CamerasMixin",
    "PickCubeFullHorizonEnv",
    "PickSingleYCBFullHorizonEnv",
    "PlugChargerFullHorizonEnv",
    "PlaceSphereFullHorizonEnv",
    "PokeCubeFullHorizonEnv",
    "PushCubeFullHorizonEnv",
    "TASKS_IN_USE",
    "with_cameras_kwarg",
    "WRIST_CAMERA_FOV",
    "WRIST_CAMERA_UID",
    "WristRig",
    "apply_lighting",
    "backend_kwargs",
    "cameras_config",
    "check_cameras",
    "resolve_cameras",
    "check_lighting",
    "clone_observations",
    "make_env",
    "resolve_sim_backend",
    "stack_observations",
    "stack_space",
    "lighting_config",
    "supports_lighting",
]
