"""Central definitions of the ManiSkill tasks this project trains and plans on.

Importing this package registers the task ids this project's datasets were collected with --
`PushCube-v1.1`, `PlaceSphere-v1.1`, `LiftPegUpright-v1.1`, `PokeCube-v1.1`, `PickCube-v1.1` and
`PickSingleYCB-v1.1`, the stock tasks with early termination removed (see `tasks`), and the
`-v1.2` ids of the same six on the Franka Research 3 with the real robot's ZED mini wrist camera (see
`fr3`). They are then
available to plain `gym.make`, not only to `make_env`.

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
    CAMERA_VIEWS,
    DEFAULT_CAMERA_RESOLUTION,
    FOCUSED_CAMERA_EYE,
    FOCUSED_CAMERA_FOV,
    FOCUSED_CAMERA_TARGET,
    FOCUSED_CAMERA_UID,
    WRIST_CAMERA_FOV,
    PANDA_RIG,
    WRIST_CAMERA_UID,
    PandaHandCam,
    WristRig,
    build_sensor_configs,
    camera_view_applied,
    canonical_camera_view,
    check_camera_view,
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
from custom_maniskill_tasks.make_env import TASKS_IN_USE, make_env
from custom_maniskill_tasks.table_material import (
    DEFAULT_TABLE_MATERIAL,
    MATERIALS,
    TableMaterialMixin,
    canonical_table_material,
)
from custom_maniskill_tasks.viewpoint import ViewpointConfig, check_viewpoint

# imported for its side effect: registering the -v1.1 and -v1.2 task ids with ManiSkill and gymnasium
from custom_maniskill_tasks.tasks import (
    CONTROL_MODE,
    FR3_TASKS,
    FULL_HORIZON_TASKS,
    FullHorizonMixin,
    LiftPegUprightFullHorizonEnv,
    PickCubeFullHorizonEnv,
    PickSingleYCBFullHorizonEnv,
    PlaceSphereFullHorizonEnv,
    PokeCubeFullHorizonEnv,
    PushCubeFullHorizonEnv,
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
    "EVALUATION_TABLE_MATERIALS",
    "PERTURBATION_CLASSES",
    "PERTURBATION_SPLITS",
    "Perturbations",
    "PerturbationClassesMixin",
    "check_perturbations",
    "perturbation_registry",
    "resolve_perturbations",
    "supports_perturbations",
    "CAMERA_VIEWS",
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
    "FOCUSED_CAMERA_UID",
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
    "PandaHandCam",
    "PickCubeFullHorizonEnv",
    "PickSingleYCBFullHorizonEnv",
    "PlaceSphereFullHorizonEnv",
    "PokeCubeFullHorizonEnv",
    "PushCubeFullHorizonEnv",
    "TASKS_IN_USE",
    "WRIST_CAMERA_FOV",
    "WRIST_CAMERA_UID",
    "WristRig",
    "apply_lighting",
    "backend_kwargs",
    "build_sensor_configs",
    "camera_view_applied",
    "canonical_camera_view",
    "check_camera_view",
    "check_lighting",
    "clone_observations",
    "make_env",
    "resolve_sim_backend",
    "stack_observations",
    "stack_space",
    "lighting_config",
    "supports_lighting",
]
