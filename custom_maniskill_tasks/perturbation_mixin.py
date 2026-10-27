"""`PerturbationClassesMixin`: gives a task the `perturbations` kwarg (see `perturbation_classes`).

Every class that is off neutral is applied when the scene is built: each aspect is either fixed or
drawn from its split, one draw per episode shared by all parallel envs (ManiSkill reconfigures
every sub-scene together), seeded by the episode seed and the (class, aspect). Anything randomized
rebuilds the scene on every reset (`reconfiguration_freq=1`, the only way to change lights,
materials and textures the renderer has already taken, or to re-pose a camera); a config of fixed
values builds it once and costs nothing per reset.

What each class does to the scene:

- `lighting`: the scene's light levels, the key light's direction and the colour temperature
  (`lighting.lighting_config`, built by `LightingMixin`);
- `table_colour`: a gain on the table's R, G and B and a brightness (the lighting config's
  `table_tint`);
- `table_material`: the table's material (`TableMaterialMixin._set_table_material`);
- `object_colour`: an offset to each of the manipulated object's R, G and B;
- `table_texture`: a scale, shear, rotation and shift of the table's texture;
- `viewpoint`: the wrist camera moved, turned and zoomed, by posing it in `sensor_configs`.

Mixed in ahead of `LightingMixin`, so it sees the kwargs first and the reconfiguration last.
`env.unwrapped.episode_perturbations` says what the current episode drew.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from custom_maniskill_tasks.cameras import PANDA_RIG, WRIST_CAMERA_UID, WristRig
from custom_maniskill_tasks.appearance import _apply_object_colour, _apply_table_texture, table_tint
from custom_maniskill_tasks.lighting import lighting_config
from custom_maniskill_tasks.perturbation_classes import Perturbations, resolve_perturbations
from custom_maniskill_tasks.viewpoint import ViewpointConfig, wrist_camera_local_pose


def viewpoint_config(aspects: Mapping[str, float], rig: WristRig) -> ViewpointConfig:
    """The `ViewpointConfig` for drawn `viewpoint` aspects (cm, degrees; see the registry's
    units). The conventions are the camera frame's: x forward, y left, z up."""
    nominal = float(np.degrees(rig.fov))
    return ViewpointConfig(
        translation=(aspects["forward"] / 100.0, aspects["left"] / 100.0, aspects["up"] / 100.0),
        yaw=float(np.deg2rad(aspects["yaw"])),
        pitch=float(-np.deg2rad(aspects["pitch"])),  # positive tilts the view up
        roll=float(np.deg2rad(aspects["roll"])),
        fov=None if abs(aspects["fov"] - nominal) < 1e-9 else float(np.deg2rad(aspects["fov"])),
        rig=rig,
    )


class PerturbationClassesMixin:
    WRIST_RIG: WristRig = PANDA_RIG
    """The wrist camera this task's robot carries; the FR3 tasks override it."""

    def __init__(self, *args, perturbations: Mapping[str, Any] | None = None, sensor_configs=None, **kwargs):
        self._perturbations: Perturbations = resolve_perturbations(
            perturbations, fov_degrees=float(np.degrees(self.WRIST_RIG.fov))
        )
        self._episode_perturbations: dict[str, dict[str, float | str]] = {}
        chosen = self._perturbations.active
        if "object_colour" in chosen and getattr(type(self), "object_of_interest", None) is None:
            raise ValueError(
                f"{type(self).__name__} has no single object of interest, so 'object_colour' has "
                "no object to change."
            )
        if self._perturbations.is_randomized:
            frequency = kwargs.setdefault("reconfiguration_freq", 1)
            if frequency != 1:
                raise ValueError(
                    f"perturbations={self._perturbations.to_config()} is redrawn on every reset, "
                    f"which needs reconfiguration_freq=1, but reconfiguration_freq={frequency} was passed."
                )
        # a copy, since the wrist camera's entry is replaced at every reconfiguration
        sensor_configs = dict(sensor_configs or {})
        self._wrist_camera_base = dict(sensor_configs.get(WRIST_CAMERA_UID, {}))
        super().__init__(*args, sensor_configs=sensor_configs, **kwargs)

    @property
    def perturbations(self) -> Perturbations:
        """What this env was built with."""
        return self._perturbations

    @property
    def episode_perturbations(self) -> dict[str, dict[str, float | str]]:
        """What each perturbed class's aspects are for the current episode, as
        {class: {aspect: value}}: the fixed ones as given, the randomized ones as drawn. Empty if
        nothing is perturbed."""
        return {cls: dict(aspects) for cls, aspects in self._episode_perturbations.items()}

    def _reconfigure(self, options=dict()):
        seed = int(self._episode_seed[0]) % 2**63
        drawn = self._episode_perturbations = {
            cls: self._perturbations.draw(cls, seed) for cls in self._perturbations.active
        }
        if "lighting" in drawn or "table_colour" in drawn:
            self._set_lighting(lighting_config(
                **drawn.get("lighting", {}),
                **({"table_tint": table_tint(drawn["table_colour"])} if "table_colour" in drawn else {}),
            ))
        if "table_material" in drawn:
            self._set_table_material(drawn["table_material"]["material"])
        super()._reconfigure(options)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        drawn = self._episode_perturbations
        if "object_colour" in drawn:
            _apply_object_colour(self, [drawn["object_colour"][c] for c in "rgb"])
        if "table_texture" in drawn:
            _apply_table_texture(self, drawn["table_texture"])

    def _setup_sensors(self, options: dict):
        if "viewpoint" in self._episode_perturbations:
            shift = viewpoint_config(self._episode_perturbations["viewpoint"], self.WRIST_RIG)
            camera = dict(self._wrist_camera_base)
            camera["pose"] = wrist_camera_local_pose(shift)
            if shift.fov is not None:
                camera["fov"] = shift.fov
            self._custom_sensor_configs = {**self._custom_sensor_configs, WRIST_CAMERA_UID: camera}
        super()._setup_sensors(options)


def supports_perturbations(task_name: str) -> bool:
    from mani_skill.utils.registration import REGISTERED_ENVS

    spec = REGISTERED_ENVS.get(task_name)
    return spec is not None and issubclass(spec.cls, PerturbationClassesMixin)


def check_perturbations(env) -> None:
    """Assert the wrist camera in the built env is where this episode's `viewpoint` put it: the
    failure it catches is a shift that silently did not happen (no wrist camera to move, or
    something other than the viewpoint posing it)."""
    task = env.unwrapped
    drawn = getattr(task, "episode_perturbations", {}).get("viewpoint")
    if drawn is None:
        return
    from custom_maniskill_tasks.viewpoint import check_viewpoint

    check_viewpoint(env, viewpoint_config(drawn, task.WRIST_RIG))
