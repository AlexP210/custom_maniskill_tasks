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
- `object_hue`: every task object's colours turned round the hue wheel (the table, ground and
  robot are not objects);
- `object_texture`: the object of interest patterned -- wood grain, paper or print layer lines --
  in its own colours (`appearance._apply_object_texture`);
- `table_texture`: a scale, shear, rotation and shift of the table's texture;
- `viewpoint`: the wrist camera moved, turned and zoomed, by posing it in `sensor_configs`;
- (not a class) the `table_video` env kwarg: a video playing over the whole table top
  (`table_video`); visual only, on either backend;
- `table_spin`: a disc carrying the table's own texture, laid on the table top and turned `speed`
  degrees before every step (`appearance.build_table_spin_disc`); visual only, on either backend;
- `moving_light`: the key light re-aimed before every step, swinging between the default's
  direction and the side's (`lighting.direction` 0 to 1) `frequency` times per 100 steps. The one class
  applied during the episode instead of at the build, so it needs `physx_cpu`: the GPU renderer
  only reads the lights when the scene is built, and re-aiming them later changes nothing in the
  rendered frames (checked: re-aiming, recolouring, `render_system.step()` and rebuilding the
  camera group all leave physx_cuda frames bit-identical). The env refuses it on physx_cuda.

Mixed in ahead of `LightingMixin`, so it sees the kwargs first and the reconfiguration last.
`env.unwrapped.episode_perturbations` says what the current episode drew.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from custom_maniskill_tasks.cameras import PANDA_RIG, WRIST_CAMERA_UID, WristRig
from custom_maniskill_tasks.appearance import (
    _apply_object_colour,
    _apply_object_hue,
    _apply_object_texture,
    _apply_table_texture,
    build_table_spin_disc,
    table_tint,
)
from custom_maniskill_tasks.lighting import _key_direction_towards_side, lighting_config
from custom_maniskill_tasks.perturbation_classes import Perturbations, resolve_perturbations
from custom_maniskill_tasks.table_video import build_table_video, resolve_table_video
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

    def __init__(self, *args, perturbations: Mapping[str, Any] | None = None, sensor_configs=None,
                 table_video: Mapping[str, Any] | None = None, **kwargs):
        self._table_video_spec = resolve_table_video(table_video)
        self._table_video = None
        self._table_video_offset = None  # per-env start frame, with random_start
        self._perturbations: Perturbations = resolve_perturbations(
            perturbations, fov_degrees=float(np.degrees(self.WRIST_RIG.fov))
        )
        self._episode_perturbations: dict[str, dict[str, float | str]] = {}
        self._table_spin_disc = None
        chosen = self._perturbations.active
        if "moving_light" in chosen and not self._perturbations.is_neutral("lighting", "direction"):
            raise ValueError(
                "moving_light sets the key light's direction every step, so lighting.direction "
                "cannot be set as well."
            )
        if "object_texture" in chosen and getattr(type(self), "object_of_interest", None) is None:
            raise ValueError(f"{type(self).__name__} has no single object of interest, so 'object_texture' has no object.")
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

    def _moving_light_period(self) -> float | None:
        """Env steps per swing there and back, or None for a still light."""
        drawn = self._episode_perturbations.get("moving_light")
        if drawn is None or float(drawn["frequency"]) == 0.0:
            return None
        return 100.0 / float(drawn["frequency"])

    def _aim_key_light(self, amount: float) -> None:
        """Turn every sub-scene's key light (the first directional light `apply_lighting` adds)
        `amount` of the way from the default's direction to the side's."""
        import sapien

        rotation = sapien.math.shortest_rotation([1, 0, 0], list(_key_direction_towards_side(amount)))
        for sub_scene in self.scene.sub_scenes:
            lights = [e for e in sub_scene.entities if e.name == "directional_light"]
            if not lights:
                raise AssertionError("moving_light found no directional light to move")
            light = lights[0].find_component_by_type(sapien.render.RenderDirectionalLightComponent)
            light.pose = sapien.Pose(light.pose.p, rotation)

    def _moving_light_amount(self, step: int) -> float:
        """Where the light is at env step `step`: 0 (default direction) at the start of each
        period, 1 (the side) halfway through, along a cosine so it slows at the two ends."""
        return float(0.5 * (1.0 - np.cos(2.0 * np.pi * step / self._moving_light_period())))

    def _check_moving_light_backend(self) -> None:
        if self.scene.gpu_sim_enabled:
            raise ValueError(
                "moving_light needs sim_backend=physx_cpu: on physx_cuda the renderer only reads "
                "the lights when the scene is built, so the light would never move."
            )

    def reset(self, *args, **kwargs):
        # The scene is only rebuilt when something is randomized, so a fixed moving_light's light
        # is wherever the last episode left it: put it back before reset renders the first frame.
        if self._moving_light_period() is not None and getattr(self, "scene", None) is not None:
            self._aim_key_light(0.0)
        out = super().reset(*args, **kwargs)
        if self._moving_light_period() is not None:
            self._check_moving_light_backend()
        return out

    def step(self, action):
        if self._table_spin_disc is not None:
            # the frame this step returns is rendered after it, at elapsed step + 1
            speed = float(self._episode_perturbations["table_spin"]["speed"])
            self._pose_table_spin_disc(speed * (self.elapsed_steps.float() + 1))
            self._apply_mid_episode_poses()
        if self._table_video is not None:
            # the frame this step returns is rendered after it, at elapsed step + 1; loops past the end
            self._show_table_video_frames((self._table_video_offset + self.elapsed_steps + 1) % self._table_video[4])
            self._apply_mid_episode_poses()
        if self._moving_light_period() is not None:
            # the frame this step returns is rendered after it, at elapsed step + 1
            step = int(self.elapsed_steps.reshape(-1)[0]) + 1
            self._aim_key_light(self._moving_light_amount(step))
        return super().step(action)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        drawn = self._episode_perturbations
        if "object_colour" in drawn:
            _apply_object_colour(self, [drawn["object_colour"][c] for c in "rgb"])
        if "object_hue" in drawn:
            _apply_object_hue(self, drawn["object_hue"]["degrees"])
        if "object_texture" in drawn:
            _apply_object_texture(self, drawn["object_texture"]["texture"])
        if "table_texture" in drawn:
            _apply_table_texture(self, drawn["table_texture"])
        # last, so the disc takes the table's material as every other class left it
        self._table_spin_disc = None
        self._table_video = None
        if self._table_video_spec is not None:
            if "table_spin" in drawn:
                raise ValueError("table_video covers the table top, so table_spin would spin nothing visible")
            self._table_video = build_table_video(self, self._table_video_spec)
        if "table_spin" in drawn and float(drawn["table_spin"]["speed"]) != 0.0:
            self._table_spin_disc, self._table_spin_rest = build_table_spin_disc(self)

    def _pose_table_spin_disc(self, degrees, env_idx=None) -> None:
        """Turn the table-spin disc to `degrees` (a tensor, one per env in `env_idx`; all envs if None)."""
        import torch
        from mani_skill.utils.structs.pose import Pose

        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        half = torch.deg2rad(torch.as_tensor(degrees, dtype=torch.float32, device=self.device)).reshape(-1) / 2
        half = half.expand(len(env_idx)) if half.numel() == 1 else half
        q = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1)
        p = torch.tensor(self._table_spin_rest.p, dtype=torch.float32, device=self.device).expand(len(env_idx), 3)
        # on physx_cuda, Actor.set_pose writes the rows of the envs in `scene._reset_mask`: the ones
        # being reset inside `_initialize_episode`, all of them during a step
        self._table_spin_disc.set_pose(Pose.create_from_pq(p, q))

    def _show_table_video_frames(self, frame, env_idx=None) -> None:
        """Raise frame `frame[j]` (an int tensor, one per env in `env_idx`; all envs if None) of the
        table video and park every other frame of those envs inside the table top."""
        import torch
        from mani_skill.utils.structs.pose import Pose

        actor, centre, shown_z, parked_z, count = self._table_video
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        frame = torch.as_tensor(frame, device=self.device).long().reshape(-1).expand(len(env_idx))
        # rows are frame-major (all envs' quads for frame 0, then 1, ...); Actor.set_pose writes the
        # rows of the envs in `scene._reset_mask`, which during a step is every env
        shown = torch.arange(count, device=self.device)[:, None] == frame[None, :]  # (count, len(env_idx))
        z = torch.where(shown, shown_z, parked_z).reshape(-1)
        p = torch.stack([torch.full_like(z, centre[0]), torch.full_like(z, centre[1]), z], dim=-1)
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(len(z), 4)
        actor.set_pose(Pose.create_from_pq(p, q))

    def _apply_mid_episode_poses(self) -> None:
        if self.scene.gpu_sim_enabled:
            # BaseEnv.step never applies rigid-body data (only reset does), so a pose written
            # mid-episode would never reach the sim. Applying the whole buffer is what
            # ManiSkill's own `Actor.hide_visual` does: every other body's row holds the state
            # fetched after the last step, so it is written back unchanged.
            self.scene.px.gpu_apply_rigid_dynamic_data()
            self.scene.px.gpu_fetch_rigid_dynamic_data()

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if self._table_spin_disc is not None:
            self._pose_table_spin_disc(0.0, env_idx)
        if self._table_video is not None:
            import torch

            count = self._table_video[4]
            if self._table_video_offset is None or len(self._table_video_offset) != self.num_envs:
                self._table_video_offset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            if self._table_video_spec["random_start"]:
                # its own stream per env from the episode seed, so the task's own draws are untouched
                seeds = np.asarray(self._episode_seed)[env_idx.cpu().numpy()]  # env_idx: ManiSkill's long tensor
                starts = [np.random.RandomState((int(seed) ^ 0x7AB1E5) % 2**32).randint(count) for seed in seeds]
                self._table_video_offset[env_idx] = torch.as_tensor(starts, dtype=torch.long, device=self.device)
            self._show_table_video_frames(self._table_video_offset[env_idx], env_idx)

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
