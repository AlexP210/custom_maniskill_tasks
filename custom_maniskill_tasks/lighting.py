"""The lighting conditions these tasks can be rendered under.

One `lighting` preset selects the lights for the whole scene, the same way one `camera_view`
selects the observation camera, and everything about a condition lives here: the ambient colour,
the directional lights, and the ranges a randomizing condition draws from.

The `"default"` preset is not a choice -- it is `BaseEnv._load_lighting` transcribed, which is
what every dataset in this project was rendered under. Treat those numbers as data. The other
presets are deliberate *shifts* away from it, meant to be named in an experiment ("evaluated under
`dim`") rather than tuned per run, so a reported number says which condition produced it.

Applied through `LightingMixin` on the registered `-v1.1` task classes (see `tasks`) rather than
around the `gym.make` call the way the camera views are. That makes `lighting` an ordinary env
kwarg: `gym.make` records it in `env.spec.kwargs`, `RecordEpisode` writes it into the trajectory
json, and a replay of that dataset rebuilds the same condition without being told. A shifted
recording is therefore distinguishable from a default-lit one on disk, which a scene patched from
the outside would not be.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from mani_skill import logger
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import REGISTERED_ENVS

Color = tuple[float, float, float]


@dataclass(frozen=True)
class DirectionalLight:
    """One directional light, in the terms `ManiSkillScene.add_directional_light` takes.

    `direction` is the direction the light travels in and is not normalized (sapien normalizes it
    when it builds the pose), so it reads the same way as ManiSkill's own `[1, 1, -1]`: a light
    from above, behind and to one side. `shadow=None` means "follow the env's `enable_shadow`",
    which is how the base task's key light behaves; True or False pins it regardless.
    """

    direction: Color
    color: Color
    shadow: bool | None = None
    shadow_scale: float = 10.0
    shadow_map_size: int = 2048


@dataclass(frozen=True)
class LightingRandomization:
    """Per-env draws applied on top of a `LightingConfig`, as multipliers on it.

    Multipliers rather than absolute ranges so a randomizing condition can sit on top of any base
    preset and stay recognisably that preset. Every range is `(low, high)` and drawn uniformly.

    `tint` is drawn once per channel, so it produces colour casts rather than pure brightness
    changes; `direction_jitter` is added to each component of a light's direction, which moves
    where the shading and any shadows fall.
    """

    ambient: tuple[float, float] = (1.0, 1.0)
    brightness: tuple[float, float] = (1.0, 1.0)
    tint: tuple[float, float] = (1.0, 1.0)
    direction_jitter: float = 0.0


@dataclass(frozen=True)
class LightingConfig:
    """A complete lighting condition: what `_load_lighting` will build.

    `randomization=None` is a fixed condition -- every parallel env is lit identically and the
    condition is fully described by the preset name. With a `LightingRandomization` each parallel
    env draws its own variation at reconfiguration; see `apply_lighting` for what that means and
    does not mean.
    """

    ambient: Color
    lights: tuple[DirectionalLight, ...]
    randomization: LightingRandomization | None = None


DEFAULT_AMBIENT: Color = (0.3, 0.3, 0.3)

DEFAULT_LIGHTING = LightingConfig(
    ambient=DEFAULT_AMBIENT,
    lights=(
        # the key light, whose shadow the `enable_shadow` env kwarg switches on
        DirectionalLight(
            direction=(1.0, 1.0, -1.0), color=(1.0, 1.0, 1.0), shadow_scale=5.0
        ),
        # the fill from straight above, which the base task never shadows
        DirectionalLight(direction=(0.0, 0.0, -1.0), color=(1.0, 1.0, 1.0), shadow=False),
    ),
)
"""`BaseEnv._load_lighting` as data, down to the shadow parameters it passes. Building this
config has to be indistinguishable from not overriding `_load_lighting` at all -- every dataset in
this project was recorded under it, so a policy evaluated at `lighting="default"` has to be
looking at the same scene its training frames came from."""


def _tinted(config: LightingConfig, tint: Color) -> LightingConfig:
    """`config` with `tint` multiplied into every colour, the ambient included.

    Multiplying the ambient too is what makes an exposure preset read as one: leaving it at 0.3
    while halving the lights would darken the lit faces and leave the shadowed ones alone, which
    is a change of contrast rather than of light level.
    """
    return replace(
        config,
        ambient=tuple(channel * scale for channel, scale in zip(config.ambient, tint)),
        lights=tuple(
            replace(light, color=tuple(c * s for c, s in zip(light.color, tint)))
            for light in config.lights
        ),
    )


LIGHTING_PRESETS: dict[str, LightingConfig] = {
    "default": DEFAULT_LIGHTING,
    # exposure shifts: the scene is lit the same way, less or more of it
    "dim": _tinted(DEFAULT_LIGHTING, (0.5, 0.5, 0.5)),
    "bright": _tinted(DEFAULT_LIGHTING, (1.5, 1.5, 1.5)),
    # colour-temperature shifts: same geometry and roughly the same exposure, different cast
    "warm": _tinted(DEFAULT_LIGHTING, (1.15, 0.9, 0.65)),
    "cool": _tinted(DEFAULT_LIGHTING, (0.7, 0.85, 1.2)),
    # a geometric shift: the key light crosses to the other side and rakes lower, so shading
    # (and, under `enable_shadow`, shadows) falls the opposite way at the same exposure
    "side": replace(
        DEFAULT_LIGHTING,
        lights=(
            replace(DEFAULT_LIGHTING.lights[0], direction=(-1.0, -1.0, -0.35)),
            DEFAULT_LIGHTING.lights[1],
        ),
    ),
    # training-time domain randomization: each parallel env draws its own condition, held for
    # the life of the env unless it reconfigures -- see `apply_lighting`
    "random": replace(
        DEFAULT_LIGHTING,
        randomization=LightingRandomization(
            ambient=(0.5, 1.6),
            brightness=(0.5, 1.5),
            tint=(0.75, 1.0),
            direction_jitter=0.5,
        ),
    ),
}
"""The named conditions, and the whole interface: a preset name is what an experiment reports and
what a recording carries in its metadata. A dict of the same shape as `LightingConfig` is accepted
wherever a name is, for a one-off condition that has not earned a name yet."""

DEFAULT_LIGHTING_PRESET = "default"


def _color(value, where: str) -> Color:
    """Three numbers, or a loud error naming the field that was wrong."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"lighting {where} must be a sequence of 3 numbers, got {value!r}")
    if len(value) != 3:
        raise ValueError(f"lighting {where} must have 3 components, got {len(value)}: {value!r}")
    return tuple(float(channel) for channel in value)


def _range(value, where: str) -> tuple[float, float]:
    """A `(low, high)` pair, low first."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"lighting {where} must be a (low, high) pair, got {value!r}")
    low, high = (float(bound) for bound in value)
    if low > high:
        raise ValueError(f"lighting {where} has low > high: {value!r}")
    return low, high


def _fields(entry, required: set[str], optional: set[str], where: str) -> dict:
    """`entry` as a plain dict, with a missing or unknown key raising rather than defaulting.

    A silently ignored key is the failure mode this guards: `{"colour": ...}` for `"color"` would
    otherwise build a perfectly working env lit exactly the way the caller was trying to change.
    """
    if not isinstance(entry, Mapping):
        raise TypeError(f"lighting {where} must be a mapping, got {entry!r}")
    keys = set(entry)
    missing = required - keys
    unknown = keys - required - optional
    # both at once, because a misspelled key produces both halves and each alone is misleading
    problems = []
    if missing:
        problems.append(f"is missing {sorted(missing)}")
    if unknown:
        problems.append(f"has unknown keys {sorted(unknown)}")
    if problems:
        raise ValueError(
            f"lighting {where} {' and '.join(problems)}; "
            f"expected {sorted(required | optional)}"
        )
    return dict(entry)


def _light_from_dict(entry, index: int) -> DirectionalLight:
    where = f"lights[{index}]"
    fields = _fields(
        entry,
        required={"direction", "color"},
        optional={"shadow", "shadow_scale", "shadow_map_size"},
        where=where,
    )
    light = DirectionalLight(
        direction=_color(fields["direction"], f"{where}.direction"),
        color=_color(fields["color"], f"{where}.color"),
    )
    shadow = fields.get("shadow", light.shadow)
    return replace(
        light,
        shadow=None if shadow is None else bool(shadow),
        shadow_scale=float(fields.get("shadow_scale", light.shadow_scale)),
        shadow_map_size=int(fields.get("shadow_map_size", light.shadow_map_size)),
    )


def _randomization_from_dict(entry) -> LightingRandomization:
    fields = _fields(
        entry,
        required=set(),
        optional={"ambient", "brightness", "tint", "direction_jitter"},
        where="randomization",
    )
    base = LightingRandomization()
    return LightingRandomization(
        ambient=_range(fields.get("ambient", base.ambient), "randomization.ambient"),
        brightness=_range(fields.get("brightness", base.brightness), "randomization.brightness"),
        tint=_range(fields.get("tint", base.tint), "randomization.tint"),
        direction_jitter=float(fields.get("direction_jitter", base.direction_jitter)),
    )


def lighting_config_from_dict(spec: Mapping) -> LightingConfig:
    """A `LightingConfig` from the json-shaped dict form, for a condition with no preset name."""
    fields = _fields(
        spec, required={"ambient", "lights"}, optional={"randomization"}, where="config"
    )
    lights = fields["lights"]
    if isinstance(lights, (str, bytes)) or not isinstance(lights, Sequence) or not lights:
        raise ValueError(f"lighting config.lights must be a non-empty sequence, got {lights!r}")
    randomization = fields.get("randomization")
    return LightingConfig(
        ambient=_color(fields["ambient"], "config.ambient"),
        lights=tuple(_light_from_dict(entry, i) for i, entry in enumerate(lights)),
        randomization=(
            None if randomization is None else _randomization_from_dict(randomization)
        ),
    )


def canonical_lighting(lighting: str | Mapping | LightingConfig) -> LightingConfig:
    """Resolve a preset name or a dict into a `LightingConfig`, rejecting unknown ones early.

    Worth doing before `gym.make` for the same reason `canonical_camera_view` is: a misspelled
    preset that fell through to "no override" would build a perfectly working env showing the
    unshifted scene, and nothing downstream would ever say so.
    """
    if isinstance(lighting, LightingConfig):
        return lighting
    if isinstance(lighting, str):
        try:
            return LIGHTING_PRESETS[lighting]
        except KeyError:
            raise ValueError(
                f"Unknown lighting preset {lighting!r}, expected one of "
                f"{', '.join(repr(name) for name in LIGHTING_PRESETS)}"
            ) from None
    if isinstance(lighting, Mapping):
        return lighting_config_from_dict(lighting)
    raise TypeError(
        f"lighting must be a preset name, a config dict or a LightingConfig, got {lighting!r}"
    )


def _drawn_configs(config: LightingConfig, episode_rng, num_scenes: int) -> list[LightingConfig]:
    """One config per parallel env, drawn from `config.randomization`.

    Every draw is one batched call on `_batched_episode_rng`, which advances each env's own
    numpy RandomState in lockstep, so a given episode seed produces the same lighting under CPU
    and GPU simulation. Drawing per scene in a python loop would not: each scene would consume a
    different number of values from its stream.
    """
    random = config.randomization
    if episode_rng.batch_size != num_scenes:
        raise AssertionError(
            f"episode rng is batched over {episode_rng.batch_size} envs but the scene has "
            f"{num_scenes} sub-scenes"
        )

    ambient_scale = episode_rng.uniform(*random.ambient)
    ambient_tint = episode_rng.uniform(*random.tint, size=(3,))
    per_light = [
        (
            episode_rng.uniform(*random.brightness),
            episode_rng.uniform(*random.tint, size=(3,)),
            episode_rng.uniform(
                -random.direction_jitter, random.direction_jitter, size=(3,)
            ),
        )
        for _ in config.lights
    ]

    drawn = []
    for i in range(num_scenes):
        lights = []
        for light, (brightness, tint, jitter) in zip(config.lights, per_light):
            lights.append(
                replace(
                    light,
                    direction=tuple(np.asarray(light.direction) + jitter[i]),
                    color=tuple(np.asarray(light.color) * brightness[i] * tint[i]),
                )
            )
        drawn.append(
            replace(
                config,
                ambient=tuple(np.asarray(config.ambient) * ambient_scale[i] * ambient_tint[i]),
                lights=tuple(lights),
                randomization=None,
            )
        )
    return drawn


def _add_light(scene, light: DirectionalLight, enable_shadow: bool, scene_idxs=None) -> None:
    scene.add_directional_light(
        [float(component) for component in light.direction],
        [float(channel) for channel in light.color],
        shadow=enable_shadow if light.shadow is None else light.shadow,
        shadow_scale=light.shadow_scale,
        shadow_map_size=light.shadow_map_size,
        scene_idxs=scene_idxs,
    )


def apply_lighting(scene, config: LightingConfig, enable_shadow: bool, episode_rng=None) -> None:
    """Build `config`'s lights into `scene`. The body of a `_load_lighting` override.

    A fixed config lights every parallel env identically. A randomizing one draws per sub-scene,
    which is per parallel env and *not* per episode: `_load_lighting` runs only inside
    `_reconfigure`, so an env keeps the condition it drew until it reconfigures. Under the default
    `reconfiguration_freq=0` that is once, at construction, off ManiSkill's fixed `2022 + i` seeds
    -- so a randomizing preset gives the same `num_envs` conditions on every run, and a reset seed
    does not move them. `reconfiguration_freq=1` redraws every reset, at the cost of rebuilding
    the scene.

    That makes randomization the right tool for training-time domain randomization (`num_envs`
    conditions at once) and the wrong one for measuring performance under a distribution of
    conditions -- for that, build the env once per named preset.
    """
    if config.randomization is None:
        scene.set_ambient_light([float(channel) for channel in config.ambient])
        for light in config.lights:
            _add_light(scene, light, enable_shadow)
        return

    if scene.parallel_in_single_scene:
        raise ValueError(
            "A randomizing lighting config needs one lighting condition per parallel env, but "
            "`parallel_in_single_scene=True` puts every env in one sapien scene, where "
            "`add_directional_light` adds a single light for all of them. Use a fixed preset "
            "here, or build the env with parallel_in_single_scene=False."
        )
    if episode_rng is None:
        raise ValueError("A randomizing lighting config needs the env's batched episode rng.")

    for scene_idx, drawn in enumerate(
        _drawn_configs(config, episode_rng, len(scene.sub_scenes))
    ):
        # per-sub-scene, because `set_ambient_light` writes the same colour to all of them
        scene.sub_scenes[scene_idx].render_system.ambient_light = [
            float(channel) for channel in drawn.ambient
        ]
        for light in drawn.lights:
            _add_light(scene, light, enable_shadow, scene_idxs=[scene_idx])


class LightingMixin:
    """Gives a task a `lighting` kwarg naming the condition it renders under.

    Mixed in ahead of the task class, so its `_load_lighting` wins. That is what makes the shift
    reach a task at all, and also means a task that lights its own scene would be overridden
    rather than shifted -- `__init__` says so loudly if it finds one, since the result would look
    like a lighting shift while actually being a different scene.

    The config is resolved and stored before `super().__init__`, because `BaseEnv.__init__`
    reconfigures (and so calls `_load_lighting`) before it returns.
    """

    def __init__(
        self,
        *args,
        lighting: str | Mapping | LightingConfig = DEFAULT_LIGHTING_PRESET,
        **kwargs,
    ):
        self._lighting = canonical_lighting(lighting)
        self._warn_if_task_lights_itself()
        super().__init__(*args, **kwargs)

    @property
    def lighting(self) -> LightingConfig:
        """The condition this env was built with; `DEFAULT_LIGHTING` unless asked otherwise."""
        return self._lighting

    def _warn_if_task_lights_itself(self) -> None:
        mro = type(self).__mro__
        after_mixin = mro[mro.index(LightingMixin) + 1 :]
        for klass in after_mixin:
            if "_load_lighting" not in vars(klass):
                continue
            if klass is BaseEnv:
                return  # the default, which is exactly what DEFAULT_LIGHTING reproduces
            logger.warning(
                f"{klass.__name__} defines its own `_load_lighting`, which LightingMixin "
                "overrides: this env is lit by its `lighting` config alone, not by the task's "
                "own lights."
            )
            return

    def _load_lighting(self, options: dict):
        apply_lighting(
            self.scene,
            self._lighting,
            enable_shadow=self.enable_shadow,
            episode_rng=self._batched_episode_rng,
        )


def supports_lighting(task_name: str) -> bool:
    """Whether `task_name`'s registered class takes a `lighting` kwarg."""
    spec = REGISTERED_ENVS.get(task_name)
    return spec is not None and issubclass(spec.cls, LightingMixin)


def check_lighting(env, config: LightingConfig) -> None:
    """Assert the condition `config` asked for is the one actually in the built scene.

    Two silent failures to catch. A scene that cannot render never has `_load_lighting` called at
    all (`BaseEnv._reconfigure`), so a shift asked for on a state-only env would simply not
    happen. And a light count that disagrees with the config means something else lit the scene.

    The default condition is not checked: it is the absence of a shift, so there is nothing it
    could silently fail to do, and every env in this project that predates `lighting` builds it.
    """
    if config == DEFAULT_LIGHTING:
        return

    scene = env.unwrapped.scene
    task = env.unwrapped.spec.id if env.unwrapped.spec else type(env.unwrapped).__name__
    if not scene.can_render():
        raise ValueError(
            f"A non-default lighting config was requested for {task}, but its scene cannot "
            "render, so ManiSkill never loaded any lighting. These observations carry no shift."
        )

    built = sum(entity.name == "directional_light" for entity in scene.sub_scenes[0].entities)
    if built != len(config.lights):
        raise ValueError(
            f"Lighting config for {task} declares {len(config.lights)} directional lights but "
            f"the built scene has {built}. Something other than this config lit the scene."
        )

    if config.randomization is None:
        ambient = np.asarray(scene.sub_scenes[0].render_system.ambient_light)[:3]
        if not np.allclose(ambient, config.ambient, atol=1e-5):
            raise ValueError(
                f"Lighting config for {task} asks for ambient {tuple(config.ambient)} but the "
                f"built scene has {tuple(ambient)}."
            )
