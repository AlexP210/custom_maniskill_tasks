"""The lights these tasks are rendered under, and how the `lighting` perturbation class sets them.

`lighting` has three aspects (see `perturbation_classes`): `brightness`, a multiple of the stock
scene's light levels; `direction`, how far the key light is turned from the stock direction towards
the opposite side of the table, 0 to 1; and `temperature`, the colour temperature of the lights,
-1 (2700 K) through 0 (6500 K, the stock white) to 1 (12000 K). `lighting_config` builds the
`LightingConfig` those three stand for, and `LightingMixin` builds its lights into the scene.

`DEFAULT_LIGHTING` is not a choice -- it is `BaseEnv._load_lighting` transcribed, which is what
every dataset in this project was rendered under. Treat those numbers as data.

The table's colour is not light at all, strictly, but an appearance shift of the same kind -- what
the wrist camera sees of the tabletop -- so a `LightingConfig` carries a `table_tint` too, which
`LightingMixin` multiplies into the table (the `table_colour` perturbation class sets it).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import sapien

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
class LightingConfig:
    """A complete lighting condition: what `_load_lighting` (and `_load_scene`) will build."""

    ambient: Color
    lights: tuple[DirectionalLight, ...]
    table_tint: Color = (1.0, 1.0, 1.0)
    """Multiplied into the table's material colours (and so its wood texture); see `_tint_table`."""


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
this project was recorded under it, so a policy evaluated under the default lighting has to be
looking at the same scene its training frames came from."""

SIDE_KEY_DIRECTION: Color = (-1.0, -1.0, -0.35)
"""Where `direction=1` puts the key light: crossed to the other side of the table and raking lower."""


def _key_direction_towards_side(amount: float) -> Color:
    """The key light direction `amount` of the way from the default's to `SIDE_KEY_DIRECTION`.

    A slerp between the two, not a lerp: both lie in the vertical plane x == y, and the short arc
    between them passes straight overhead, so equal steps in `amount` turn the light through equal
    angles (about 13 degrees per 0.1). A lerp would bunch the steps up near the overhead point.
    The two endpoints come back as the vectors themselves rather than their normalized forms.
    """
    start = DEFAULT_LIGHTING.lights[0].direction
    if amount == 0.0:
        return start
    if amount == 1.0:
        return SIDE_KEY_DIRECTION
    a = np.asarray(start) / np.linalg.norm(start)
    b = np.asarray(SIDE_KEY_DIRECTION) / np.linalg.norm(SIDE_KEY_DIRECTION)
    angle = np.arccos(np.clip(a @ b, -1.0, 1.0))
    direction = (np.sin((1.0 - amount) * angle) * a + np.sin(amount * angle) * b) / np.sin(angle)
    return tuple(float(component) for component in direction)


DEFAULT_KELVIN = 6500.0
"""The colour temperature the stock white lights are taken to be; `temperature`'s zero."""
TEMPERATURE_KELVIN = {"warm": 2700.0, "cool": 12000.0}
"""Where `temperature=-1` and `temperature=1` end: a household incandescent bulb, and open shade
under a clear blue sky. Both about as far as real light goes before it stops reading as white."""

# linear sRGB from CIE XYZ, D65 white
_XYZ_TO_LINEAR_SRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])
_LUMINANCE = np.array([0.2126, 0.7152, 0.0722])


def _planckian_linear_rgb(kelvin: float) -> np.ndarray:
    """A blackbody's colour at `kelvin`, in linear sRGB at luminance 1.

    From the cubic-spline fit to the Planckian locus of Kim et al. (2002), valid 1667-25000 K.
    Linear rather than gamma-encoded because light colours multiply radiance in the renderer.
    """
    t = kelvin
    if t <= 4000.0:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return _XYZ_TO_LINEAR_SRGB @ np.array([x / y, 1.0, (1.0 - x - y) / y])


def _temperature_kelvin(hue: str, amount: float) -> float:
    """The colour temperature `amount` of the way from `DEFAULT_KELVIN` to
    `TEMPERATURE_KELVIN[hue]`, stepped in mired (1e6 / kelvin), the scale colour-temperature
    differences are perceived evenly on."""
    start, end = 1e6 / DEFAULT_KELVIN, 1e6 / TEMPERATURE_KELVIN[hue]
    return 1e6 / (start + amount * (end - start))


def _temperature_tint(temperature: float) -> Color:
    """The tint of the lights at `temperature` (-1 to 1), relative to the default's blackbody so
    that 0 is exactly no tint, and rescaled to luminance 1, so it moves the hue and not the
    exposure: that is `brightness`."""
    if temperature == 0.0:
        return (1.0, 1.0, 1.0)
    hue = "warm" if temperature < 0 else "cool"
    kelvin = _temperature_kelvin(hue, abs(temperature))
    tint = _planckian_linear_rgb(kelvin) / _planckian_linear_rgb(DEFAULT_KELVIN)
    return tuple(float(channel) for channel in tint / (_LUMINANCE @ tint))


def lighting_config(
    brightness: float = 1.0, direction: float = 0.0, temperature: float = 0.0, table_tint: Color = (1.0, 1.0, 1.0)
) -> LightingConfig:
    """The condition the `lighting` aspects stand for: the ambient at `0.3 * brightness` and both
    lights at `brightness` (grey), the key light `direction` of the way to the side, and the
    lights' colour tinted to `temperature`. At the neutral values this is `DEFAULT_LIGHTING`."""
    tint = _temperature_tint(temperature)
    levels = (brightness,) * 3
    key = replace(DEFAULT_LIGHTING.lights[0], direction=_key_direction_towards_side(direction))
    lights = (key, *DEFAULT_LIGHTING.lights[1:])
    return LightingConfig(
        ambient=tuple(0.3 * level * t for level, t in zip(levels, tint)),
        lights=tuple(
            replace(light, color=tuple(level * t for level, t in zip(levels, tint))) for light in lights
        ),
        table_tint=table_tint,
    )


def _add_light(scene, light: DirectionalLight, enable_shadow: bool) -> None:
    scene.add_directional_light(
        [float(component) for component in light.direction],
        [float(channel) for channel in light.color],
        shadow=enable_shadow if light.shadow is None else light.shadow,
        shadow_scale=light.shadow_scale,
        shadow_map_size=light.shadow_map_size,
    )


def apply_lighting(scene, config: LightingConfig, enable_shadow: bool) -> None:
    """Build `config`'s lights into `scene`, every parallel env lit identically. The body of a
    `_load_lighting` override."""
    scene.set_ambient_light([float(channel) for channel in config.ambient])
    for light in config.lights:
        _add_light(scene, light, enable_shadow)


def _srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def _remapped_texture(texture, remap: Callable[[np.ndarray], np.ndarray]):
    """A new texture holding `texture`'s pixels with `remap` applied to their linear colours.

    New rather than `texture.upload`-ed in place: sapien caches a model's textures for the whole
    process, so rewriting one would recolour that model in every env built after this one too,
    the default scene included. `remap` takes and returns an (..., 3) array of linear RGB -- an
    sRGB texture is decoded before and re-encoded after, so a 0.4 tint reflects 40% of the light
    the way a 0.4 light colour would -- and alpha is untouched.
    """
    pixels = texture.download()
    if pixels.dtype != np.uint8:
        raise TypeError(f"expected a uint8 texture, got {pixels.dtype}")
    colour = pixels[..., :3].astype(np.float64) / 255.0
    if texture.is_srgb:
        colour = _linear_to_srgb(np.clip(remap(_srgb_to_linear(colour)), 0.0, 1.0))
    else:
        colour = remap(colour)
    tinted = pixels.copy()
    tinted[..., :3] = np.clip(np.round(colour * 255.0), 0, 255).astype(np.uint8)
    return sapien.render.RenderTexture2D(
        tinted,
        texture.format,
        texture.mipmap_levels,
        texture.filter_mode,
        texture.address_mode,
        texture.is_srgb,
    )


def _tint_table(env, tint: Color) -> None:
    """Multiply `tint` into the colour of `env`'s table: the wood texture of its top, and the
    `base_color` of its untextured parts.

    Only takes effect before the scene first renders, so it runs from `_load_scene`: a material
    edited after that is stored but never reaches the renderer. And the texture has to be replaced
    rather than scaled through `base_color`, which sapien ignores on a textured material.

    Every parallel env's table is the same model, so the tinted materials are worked out once, from
    the first env's, and assigned to all of them. That is also what keeps the tint from
    compounding: envs can share one material, and tinting each env's in turn would tint a shared
    one again. A task with no `TableSceneBuilder` table fails loudly rather than rendering the
    untinted scene under a condition that says otherwise.
    """
    table_scene = getattr(env, "table_scene", None)
    if table_scene is None:
        raise ValueError(
            f"{type(env).__name__} has no `table_scene`, so a lighting condition with a "
            f"table_tint of {tint} has no table to tint."
        )

    def parts(entity):
        body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        return [part for shape in body.render_shapes for part in shape.parts]

    entities = table_scene.table._objs
    template = parts(entities[0])
    tinted = []
    for part in template:
        texture = part.material.base_color_texture
        r, g, b, a = part.material.base_color
        tinted.append((
            None if texture is None else _remapped_texture(texture, lambda c: c * np.asarray(tint)),
            [r * tint[0], g * tint[1], b * tint[2], a],
        ))
    for entity in entities:
        entity_parts = parts(entity)
        if len(entity_parts) != len(template):
            raise AssertionError(
                f"table entity {entity.name} has {len(entity_parts)} render parts, the first has "
                f"{len(template)}: the parallel envs' tables are not the same model"
            )
        for part, (texture, base_color) in zip(entity_parts, tinted):
            if texture is None:
                part.material.base_color = base_color
            else:
                part.material.base_color_texture = texture


class LightingMixin:
    """Builds a task's lights from a `LightingConfig`, which `PerturbationClassesMixin` sets
    (`_set_lighting`) before each reconfiguration. Without one the task is lit as it always was.

    Mixed in ahead of the task class, so its `_load_lighting` wins. That is what makes the shift
    reach a task at all, and also means a task that lights its own scene would be overridden
    rather than shifted -- `__init__` says so loudly if it finds one, since the result would look
    like a lighting shift while actually being a different scene. `_load_scene` is mixed in the
    same way but *adds* to the task's own scene rather than replacing it: a config with a
    `table_tint` has the table recoloured.
    """

    def __init__(self, *args, **kwargs):
        self._lighting = DEFAULT_LIGHTING
        self._warn_if_task_lights_itself()
        super().__init__(*args, **kwargs)

    @property
    def lighting(self) -> LightingConfig:
        """The condition the scene is currently built with; `DEFAULT_LIGHTING` unless perturbed."""
        return self._lighting

    def _set_lighting(self, config: LightingConfig) -> None:
        """Take `config` as this env's lighting from the next reconfiguration."""
        self._lighting = config

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
                "overrides: this env is lit by its lighting config alone, not by the task's "
                "own lights."
            )
            return

    def _load_lighting(self, options: dict):
        apply_lighting(self.scene, self._lighting, enable_shadow=self.enable_shadow)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        if self._lighting.table_tint != (1.0, 1.0, 1.0):
            _tint_table(self, self._lighting.table_tint)


def supports_lighting(task_name: str) -> bool:
    """Whether `task_name`'s registered class takes a `lighting` config."""
    spec = REGISTERED_ENVS.get(task_name)
    return spec is not None and issubclass(spec.cls, LightingMixin)


def check_lighting(env, config: LightingConfig) -> None:
    """Assert the condition `config` asked for is the one actually in the built scene.

    Two silent failures to catch. A scene that cannot render never has `_load_lighting` called at
    all (`BaseEnv._reconfigure`), so a shift asked for on a state-only env would simply not
    happen. And a light count that disagrees with the config means something else lit the scene.

    The default condition is not checked: it is the absence of a shift, so there is nothing it
    could silently fail to do.
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

    ambient = np.asarray(scene.sub_scenes[0].render_system.ambient_light)[:3]
    if not np.allclose(ambient, config.ambient, atol=1e-5):
        raise ValueError(
            f"Lighting config for {task} asks for ambient {tuple(config.ambient)} but the "
            f"built scene has {tuple(ambient)}."
        )
