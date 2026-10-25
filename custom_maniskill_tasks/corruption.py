"""`corruption`: degrade the camera images a task returns.

A corruption is a keyword or several joined with "+" (`"blur-2+noise-0.05"`), each a name and a
positive number:

| keyword | meaning |
| --- | --- |
| `blur-<sigma>` | Gaussian blur, `sigma` pixels wide |
| `noise-<std>` | additive Gaussian noise, `std` as a fraction of full scale (0.05 is 13 grey levels) |
| `lowres-<factor>` | the image shrunk by `factor` and blown back up, so the detail is gone but the shape is not |

Stacked, they are applied in a fixed order whatever order they are named in: `lowres`, then
`blur`, then `noise`, as a camera would degrade a frame. `"default"` (or an empty list) is none.

It is an ordinary env kwarg (`CorruptionMixin`), applied where the env produces its sensor
observations, so it reaches every camera image a policy is given and is recorded into
trajectories -- the frames stored are the corrupted ones. The human render camera (`env.render()`)
is left alone. Images keep their shape and dtype (`uint8`, `[0, 255]`); `lowres` renders at full
resolution and degrades afterwards, so the policy's input size does not change.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

DEFAULT_CORRUPTION = "default"

_KEYWORD = re.compile(r"^(?P<name>blur|noise|lowres)-(?P<value>\d+(?:\.\d+)?)$")
_ORDER = ("lowres", "blur", "noise")


@dataclass(frozen=True)
class CorruptionConfig:
    """How much of each corruption; 0 (1 for `lowres`) is none of it."""

    lowres: float = 1.0
    blur: float = 0.0
    noise: float = 0.0

    @property
    def is_default(self) -> bool:
        return self == CorruptionConfig()


DEFAULT_CORRUPTION_CONFIG = CorruptionConfig()


def canonical_corruption(corruption: str | Sequence[str] | CorruptionConfig | None) -> CorruptionConfig:
    """Resolve a `corruption` into a `CorruptionConfig`, rejecting unknown keywords early -- a
    misspelled one that fell through would build a working env feeding clean images."""
    if isinstance(corruption, CorruptionConfig):
        return corruption
    if corruption is None:
        return DEFAULT_CORRUPTION_CONFIG
    parts = corruption.split("+") if isinstance(corruption, str) else list(corruption)
    values = {"lowres": 1.0, "blur": 0.0, "noise": 0.0}
    for part in parts:
        if part in (DEFAULT_CORRUPTION, ""):
            continue
        match = _KEYWORD.match(part)
        if match is None:
            raise ValueError(
                f"Unknown corruption {part!r}, expected 'default' or a \"+\"-joined stack of "
                "'blur-<sigma>' (e.g. 'blur-2'), 'noise-<std>' (e.g. 'noise-0.05') and "
                "'lowres-<factor>' (e.g. 'lowres-4')."
            )
        name, value = match.group("name"), float(match.group("value"))
        if name == "lowres" and value < 1.0:
            raise ValueError(f"lowres-{value:g} would enlarge the image: the factor is 1 or more.")
        values[name] = value
    return CorruptionConfig(**values)


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """`image` (N, C, H, W) blurred by a separable Gaussian, edges reflected."""
    radius = max(1, math.ceil(3.0 * sigma))
    axis = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-0.5 * (axis / sigma) ** 2)
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    padded = F.pad(image, (radius, radius, radius, radius), mode="reflect")
    padded = F.conv2d(padded, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
    return F.conv2d(padded, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)


def corrupt_images(
    images: torch.Tensor, config: CorruptionConfig, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Corrupt uint8 `images` (N, H, W, 3) as `config` says; same shape and dtype back.

    `generator` supplies the noise. Everything else is deterministic.
    """
    if config.is_default:
        return images
    frame = images.permute(0, 3, 1, 2).to(torch.float32)
    height, width = frame.shape[-2:]
    if config.lowres > 1.0:
        small = (max(1, round(height / config.lowres)), max(1, round(width / config.lowres)))
        frame = F.interpolate(frame, size=small, mode="area")
        frame = F.interpolate(frame, size=(height, width), mode="bilinear", align_corners=False)
    if config.blur > 0.0:
        frame = _gaussian_blur(frame, config.blur)
    if config.noise > 0.0:
        frame = frame + config.noise * 255.0 * torch.randn(
            frame.shape, generator=generator, device=frame.device, dtype=frame.dtype
        )
    return frame.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


class CorruptionMixin:
    """Gives a task the `corruption` kwarg: its sensor images are corrupted as it says.

    The noise is drawn from a generator re-seeded on every reset from the episode rng, so the same
    reset seed gives the same noisy episode, whatever else the env was doing before.
    """

    def __init__(self, *args, corruption=DEFAULT_CORRUPTION, **kwargs):
        self._corruption_spec = corruption
        self._corruption = canonical_corruption(corruption)
        self._corruption_seed = 0
        self._corruption_generators: dict = {}
        super().__init__(*args, **kwargs)

    @property
    def corruption(self) -> CorruptionConfig:
        """The corruption this env was built with; `DEFAULT_CORRUPTION_CONFIG` unless asked."""
        return self._corruption

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if not self._corruption.is_default:
            self._corruption_seed = int(self._batched_episode_rng[env_idx][0].randint(2**31))
            self._corruption_generators = {}

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True) -> dict:
        data = super()._get_obs_sensor_data(apply_texture_transforms)
        if self._corruption.is_default:
            return data
        for camera in data.values():
            if "rgb" in camera:
                # the images may be on the sim device or the render one, so the generator follows
                device = camera["rgb"].device
                if device not in self._corruption_generators:
                    self._corruption_generators[device] = torch.Generator(device=device).manual_seed(
                        self._corruption_seed
                    )
                camera["rgb"] = corrupt_images(
                    camera["rgb"], self._corruption, self._corruption_generators[device]
                )
        return data


def supports_corruption(task_name: str) -> bool:
    from mani_skill.utils.registration import REGISTERED_ENVS

    spec = REGISTERED_ENVS.get(task_name)
    return spec is not None and issubclass(spec.cls, CorruptionMixin)
