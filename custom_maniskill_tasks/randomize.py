"""`"randomize"`: any perturbation kwarg drawn afresh on every reset, from what the evaluation runs.

    make_env("LiftPegUpright-v1.1", camera_view="wrist", lighting="randomize", corruption="randomize")

Each of `lighting`, `texture`, `table_material`, `distractors`, `viewpoint` and `corruption` takes
`"randomize"` in place of a condition. On every reset that kwarg is then one value out of
`axes.randomization_pool(kwarg)`: the runs `analysis/visual_robustness/run_sweep.py` (a point on
each graded axis's grid) and `run_ungraded.py` (a named material or pattern) evaluate, plus the
unperturbed scene, each equally likely. The kwargs are independent, so randomizing several stacks
one draw from each; a kwarg left at a fixed value stays that value.

The draw is made from the episode seed, so the same `reset(seed=...)` gives the same condition and
an unseeded reset a fresh one, and it never touches the episode rng the task itself draws from.
It is one draw per reset shared by every parallel env. `episode_perturbation` says what was drawn.

What a kwarg changes decides when it can be redrawn:

- `corruption` is applied to the images after rendering, so it is redrawn at the start of the
  episode, as any reset can.
- the others change what is built (the lights, the table, the YCB objects, the camera), so a draw
  takes effect only by reconfiguring, which this does on every reset (`reconfiguration_freq=1`).
  ManiSkill cannot reconfigure on a partial reset, so every env must reset together, which the
  `-v1.1` ids' never-terminating episodes guarantee.

Each mixin says how to take a drawn condition in its `_set_<kwarg>`. The kwarg is replaced by its
default before the mixins see it, so the first scene is built from the default and redrawn at once.
"""

from __future__ import annotations

import zlib

import numpy as np

from custom_maniskill_tasks.axes import DEFAULT_CONDITIONS, RANDOMIZE, draw_condition

SCENE_KWARGS = ("lighting", "texture", "table_material", "distractors", "viewpoint")
"""The kwargs whose condition is built into the scene or the cameras: redrawn at reconfiguration."""
IMAGE_KWARGS = ("corruption",)
"""The kwargs applied to the images, redrawn at the start of every episode."""


def is_randomize(value) -> bool:
    return isinstance(value, str) and value == RANDOMIZE


class RandomizationMixin:
    """Gives a task `"randomize"` as a value of its perturbation kwargs. Mixed in ahead of
    `LightingMixin`, so it sees the kwargs first and the reconfiguration last."""

    def __init__(self, *args, **kwargs):
        self._randomized = tuple(k for k in (*SCENE_KWARGS, *IMAGE_KWARGS) if is_randomize(kwargs.get(k)))
        self._episode_perturbation: dict[str, str | int] = {}
        for kwarg in self._randomized:
            kwargs[kwarg] = DEFAULT_CONDITIONS[kwarg]
        if any(k in SCENE_KWARGS for k in self._randomized):
            frequency = kwargs.setdefault("reconfiguration_freq", 1)
            if frequency != 1:
                raise ValueError(
                    f"{self._randomized} are redrawn on every reset, which needs "
                    f"reconfiguration_freq=1, but reconfiguration_freq={frequency} was passed."
                )
        super().__init__(*args, **kwargs)

    @property
    def episode_perturbation(self) -> dict[str, str | int]:
        """What each randomized kwarg drew for the current episode, as the kwarg value it stands
        for; empty if nothing is randomized."""
        return dict(self._episode_perturbation)

    def _accepts_randomized(self, kwarg: str, condition) -> bool:
        """Whether this env can take `condition` for `kwarg`: `object-` patterns need a task with a
        single object of interest."""
        if kwarg == "texture" and condition.startswith("object-"):
            return getattr(self, "object_of_interest", None) is not None
        return True

    def _draw_perturbation(self, kwarg: str) -> None:
        # seeded by the episode seed and the kwarg, so it is reproducible and independent of the
        # task's own draws and of the other kwargs'
        seed = int(self._episode_seed[0]) % 2**63
        rng = np.random.default_rng([seed, zlib.crc32(kwarg.encode())])
        condition = draw_condition(kwarg, rng, lambda c: self._accepts_randomized(kwarg, c))
        getattr(self, f"_set_{kwarg}")(condition)
        self._episode_perturbation[kwarg] = condition

    def _reconfigure(self, options=dict()):
        for kwarg in self._randomized:
            if kwarg in SCENE_KWARGS:
                self._draw_perturbation(kwarg)
        super()._reconfigure(options)

    def _initialize_episode(self, env_idx, options):
        # every env at once, so a partial reset cannot change the images of envs mid-episode
        if len(env_idx) == self.num_envs:
            for kwarg in self._randomized:
                if kwarg in IMAGE_KWARGS:
                    self._draw_perturbation(kwarg)
        super()._initialize_episode(env_idx, options)
