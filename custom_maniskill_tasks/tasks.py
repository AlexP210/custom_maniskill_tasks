"""The task ids this project's datasets and policies actually use.

Every offline dataset here was collected by `tools/ppo_stages_fast.py`, which builds the stock
ManiSkill task and then wraps it in `ManiSkillVectorEnv(..., ignore_terminations=True)`: an episode
is never cut short by reaching the goal, so it always runs the full task horizon, and because the
dense reward keeps accruing while the goal is held, reaching it sooner is worth strictly more
return. That is a property of the task the data describes rather than of the training loop that
happened to collect it, so it is a registered task id here -- `gym.make("PushCube-v1.1")` cannot
forget the flag, and the `env_id` in a recorded trajectory's json says which convention produced it.

Registered by importing `custom_maniskill_tasks`. Note that other processes have to import it too:
a tool that only does `import mani_skill.envs` (`tools/replay_trajectory.py`,
`tools/ppo_stages_fast.py`) will not find these ids.

These ids also take kwargs that change how the scene looks, each defaulting to the stock scene so
that it changes nothing unless asked: `lighting` (`LightingMixin`, see `lighting`) for the lighting
and colour condition, and (`PerturbationsMixin`, see `perturbations`) `texture` for patterns on the
table and objects, `distractors` for YCB objects scattered on the table, `viewpoint` for a moved
or zoomed wrist camera, and `corruption` for blurred, noisy or low-resolution images. Being
ordinary env kwargs rather than something applied around `gym.make`, they are recorded into any
trajectory collected under them and replay from that metadata on its own.

Everything else is the stock task: same scene, reward, success predicate and 50-step horizon, and
the same `normalized_dense` default reward mode. The one baked-in default is
`control_mode="pd_ee_delta_pos"`, which every recording in this project used; it is a registration
default, so an explicit `gym.make(..., control_mode=...)` still wins.
"""

from __future__ import annotations

import torch

from mani_skill.envs.tasks.tabletop.lift_peg_upright import LiftPegUprightEnv
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.pick_single_ycb import PickSingleYCBEnv
from mani_skill.envs.tasks.tabletop.place_sphere import PlaceSphereEnv
from mani_skill.envs.tasks.tabletop.poke_cube import PokeCubeEnv
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import REGISTERED_ENVS, register_env

from custom_maniskill_tasks.lighting import LightingMixin
from custom_maniskill_tasks.perturbations import PerturbationsMixin

CONTROL_MODE = "pd_ee_delta_pos"

FULL_HORIZON_TASKS = {
    "PushCube-v1.1": "PushCube-v1",
    "PlaceSphere-v1.1": "PlaceSphere-v1",
    "LiftPegUpright-v1.1": "LiftPegUpright-v1",
    "PokeCube-v1.1": "PokeCube-v1",
    "PickCube-v1.1": "PickCube-v1",
    "PickSingleYCB-v1.1": "PickSingleYCB-v1",
}
"""Each registered variant and the stock task it derives from. The `-v1.1` suffix is not a
gymnasium version (gymnasium only parses integer versions, so these are unversioned ids whose name
happens to contain a dot) -- it reads as "our revision of -v1"."""


def _horizon(base_task: str) -> int:
    """The stock task's registered time limit, so the variant cannot drift from it."""
    return REGISTERED_ENVS[base_task].max_episode_steps


def _asset_download_ids(base_task: str) -> list[str]:
    """The assets the stock task declares, so the variant prompts to fetch them too.

    `register_env` records these and `gym.make` checks them, so a variant that dropped them would
    fail deep inside the task's `__init__` (a missing json) instead of offering the download.
    """
    return REGISTERED_ENVS[base_task].asset_download_ids


class FullHorizonMixin:
    """Never terminate: every episode runs to its time limit.

    ManiSkill sets `terminated` from `info["success"]`/`info["fail"]` recomputed from the current
    state on every step (`BaseEnv.step`), so it is a momentary predicate rather than an absorbing
    state -- it flips back to False when the goal stops being satisfied. Suppressing it here, in
    the task, means nothing downstream has to remember to: `ManiSkillVectorEnv` has no termination
    left to auto-reset on, and the `IgnoreTerminations` wrapper becomes a no-op (`make_env` still
    applies it, harmlessly, since it also serves the stock `-v1` ids).

    The signal itself is untouched and still reported as `info["success"]`.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, torch.zeros_like(terminated), truncated, info


@register_env(
    "PushCube-v1.1", max_episode_steps=_horizon("PushCube-v1"), control_mode=CONTROL_MODE
)
class PushCubeFullHorizonEnv(FullHorizonMixin, LightingMixin, PerturbationsMixin, PushCubeEnv):
    """PushCube-v1 with no early termination and the appearance kwargs above."""

    object_of_interest = "obj"
    distractor_avoid = (
        ("obj", (0.0, 0.0), (0.03, 0.03)),
        ("obj", (0.12, 0.0), (0.12, 0.05)),  # the cube's way to its goal
    )


@register_env(
    "PlaceSphere-v1.1", max_episode_steps=_horizon("PlaceSphere-v1"), control_mode=CONTROL_MODE
)
class PlaceSphereFullHorizonEnv(
    FullHorizonMixin, LightingMixin, PerturbationsMixin, PlaceSphereEnv
):
    """PlaceSphere-v1 with no early termination and the appearance kwargs above."""

    object_of_interest = "obj"
    distractor_avoid = (("obj", (0.0, 0.0), (0.03, 0.03)), ("bin", (0.0, 0.0), (0.06, 0.06)))


@register_env(
    "LiftPegUpright-v1.1",
    max_episode_steps=_horizon("LiftPegUpright-v1"),
    control_mode=CONTROL_MODE,
)
class LiftPegUprightFullHorizonEnv(
    FullHorizonMixin, LightingMixin, PerturbationsMixin, LiftPegUprightEnv
):
    """LiftPegUpright-v1 with no early termination and the appearance kwargs above."""

    object_of_interest = "peg"
    distractor_avoid = (("peg", (0.0, 0.0), (0.12, 0.03)),)


@register_env(
    "PokeCube-v1.1", max_episode_steps=_horizon("PokeCube-v1"), control_mode=CONTROL_MODE
)
class PokeCubeFullHorizonEnv(FullHorizonMixin, LightingMixin, PerturbationsMixin, PokeCubeEnv):
    """PokeCube-v1 with no early termination and the appearance kwargs above.

    Has no single object of interest (a cube and the peg that pokes it), so `texture` has no
    `object` here: name `cube` or `peg`."""

    distractor_avoid = (("cube", (0.0, 0.0), (0.03, 0.03)), ("peg", (0.0, 0.0), (0.12, 0.03)))


@register_env(
    "PickCube-v1.1", max_episode_steps=_horizon("PickCube-v1"), control_mode=CONTROL_MODE
)
class PickCubeFullHorizonEnv(FullHorizonMixin, LightingMixin, PerturbationsMixin, PickCubeEnv):
    """PickCube-v1 with no early termination and the appearance kwargs above."""

    object_of_interest = "cube"
    distractor_avoid = (("cube", (0.0, 0.0), (0.03, 0.03)),)


@register_env(
    "PickSingleYCB-v1.1",
    max_episode_steps=_horizon("PickSingleYCB-v1"),
    asset_download_ids=_asset_download_ids("PickSingleYCB-v1"),
    control_mode=CONTROL_MODE,
)
class PickSingleYCBFullHorizonEnv(
    FullHorizonMixin, LightingMixin, PerturbationsMixin, PickSingleYCBEnv
):
    """PickSingleYCB-v1 with no early termination and the appearance kwargs above.

    The only one of these variants whose scene is not fixed: which YCB object is in it is drawn
    per parallel env at reconfiguration, so the stock task defaults `reconfiguration_freq` to 1 at
    `num_envs=1` (a new object every reset) and to 0 above it (one draw, held for the run). That
    default is the task's own and is untouched here.
    """

    object_of_interest = "obj"
    distractor_avoid = (("obj", (0.0, 0.0), (0.07, 0.07)),)
