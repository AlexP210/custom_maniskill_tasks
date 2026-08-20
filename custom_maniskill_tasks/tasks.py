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

Everything else is the stock task: same scene, reward, success predicate and 50-step horizon, and
the same `normalized_dense` default reward mode. The one baked-in default is
`control_mode="pd_ee_delta_pos"`, which every recording in this project used; it is a registration
default, so an explicit `gym.make(..., control_mode=...)` still wins.
"""

from __future__ import annotations

import torch

from mani_skill.envs.tasks.tabletop.lift_peg_upright import LiftPegUprightEnv
from mani_skill.envs.tasks.tabletop.place_sphere import PlaceSphereEnv
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import REGISTERED_ENVS, register_env

CONTROL_MODE = "pd_ee_delta_pos"

FULL_HORIZON_TASKS = {
    "PushCube-v1.1": "PushCube-v1",
    "PlaceSphere-v1.1": "PlaceSphere-v1",
    "LiftPegUpright-v1.1": "LiftPegUpright-v1",
}
"""Each registered variant and the stock task it derives from. The `-v1.1` suffix is not a
gymnasium version (gymnasium only parses integer versions, so these are unversioned ids whose name
happens to contain a dot) -- it reads as "our revision of -v1"."""


def _horizon(base_task: str) -> int:
    """The stock task's registered time limit, so the variant cannot drift from it."""
    return REGISTERED_ENVS[base_task].max_episode_steps


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
class PushCubeFullHorizonEnv(FullHorizonMixin, PushCubeEnv):
    """PushCube-v1 with no early termination -- see `FullHorizonMixin`."""


@register_env(
    "PlaceSphere-v1.1", max_episode_steps=_horizon("PlaceSphere-v1"), control_mode=CONTROL_MODE
)
class PlaceSphereFullHorizonEnv(FullHorizonMixin, PlaceSphereEnv):
    """PlaceSphere-v1 with no early termination -- see `FullHorizonMixin`."""


@register_env(
    "LiftPegUpright-v1.1",
    max_episode_steps=_horizon("LiftPegUpright-v1"),
    control_mode=CONTROL_MODE,
)
class LiftPegUprightFullHorizonEnv(FullHorizonMixin, LiftPegUprightEnv):
    """LiftPegUpright-v1 with no early termination -- see `FullHorizonMixin`."""
