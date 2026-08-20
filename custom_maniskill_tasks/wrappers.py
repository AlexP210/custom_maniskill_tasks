"""Wrappers shared by everything that runs these tasks: online RL, planning, dataset replay.

`FrameSkip` and `FrameStack` are written to compose in that order (skip inside, stack outside),
which is what makes a stacked observation span `n_frames` *macro* steps rather than `n_frames`
primitive sim steps -- see `FrameStack`.

Both are agnostic about the observation layout so that they can sit on either side of a
project-specific observation adapter: ManiSkill's nested dicts of batched torch tensors, a
`TensorDict`, or a single tensor/array all work.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import torch

try:  # tensordict is a TSD dependency, not a ManiSkill one
    from tensordict import TensorDict, TensorDictBase
except ImportError:  # pragma: no cover
    TensorDict = TensorDictBase = None


# --------------------------------------------------------------------------------------------- #
# observation-tree helpers
# --------------------------------------------------------------------------------------------- #


def _is_mapping(x) -> bool:
    return isinstance(x, dict) or (
        TensorDictBase is not None and isinstance(x, TensorDictBase)
    )


def _rebuild_mapping(template, items: dict):
    if TensorDictBase is not None and isinstance(template, TensorDictBase):
        return TensorDict(items)
    return dict(items)


def clone_observations(obs):
    """A private copy of `obs`, safe to hold on to across steps.

    Necessary, not defensive: ManiSkill hands back its camera capture buffer itself rather than a
    copy, so a `sensor_data/*/rgb` tensor (and `info["elapsed_steps"]`) is *overwritten in place*
    on the next step -- verified by `data_ptr` staying constant while the contents change. A frame
    buffer that stored the tensor instead of a copy would therefore hold N references to the
    current frame and stack it with itself. Rewards, termination flags, state observations and
    other extras are freshly allocated each step and need no copy.
    """
    if _is_mapping(obs):
        return _rebuild_mapping(
            obs, {key: clone_observations(value) for key, value in obs.items()}
        )
    if isinstance(obs, torch.Tensor):
        return obs.clone()
    if isinstance(obs, np.ndarray):
        return obs.copy()
    return obs


def stack_observations(frames, axis: int):
    """Stack a sequence of like-structured observations along a new axis at `axis`."""
    first = frames[0]
    if _is_mapping(first):
        return _rebuild_mapping(
            first,
            {key: stack_observations([f[key] for f in frames], axis) for key in first.keys()},
        )
    if isinstance(first, torch.Tensor):
        return torch.stack(list(frames), dim=axis)
    return np.stack([np.asarray(f) for f in frames], axis=axis)


def stack_space(space: gym.Space, n_frames: int, axis: int) -> gym.Space:
    """The observation space of `stack_observations` applied to samples from `space`."""
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {key: stack_space(sub, n_frames, axis) for key, sub in space.spaces.items()}
        )
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.stack([space.low] * n_frames, axis=axis),
            high=np.stack([space.high] * n_frames, axis=axis),
            dtype=space.dtype,
        )
    raise TypeError(
        f"FrameStack can only describe Dict and Box observation spaces, got {type(space)}"
    )


def _wrapper_attr(env, name: str, default=None):
    """`env.<name>` from anywhere in the wrapper stack, or `default`.

    gymnasium >= 1.0 dropped `Wrapper.__getattr__`, so a plain `getattr` only sees the outermost
    wrapper; `get_wrapper_attr` is the replacement that walks inwards.
    """
    try:
        return env.get_wrapper_attr(name)
    except (AttributeError, TypeError):
        return getattr(env, name, default)


class _AttributeForwarding:
    """Forward unknown public attributes inwards, the way pre-1.0 gymnasium wrappers did.

    Everything in this project reaches through the wrapper stack for ManiSkill's own API
    (`base_env`, `control_freq`, `get_state_dict`, ...), which gymnasium >= 1.0 only supports via
    the explicit `get_wrapper_attr`. Private names are deliberately not forwarded: letting
    `__deepcopy__`/`__getstate__`/`_frames` style lookups fall through hides real bugs.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "env":
            raise AttributeError(name)
        env = self.__dict__.get("env")
        if env is None:
            raise AttributeError(name)
        try:
            return env.get_wrapper_attr(name)
        except (AttributeError, TypeError):
            raise AttributeError(
                f"{type(self).__name__} and the envs it wraps have no attribute {name!r}"
            ) from None


# --------------------------------------------------------------------------------------------- #
# episode-boundary helpers, shared by FrameSkip
# --------------------------------------------------------------------------------------------- #


def _episode_over(terminated, truncated):
    if isinstance(terminated, torch.Tensor) or isinstance(truncated, torch.Tensor):
        return torch.as_tensor(terminated) | torch.as_tensor(truncated)
    return bool(terminated) or bool(truncated)


def _all(flag) -> bool:
    if isinstance(flag, torch.Tensor):
        return bool(flag.all())
    return bool(np.all(flag))


def _keep_running(done):
    """The complement of `done`: which envs' next sub-step counts."""
    if isinstance(done, torch.Tensor):
        return ~done
    return not done


def _select(new, old, keep):
    """`new` where `keep`, `old` elsewhere, elementwise over the leading (num_envs) axis.

    Leaves that are not tensors of a maskable shape are taken from `new` wholesale -- a non-tensor
    info entry, or one that is not batched per env, carries no per-env structure to mask.
    """
    if keep is True or keep is False:
        return new if keep else old
    if _all(keep):
        return new
    if _is_mapping(new) and _is_mapping(old):
        return _rebuild_mapping(
            new,
            {
                key: _select(value, old[key], keep) if key in old else value
                for key, value in new.items()
            },
        )
    if (
        isinstance(new, torch.Tensor)
        and isinstance(old, torch.Tensor)
        and new.shape == old.shape
        and new.ndim > 0
        and new.shape[0] == keep.shape[0]
    ):
        mask = keep.reshape((-1,) + (1,) * (new.ndim - 1))
        return torch.where(mask, new, old)
    return new


# --------------------------------------------------------------------------------------------- #
# wrappers
# --------------------------------------------------------------------------------------------- #


class IgnoreTerminations(_AttributeForwarding, gym.Wrapper):
    """Report `terminated=False` always, so episodes end only when the caller says so.

    ManiSkill sets `terminated` to `info["success"]` recomputed from the current state on every
    step (`BaseEnv.step`), so it is a momentary predicate, not a true absorbing state -- it flips
    back to False if the goal stops being satisfied. Honouring it would end the episode the first
    time the goal is met, turning "reach the goal and hold it" into "reach it once" and forfeiting
    every subsequent reward (ManiSkill's dense rewards are positive at every step, so an early
    cutoff lowers the return). The tabletop tasks used here have no genuine absorbing states, so
    episodes end only on truncation (the time limit). Nothing is lost: the signal is still
    available as `info["success"]`.

    ManiSkill offers no construction-time equivalent that is usable underneath other wrappers:
    `BaseEnv` takes no termination-related kwarg, `ManiSkillVectorEnv(ignore_terminations=True)`
    also brings auto-reset and the gym vector API, and `CPUGymWrapper(ignore_terminations=True)`
    converts observations to unbatched numpy and must be outermost.

    Note this wrapper is *not* what keeps the env from resetting itself: plain `gym.make` never
    auto-resets (only `ManiSkillVectorEnv`/`make_vec` does), so a truncated episode stays in its
    final state until the caller calls `reset`.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(terminated, torch.Tensor):
            terminated = torch.zeros_like(terminated)
        else:
            terminated = False
        return obs, reward, terminated, truncated, info


class FrameSkip(_AttributeForwarding, gym.Wrapper):
    """Act in the space of `frame_skip` consecutive primitive actions.

    One `step` takes the concatenation of the next `frame_skip` actions -- shape `(frame_skip * D,)`
    where the wrapped env takes `(D,)`, or `(num_envs, frame_skip * D)` batched -- applies them in
    order, and reports the observation after the last one. This is the open-loop-chunk action space
    DINO-WM plans in (`rearrange(actions, "b (t f) d -> b t (f d)")` in its planner and
    `TrajSlicerDataset`'s `action_dim * frameskip`); as a wrapper, the env itself now speaks it, so
    the planner, the online trainer and the dataset all see the same interface.

    Semantics across the skipped steps:

    - reward is the **sum** over the sub-steps, which keeps returns comparable to the unskipped env
    - `terminated`/`truncated` are OR-ed
    - the observation and info are those of the last sub-step actually taken

    Sub-stepping stops early once every env reports done, so an episode cannot run past its time
    limit just because the limit fell in the middle of a chunk. With `num_envs > 1` and envs
    finishing at different sub-steps, data for an env that has already finished is frozen at its
    final step (the masking `mani_skill.utils.wrappers.ActionRepeatWrapper` does for the same
    reason), so a finished env contributes no further reward and no later observation.

    `frame_skip=1` is a pass-through apart from the (unchanged) action space.
    """

    def __init__(self, env: gym.Env, frame_skip: int = 1):
        super().__init__(env)
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")
        self.frame_skip = frame_skip

        inner_action_space = env.action_space
        self._action_dim = int(inner_action_space.shape[-1])
        self.action_space = self._repeat_action_space(inner_action_space)
        single = _wrapper_attr(env, "single_action_space")
        if single is not None:
            # ManiSkill's per-env action space, which stays meaningful when num_envs > 1
            self.single_action_space = self._repeat_action_space(single)

    def _repeat_action_space(self, space: gym.Space) -> gym.Space:
        if self.frame_skip == 1:
            return space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError(
                f"FrameSkip needs a Box action space to concatenate, got {type(space)}. "
                "Flatten dict actions first (mani_skill.utils.wrappers.FlattenActionSpaceWrapper)."
            )
        return gym.spaces.Box(
            low=np.concatenate([space.low] * self.frame_skip, axis=-1),
            high=np.concatenate([space.high] * self.frame_skip, axis=-1),
            dtype=space.dtype,
        )

    def _split_action(self, action):
        if self.frame_skip == 1:
            return [action]
        if not isinstance(action, (torch.Tensor, np.ndarray)):
            action = np.asarray(action, dtype=np.float32)
        width = action.shape[-1]
        if width != self.frame_skip * self._action_dim:
            raise ValueError(
                f"FrameSkip(frame_skip={self.frame_skip}) expects {self.frame_skip} x "
                f"{self._action_dim} = {self.frame_skip * self._action_dim} action values, got "
                f"{width}"
            )
        if isinstance(action, torch.Tensor):
            return list(torch.split(action, self._action_dim, dim=-1))
        return list(np.split(action, self.frame_skip, axis=-1))

    def step(self, action):
        sub_actions = self._split_action(action)
        obs, reward, terminated, truncated, info = self.env.step(sub_actions[0])
        for sub_action in sub_actions[1:]:
            done = _episode_over(terminated, truncated)
            if _all(done):
                break
            keep = _keep_running(done)
            if not _all(keep):
                # Some env has finished, so its observation has to survive the sub-steps the
                # others still take -- and ManiSkill would overwrite it in place (see
                # clone_observations). Only reached with num_envs > 1 and staggered episode ends,
                # so the common path pays nothing.
                obs = clone_observations(obs)
                info = clone_observations(info)
            new_obs, new_reward, new_terminated, new_truncated, new_info = self.env.step(
                sub_action
            )
            obs = _select(new_obs, obs, keep)
            info = _select(new_info, info, keep)
            # `+`/`|` rather than in-place ops: the env owns the tensors it returned
            reward = reward + new_reward * keep
            terminated = _select(new_terminated | terminated, terminated, keep)
            truncated = _select(new_truncated | truncated, truncated, keep)
        return obs, reward, terminated, truncated, info

    def rand_act(self):
        """A uniform random macro action, as a float32 torch tensor.

        Here rather than only on an inner wrapper because a `rand_act` that sampled the *primitive*
        action space would silently produce actions `frame_skip` times too short.
        """
        return torch.from_numpy(np.asarray(self.action_space.sample(), dtype=np.float32))


class FrameStack(_AttributeForwarding, gym.Wrapper):
    """Return the last `n_frames` observations stacked along a new axis.

    The axis is inserted at `frame_axis`, which defaults to 1 because ManiSkill observations carry
    a leading `num_envs` axis: `(num_envs, ...)` becomes `(num_envs, n_frames, ...)`. Pass
    `frame_axis=0` when an observation adapter underneath has already squeezed the batch axis away
    (as both the TSD and DINO-WM adapters do at `num_envs=1`), giving `(n_frames, ...)`.

    Stacked at whatever rate the env underneath steps at, so putting this outside `FrameSkip` --
    which `make_env` does -- stacks frames `frame_skip` primitive steps apart, matching how
    `TrajSlicerDataset` reads observations at the frameskip stride. Putting it inside would stack
    consecutive sim frames instead.

    The frame axis is added even at `n_frames=1`, so that observations keep the same rank (and a
    model keeps the same input shape) as `n_frames` is varied.

    After `reset` the buffer holds `n_frames` copies of the initial observation.
    """

    def __init__(self, env: gym.Env, n_frames: int = 1, frame_axis: int = 1):
        super().__init__(env)
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")
        self.n_frames = n_frames
        self.frame_axis = frame_axis
        self._frames: deque = deque([], maxlen=n_frames)
        self.observation_space = stack_space(env.observation_space, n_frames, frame_axis)

    def _stacked(self, frame, is_reset: bool = False):
        # copied because ManiSkill mutates its image buffers in place; see clone_observations
        frame = clone_observations(frame)
        for _ in range(self.n_frames if is_reset else 1):
            self._frames.append(frame)
        return stack_observations(list(self._frames), self.frame_axis)

    def reset(self, **kwargs):
        options = kwargs.get("options")
        num_envs = _wrapper_attr(self.env, "num_envs", 1)
        if (
            isinstance(options, dict)
            and "env_idx" in options
            and len(options["env_idx"]) < num_envs
        ):
            raise RuntimeError(
                "FrameStack cannot honour a partial reset: it keeps one frame buffer for the whole "
                "batch, so the envs left running and the ones just reset would share a history."
            )
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            frame, info = result
            return self._stacked(frame, is_reset=True), info
        # the TSD observation adapters return the observation alone
        return self._stacked(result, is_reset=True)

    def step(self, action):
        frame, reward, terminated, truncated, info = self.env.step(action)
        return self._stacked(frame), reward, terminated, truncated, info
