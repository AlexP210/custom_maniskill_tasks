"""The one place a ManiSkill task in this project gets built."""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401  (registers the task ids with gymnasium)

from custom_maniskill_tasks.backends import backend_kwargs
from custom_maniskill_tasks.cameras import CamerasMixin, cameras_config, check_cameras, resolve_cameras
from custom_maniskill_tasks.lighting import check_lighting
from custom_maniskill_tasks.perturbation_classes import resolve_perturbations
from custom_maniskill_tasks.perturbation_mixin import check_perturbations, supports_perturbations
from custom_maniskill_tasks.viewpoint import rig_for_task
from custom_maniskill_tasks.wrappers import FrameSkip, FrameStack, IgnoreTerminations

TASKS_IN_USE = (
    # the stock tasks minus early termination, registered by importing this package (see `tasks`);
    # this project's datasets are collected with these
    "PushCube-v1.1",
    "PlaceSphere-v1.1",
    "LiftPegUpright-v1.1",
    "PokeCube-v1.1",
    "PickCube-v1.1",
    "PickSingleYCB-v1.1",
    # the same on the Franka Research 3 with the real robot's ZED mini wrist camera (see `fr3`),
    # for sim-to-real; these ones want a `wrist` camera
    "PushCube-FR3",
    "PlaceSphere-FR3",
    "LiftPegUpright-FR3",
    "PokeCube-FR3",
    "PickCube-FR3",
    "PickSingleYCB-FR3",
    # LiftPegUpright-v1.1 with the peg to be stood on its marked end (see `marked_peg`)
    "LiftPegUpright-HarderPerception",
    # PegInsertionSide minus early termination on the plain panda, 5-D pd_ee_delta_pos_yaw control
    # (see `peg_insertion`, `controllers`)
    "PegInsertionSide-v1.1",
    # PlugCharger minus early termination, on the stock scene camera, sparse reward (see `tasks`)
    "PlugCharger-v1.1",
    # -v1.1 tasks with scripted distractor robots beside the agent, seen through the shared front
    # camera, plus an optional video backdrop (see `distractor_scene`, `distractor_tasks`)
    "PegInsertionSide-MultiArmDistractors",
    "LiftPegUpright-MultiArmDistractors",
    "PickCube-MultiArmDistractors",
    "PickCube-MultiArmDistractors-ShuffledPositions",
    # stock ManiSkill tasks that older datasets and configs still name
    "PushCube-v1",
    "PickCube-v1",
    "PegInsertionSide-v1",
    "PlaceSphere-v1",
    "LiftPegUpright-v1",
    "PokeCube-v1",
)
"""The tasks this project has datasets and configs for. Documentation, not a restriction:
`make_env` works for any registered ManiSkill task id, including the deprecated aliases in
`tasks.DEPRECATED_TASK_IDS`."""


def make_env(
    task_name: str,
    *,
    obs_mode: str = "rgb",
    control_mode: str | None = "task",
    num_envs: int = 1,
    cameras: Sequence[Mapping[str, Any]] | None = None,
    perturbations: Mapping[str, Any] | None = None,
    frame_skip: int = 1,
    n_frames: int | None = None,
    frame_axis: int = 1,
    ignore_terminations: bool = True,
    sim_backend: str | None = None,
    render_backend: str | None = None,
    render_mode: str | None = "rgb_array",
    reward_wrappers: Sequence[Callable[[gym.Env], gym.Env]] = (),
    obs_wrappers: Sequence[Callable[[gym.Env], gym.Env]] = (),
    **env_kwargs,
) -> gym.Env:
    """Build `task_name` with the cameras, action chunking and frame stacking asked for.

    The env never resets itself: `gym.make` applies no auto-reset (only `make_vec` /
    `ManiSkillVectorEnv` do), `ignore_terminations` stops a momentary success from ending the
    episode, and nothing here calls `reset`, so the caller owns every reset and the RNG stream that
    seeds it. Truncation from the task's time limit is still reported, as the signal to reset.

    Wrapper order is `IgnoreTerminations` -> `FrameSkip` -> `reward_wrappers` -> `obs_wrappers` ->
    `FrameStack`, so a stacked observation spans `n_frames` macro steps of `frame_skip` primitive
    steps each, and an observation adapter placed in `obs_wrappers` sees single unstacked frames.

    Args:
        task_name: a registered ManiSkill task id, e.g. "PushCube-v1.1"; see `TASKS_IN_USE`. The
            `-v1.1` ids are this project's own (registered by importing this package): the stock
            tasks with early termination removed, which is how every offline dataset here was
            collected. On those, `ignore_terminations` is already true of the task itself.
        obs_mode, control_mode, num_envs: passed to `gym.make`. `control_mode="task"` (the
            default) is the task id's registered default where it has one (`pd_ee_delta_pos` on
            every `-v1.1` id but PegInsertionSide's 5-D `pd_ee_delta_pos_yaw`), and otherwise (the
            stock `-v1` ids) `pd_ee_delta_pos`, the one every recorded dataset in this project
            uses; a policy run under a different control mode than its data was recorded with is a
            silent failure. `control_mode=None`
            means "whatever the task id already defaults to" and is not forwarded, since
            `gym.make(control_mode=None)` would override a registration default (the `-v1.1` ids
            carry one) rather than defer to it.
        perturbations: how the scene differs from the stock one, as {class: spec} -- see
            `perturbation_classes`. The classes are `lighting`, `object_colour`, `table_colour`,
            `table_texture`, `table_material` and `viewpoint` (which needs a `wrist` camera),
            each a group of aspects. A class is "default" (every aspect neutral), "randomize-train"
            / "randomize-evaluation" / "randomize-full" (every aspect drawn afresh each episode
            from that part of its range), or a dict of aspects, each of which is "default", a
            number inside its range, or one of those "randomize-*" strings; aspects not named are
            default, and so are classes not named. Anything randomized rebuilds the scene on
            every reset (`reconfiguration_freq=1`). `env.unwrapped.episode_perturbations` says
            what the current episode drew. Recorded as the env kwarg, in canonical form, only
            when something is not default.
        cameras: the observation cameras, a list of `{name, preset, resolution, fov}` specs (see
            `cameras`), e.g. `[{"name": "wrist", "preset": "wrist"}]`; the env renders exactly
            these, each under its preset's sensor uid. None leaves the task's own cameras exactly
            as registered (ManiSkill's 128 px). Recorded in `env.spec.kwargs`. On a stock `-v1`
            id, which has no `cameras` kwarg, the env is built from a subclass that adds it.
        frame_skip: act in chunks of this many primitive actions -- see `FrameSkip`. The task's
            time limit is unchanged and still counted in primitive steps, so an episode lasts
            `max_episode_steps / frame_skip` macro steps.
        n_frames: stack this many past observations -- see `FrameStack`. None leaves observations
            exactly as ManiSkill returns them; any integer (1 included) adds the frame axis.
        frame_axis: where `FrameStack` inserts that axis. 1 for raw ManiSkill observations, whose
            leading axis is `num_envs`; 0 if an `obs_wrappers` entry squeezed that axis away.
        ignore_terminations: never report `terminated` -- see `IgnoreTerminations`.
        sim_backend, render_backend: resolved and pinned to the same cuda device by
            `backend_kwargs`; None means ManiSkill's own choice (physx_cpu at `num_envs=1`).
        render_mode: for `env.render()`, which uses the separate human render camera and is
            unaffected by `cameras`.
        reward_wrappers: callables applied in order between `FrameSkip` and `obs_wrappers`, for
            wrappers that rewrite the reward (`DINORewardWrapper`). Underneath the observation
            adapters, so they still see ManiSkill's own observation dict; outside `FrameSkip`, so
            they are called once per macro step rather than once per primitive step.
        obs_wrappers: callables applied in order between `reward_wrappers` and `FrameStack`, for
            project-specific observation adapters (TSD's TensorDict view of the observation dict,
            DINO-WM's flat state/proprio one).
        **env_kwargs: anything else `gym.make` takes -- `reward_mode`, `reconfiguration_freq`,
            `max_episode_steps`, `shader_dir`, a `sensor_configs` dict (ManiSkill's own per-uid
            overrides, applied over the cameras), ...
    """
    specs = resolve_cameras(cameras)
    rig = rig_for_task(task_name)
    perturbation_config = resolve_perturbations(perturbations, fov_degrees=float(np.degrees(rig.fov)))
    if "viewpoint" in perturbation_config.active and not any(s.preset == "wrist" for s in specs or ()):
        raise ValueError(
            f"perturbations has 'viewpoint', which moves the wrist camera, but cameras={cameras_config(specs)} has none."
        )
    if "viewpoint" in perturbation_config.active and any(s.remounted for s in specs):
        raise ValueError(
            "perturbations' 'viewpoint' poses the wrist camera from its rig's mount and would undo the "
            "camera's `pitch`/`offset`; use one or the other."
        )
    if specs is not None:
        env_kwargs["cameras"] = cameras_config(specs)

    if control_mode == "task":
        from mani_skill.utils.registration import REGISTERED_ENVS

        control_mode = REGISTERED_ENVS[task_name].default_kwargs.get("control_mode", "pd_ee_delta_pos")
    if control_mode is not None:
        env_kwargs["control_mode"] = control_mode

    if not perturbation_config.is_default:
        if not supports_perturbations(task_name):
            raise ValueError(
                f"perturbations={perturbation_config.to_config()} was asked for, but {task_name} is "
                "not one of this project's own task ids and its class takes no `perturbations` "
                "kwarg. Use the corresponding -v1.1 id (see `TASKS_IN_USE`), which does."
            )
        # plain dicts and lists, so a hydra DictConfig is recorded into env.spec.kwargs (and any
        # trajectory json) as json
        env_kwargs["perturbations"] = perturbation_config.to_config()

    with with_cameras_kwarg(task_name, needed=specs is not None):
        env = gym.make(
            id=task_name,
            obs_mode=obs_mode,
            num_envs=num_envs,
            render_mode=render_mode,
            **backend_kwargs(sim_backend, render_backend, num_envs),
            **env_kwargs,
        )
    check_cameras(env, specs)
    # checked against what the env drew for its first episode
    if hasattr(env.unwrapped, "lighting"):  # the stock `-v1` ids have none
        check_lighting(env, env.unwrapped.lighting)
    check_perturbations(env)

    if ignore_terminations:
        env = IgnoreTerminations(env)
    if frame_skip > 1:
        env = FrameSkip(env, frame_skip)
    for wrapper in reward_wrappers:
        env = wrapper(env)
    for wrapper in obs_wrappers:
        env = wrapper(env)
    if n_frames is not None:
        env = FrameStack(env, n_frames, frame_axis=frame_axis)
    return env


@contextlib.contextmanager
def with_cameras_kwarg(task_name: str, needed: bool):
    """Build a task whose class lacks `CamerasMixin` (a stock `-v1` id) from a subclass that has it.

    The registry's class is swapped only for the duration of the `gym.make` call: the instance
    stays on the subclass (so a later reconfiguration still builds the listed cameras), while other
    envs of the same id in this process are untouched.
    """
    from mani_skill.utils.registration import REGISTERED_ENVS

    spec = REGISTERED_ENVS.get(task_name)
    if not needed or spec is None or issubclass(spec.cls, CamerasMixin):
        # an unknown id: let gym.make raise its own error
        yield
        return
    original_cls = spec.cls
    spec.cls = type(original_cls.__name__ + "WithCameras", (CamerasMixin, original_cls), {})
    try:
        yield
    finally:
        spec.cls = original_cls
