# custom_maniskill_tasks

One definition of the ManiSkill tasks this project trains, plans and records datasets with, so that
an online rollout, a planning rollout and a recorded dataset cannot drift apart. The pieces that
used to be copied between `agents/tsd/tsd/tasks/maniskill_task.py`, `tools/replay_trajectory.py`
and `agents/dino_wm/dino_wm/env/pushcube/pushcube_wrapper.py` all live here.

```bash
pip install -e environments/custom_maniskill_tasks
```

## Registered task ids

Importing the package registers the three ids this project's datasets are collected with:

| id | is |
| --- | --- |
| `PushCube-v1.1` | `PushCube-v1` with no early termination |
| `PlaceSphere-v1.1` | `PlaceSphere-v1` with no early termination |
| `LiftPegUpright-v1.1` | `LiftPegUpright-v1` with no early termination |

"No early termination" is the one environment-level convention
[tools/ppo_stages_fast.py](../../tools/ppo_stages_fast.py) collected every offline dataset under
(`ManiSkillVectorEnv(..., ignore_terminations=True)`): an episode is never cut short by reaching the
goal, so it runs the full 50-step horizon and holding the goal early earns strictly more dense
reward. Baking it into the task id means nothing downstream has to remember the flag, and a
recorded trajectory's `env_id` says which convention produced it. `info["success"]` is untouched;
scene, dynamics, reward and success predicate are the stock task's, asserted in
`tests/test_tasks.py`. `control_mode` defaults to the `pd_ee_delta_pos` every recording used, and an
explicit `gym.make(..., control_mode=...)` still wins.

```python
import custom_maniskill_tasks  # noqa: F401  -- registers the ids
import gymnasium as gym

env = gym.make("PushCube-v1.1", num_envs=1)   # plain gym.make works, no flags to remember
```

The registry is per process, so a tool that only does `import mani_skill.envs` will not find these
ids. `tools/ppo_stages_fast.py` (which collects the datasets) and `tools/replay_trajectory.py`
(which converts them) both import this package, so both accept a `-v1.1` task; anything else that
should needs the same import. Datasets recorded so far name the stock `-v1` ids and replay
unchanged.

## Usage

```python
from custom_maniskill_tasks import make_env

env = make_env(
    "PushCube-v1.1",
    obs_mode="rgb",
    control_mode="pd_ee_delta_pos",
    camera_view="wrist",     # "default" | "focused" | "wrist"
    camera_resolution=224,
    frame_skip=3,            # one step takes 3 concatenated actions
    n_frames=2,              # observations are the last 2 (macro) frames, stacked
)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.rand_act())
```

The env never resets itself. `gym.make` applies no auto-reset (only `make_vec` /
`ManiSkillVectorEnv` do), `ignore_terminations=True` (the default) stops a momentary success from
ending the episode, and `make_env` does not call `reset`, so every reset — and the RNG stream that
seeds it — belongs to the caller. Truncation from the task's time limit is still reported, as the
signal that a reset is due.

Wrapper order is

```
gym.make -> IgnoreTerminations -> FrameSkip -> obs_wrappers -> FrameStack
```

`obs_wrappers` is where a project-specific observation adapter goes (TSD's TensorDict view of the
observation dict, DINO-WM's flat state/proprio one). It sits below `FrameStack`, so it sees single
unstacked frames; pass `frame_axis=0` when it squeezes ManiSkill's leading `num_envs` axis away.

## Camera views

`camera_view` picks the camera the *observations* come from. `env.render()` is unaffected: it uses
ManiSkill's separate human render camera.

| view | what it is |
| --- | --- |
| `default` (a.k.a. `standard`) | the task's own `base_camera`, at its registered pose and fov |
| `focused` | that camera re-posed onto the tabletop workspace, through a narrow 36 degree fov |
| `wrist` | a hand-mounted fisheye (0.6π fov); the task's own camera is dropped unless `wrist_only=False` |

The poses, fovs and the 224 default resolution are the ones every recorded dataset in this project
was rendered through — data, not tunables. A dataset recorded through one view is useless to a
policy running in another, so `make_env` fails loudly if the view it was asked for is not in the
built scene rather than quietly handing back the task default.

Two caveats inherited from ManiSkill:

- `camera_view="wrist"` on a task whose robot has no wrist camera (`PushCube-v1`, `PlaceSphere-v1`,
  anything on plain `panda`) has to repoint the `panda` agent uid at `PandaHandCam`, and that is
  process-wide and permanent — the agent is rebuilt from the registry on every reconfiguration, so
  it cannot be scoped to one env. Building a `default`-view env for a `panda` task afterwards in
  the same process gives it a wrist camera too; the module warns when that happens. Build the two
  views in separate processes. Tasks already on `panda_wristcam` (`PegInsertionSide-v1` and
  friends) need none of this and keep the robot they were recorded with.
- `camera_resolution` applies to the default view too, where `ManiSkillTask.make_env` used to leave
  ManiSkill's own 128. Pass `camera_resolution=None` to reproduce that.

## FrameSkip and FrameStack

`FrameSkip(frame_skip=k)` makes the action space the concatenation of the next `k` primitive
actions — `(k * D,)`, or `(num_envs, k * D)` batched — applies them in order, and reports the
observation after the last. Reward is summed over the sub-steps, `terminated`/`truncated` are
OR-ed, and sub-stepping stops once every env is done so a chunk cannot overrun the time limit.
This is the chunked action space DINO-WM already plans in
(`rearrange(actions, "b (t f) d -> b t (f d)")`); as a wrapper, the env itself now speaks it.

`FrameStack(n_frames=n)` returns the last `n` observations stacked along a new axis, inserted at
`frame_axis` (1 by default, i.e. after ManiSkill's `num_envs` axis). The axis is added even at
`n_frames=1`, so a model's input rank does not change with `n_frames`.

They compose in that order, so a stack spans `n_frames` macro steps of `frame_skip` primitive steps
each — the same stride `TrajSlicerDataset` reads recorded observations at. The task's time limit is
unchanged and still counted in primitive steps, so an episode lasts `max_episode_steps / frame_skip`
macro steps.

`FrameStack` copies each frame before buffering it, which is not optional: ManiSkill hands back its
camera capture buffer itself, so an `rgb` tensor (and `info["elapsed_steps"]`) is overwritten in
place on the next step. A frame buffer that stored the tensor would hold N references to the
current frame and stack it with itself — an env that looks fine until a model is asked to infer
motion from it. `tests/test_make_env.py::test_image_frames_are_not_aliased` guards this.

## Tests

No pytest in the project environment, so both files run standalone:

```bash
python tests/test_wrappers.py   # wrapper semantics against a fake env, no simulator
python tests/test_tasks.py      # the -v1.1 ids: registration, and what they do/don't change
python tests/test_make_env.py   # real PushCube-v1 envs: resets, views, frame_skip equivalence
```

`test_wrappers.py` covers what the simulator cannot show: staggered episode ends across a batch
(the tasks share one time limit, so every env truncates together), where `FrameSkip` freezes a
finished env's observation and stops accumulating its reward.

## Call sites

Already using this module:

- [tools/ppo_stages_fast.py](../../tools/ppo_stages_fast.py) — builds both its training and eval envs
  with `make_env` and pins its devices with `backend_kwargs`; `--env_id` accepts the `-v1.1` ids and
  defaults to `PushCube-v1.1`. Driven by
  [tools/make_ppo_staged_maniskill_data.sh](../../tools/make_ppo_staged_maniskill_data.sh), whose
  per-task settings accept either id spelling.
- [tools/replay_trajectory.py](../../tools/replay_trajectory.py) — `--camera-view` is
  `build_sensor_configs` + `camera_view_applied`, device pinning is `backend_kwargs`, and the task
  ids come from the import; it no longer depends on the `tsd` package at all. It does not call
  `make_env`: it builds the env from the kwargs recorded in the trajectory json, and `RecordEpisode`
  needs one primitive action per step, which `FrameSkip`/`FrameStack` would break.

- [agents/tsd/tsd/tasks/maniskill_task.py](../../agents/tsd/tsd/tasks/maniskill_task.py) —
  `make_env` builds the online env, with TSD's `ManiSkillWrapper` passed in as an `obs_wrappers`
  entry and `n_frames=cfg.num_frames, frame_axis=0`. Its own `PandaHandCam`, `WRIST_CAMERA_FOV`,
  `sensor_configs` branches, `StackFrames` and termination suppression are gone (−101 lines).

Still to migrate:

| currently in | replace with |
| --- | --- |
| `pushcube_wrapper.py`'s `CAMERA_*` constants and its `gymnasium.make` (DINO-WM, and the copies under agents/TC-WM, agents/dino_bsmpc, agents/sparse_imagination) | `make_env("PushCube-v1", camera_view="focused", ...)`, keeping its flat-state adapter |

Nothing in this module reaches back into an agent package, so it can be adopted one call site at a
time.
