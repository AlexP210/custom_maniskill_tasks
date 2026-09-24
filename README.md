# custom_maniskill_tasks

One definition of the ManiSkill tasks this project trains, plans and records datasets with, so that
an online rollout, a planning rollout and a recorded dataset cannot drift apart. The pieces that
used to be copied between `agents/tsd/tsd/tasks/maniskill_task.py`, `tools/replay_trajectory.py`
and `agents/dino_wm/dino_wm/env/pushcube/pushcube_wrapper.py` all live here.

```bash
pip install -e environments/custom_maniskill_tasks
```

## Registered task ids

Importing the package registers the ids this project's datasets are collected with:

| id | is |
| --- | --- |
| `PushCube-v1.1` | `PushCube-v1` with no early termination |
| `PlaceSphere-v1.1` | `PlaceSphere-v1` with no early termination |
| `LiftPegUpright-v1.1` | `LiftPegUpright-v1` with no early termination |
| `PokeCube-v1.1` | `PokeCube-v1` with no early termination |
| `PickCube-v1.1` | `PickCube-v1` with no early termination |
| `PickSingleYCB-v1.1` | `PickSingleYCB-v1` with no early termination |

`PickSingleYCB-v1.1` is the one id whose scene is not fixed: the YCB object in it is drawn per
parallel env at reconfiguration, and the stock task's own `reconfiguration_freq` default (1 at
`num_envs=1`, so a new object every reset; 0 above it, so one draw held for the run) is carried
over untouched. It also needs the YCB assets, which the variant declares exactly as the stock task
does, so `gym.make` offers the download:

```bash
python -m mani_skill.utils.download_asset ycb
```

"No early termination" is the one environment-level convention
[tools/ppo_stages_fast.py](../../tools/ppo_stages_fast.py) collected every offline dataset under
(`ManiSkillVectorEnv(..., ignore_terminations=True)`): an episode is never cut short by reaching the
goal, so it runs the full 50-step horizon and holding the goal early earns strictly more dense
reward. Baking it into the task id means nothing downstream has to remember the flag, and a
recorded trajectory's `env_id` says which convention produced it. `info["success"]` is untouched;
scene, dynamics, reward and success predicate are the stock task's, asserted in
`tests/test_tasks.py`. `control_mode` defaults to the `pd_ee_delta_pos` every recording used, and an
explicit `gym.make(..., control_mode=...)` still wins.

These ids also take a `lighting` kwarg (see [Lighting](#lighting)), which the stock `-v1` ids do
not.

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
    lighting="default",      # "default" | "dim" | "bright" | "warm" | "cool" | "side" | "random"
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

## Lighting

`lighting` picks the condition the scene is rendered under, for evaluating a policy on a domain it
was not trained on. Unlike `camera_view` it is an ordinary env kwarg on the `-v1.1` ids
(`LightingMixin`), so `gym.make` records it in `env.spec.kwargs`, `RecordEpisode` writes it into the
trajectory json, and a replay of that dataset rebuilds the same condition without being told —
a shifted recording is distinguishable from a default-lit one on disk. Asking for a shift on a
stock `-v1` id raises rather than silently doing nothing.

| preset | what it is |
| --- | --- |
| `default` | `BaseEnv._load_lighting` transcribed: ambient `0.3`, two white directional lights |
| `dim` / `bright` | the same lighting at half / one-and-a-half times the level |
| `bright-set-<A>-<B>` | ambient set to `A` and both directional lights to `B` (white), e.g. `bright-set-0.45-1.5` |
| `warm` / `cool` | the same geometry and roughly the same exposure, under a colour cast |
| `warm-set-<t>` / `cool-set-<t>` | every light tinted `t` (0 to 1) of the way along the blackbody curve from 6500 K to 2700 K (incandescent) / 12000 K (blue-sky shade), at constant luminance; `t = 0` is `default`, and `t` runs on past 1 up to 1.6 (about 2000 K / 24400 K) |
| `side` | the key light crossed to the other side and raked lower, so shading falls the other way |
| `side-set-<t>` | the key light turned `t` (0 to 1) of the way from `default`'s direction to `side`'s, over the top; `side-set-0` is `default`, `side-set-1` is `side` |
| `random` | one condition drawn per parallel env — training-time domain randomization |

A dict of the same shape as `LightingConfig` is accepted wherever a preset name is, for a one-off
condition; it stays json-serializable, so it still records and replays. Unknown preset names and
misspelled config keys raise before the env is built, because a shift that quietly did not happen
looks exactly like a policy that is robust to it.

`lighting="default"` is not merely close to the stock lighting, it is the stock lighting: the
preset reproduces `BaseEnv`'s numbers down to the shadow parameters, it is not passed to `gym.make`
at all, and `tests/test_lighting.py::test_default_is_byte_identical` asserts the rendered frame is
unchanged. That matters because every dataset here was recorded under it, and a policy evaluated at
`default` has to be looking at the same scene its training frames came from.

One caveat, inherited from where ManiSkill loads lighting: `_load_lighting` runs only inside
`_reconfigure`, so `random` draws once per env at construction (off ManiSkill's fixed `2022 + i`
seeds) and holds it — the same `num_envs` conditions every run, and a reset seed does not move
them. `reconfiguration_freq=1` redraws each reset at the cost of rebuilding the scene. To measure
performance across conditions, build the env once per named preset instead.

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

No pytest in the project environment, so the files run standalone:

```bash
python tests/test_wrappers.py   # wrapper semantics against a fake env, no simulator
python tests/test_tasks.py      # the -v1.1 ids: registration, and what they do/don't change
python tests/test_make_env.py   # real PushCube-v1 envs: resets, views, frame_skip equivalence
python tests/test_lighting.py   # presets: default parity, that each shift shows, per-env draws
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
