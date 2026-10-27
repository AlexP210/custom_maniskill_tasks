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
| `PushCube-v1.2`, `PlaceSphere-v1.2`, `LiftPegUpright-v1.2`, `PokeCube-v1.2`, `PickCube-v1.2`, `PickSingleYCB-v1.2` | the `-v1.1` task of the same name on the Franka Research 3, with the real robot's ZED mini wrist camera: see [FR3 and the ZED mini](#fr3-and-the-zed-mini-v12) |

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

These ids also take `perturbations`, which changes how the scene looks (see
[Perturbations](#perturbations)) and which the stock `-v1` ids do not.

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

## Perturbations

One kwarg, `perturbations`, changes how the scene looks, for training and evaluating a policy on
domains it was not recorded in. It names *perturbation classes*, each a group of *aspects* that
change together, and says per aspect what to do:

| class | aspects |
| --- | --- |
| `lighting` | `brightness` (x the stock light levels), `direction` (of the key light, 0 stock to 1 from the side), `temperature` (-1 is 2700 K, 0 the stock 6500 K, 1 is 12000 K) |
| `object_colour` | `r`, `g`, `b` (added to the manipulated object's channels) |
| `table_colour` | `r`, `g`, `b` (gains on the table's channels), `brightness` |
| `table_texture` | `scale`, `shear`, `rotation`, `shift_x`, `shift_y` of the table's texture |
| `table_material` | `material`: wood (`default`), `marble`, `metal`, `concrete`, `rubber`, `glass`, `gold`. Visual only: friction and contact are unchanged |
| `viewpoint` | `left`, `up`, `forward` (cm), `pitch`, `yaw`, `roll` (degrees), `fov` (degrees, about the rig's own); wrist camera only |

Units and ranges are in `perturbation_classes.registry()`. A class is

- `default`: every aspect at its neutral value, the scene every dataset here was recorded in;
- `randomize-train`, `randomize-evaluation` or `randomize-full`: every aspect drawn afresh each
  episode, from the central 80% of its range, from the 20% training leaves out (both ends of a
  two-sided range, the top end of a one-sided one), or from all of it. For `table_material`,
  training draws all but `glass` and `gold`, which are the evaluation materials;
- a mapping of aspects, each `default`, a number inside its range, or one of those `randomize-*`
  strings. Aspects not named are `default`.

Classes not named are `default`.

```python
env = make_env(
    "PickCube-v1.1", camera_view="wrist",
    perturbations={
        "lighting": {"temperature": "randomize-train", "brightness": 1.5},  # direction stays stock
        "table_colour": "randomize-train",
        "table_material": "randomize-evaluation",                           # glass or gold
        "viewpoint": {"pitch": -5, "yaw": "randomize-train"},
    },
)
env.unwrapped.episode_perturbations   # what the current episode drew, as {class: {aspect: value}}
```

Fixed values are checked against the aspect's full range and anything outside it raises, as does
an unknown class, aspect or split: a perturbation that quietly did not happen looks exactly like a
policy that is robust to it. Asking for any on a stock `-v1` id raises too, and so does
`viewpoint` without `camera_view="wrist"`.

Anything randomized rebuilds the scene on every reset (`reconfiguration_freq=1`, roughly half a
second), since lights, materials and textures cannot change once the renderer has them and the
wrist camera is re-posed with the sensors; a config of fixed values builds it once. Draws are
seeded by the reset's seed and the (class, aspect), so the same `reset(seed=...)` gives the same
scene, an unseeded reset a fresh one, and fixing or randomizing one aspect never changes what
another draws. One draw is shared by every parallel env, which reset together. They never touch
the task's own episode rng.

Like every kwarg the tasks take, `perturbations` reaches the env itself, so `gym.make` records it
in `env.spec.kwargs` (in canonical form: only what is not default), `RecordEpisode` writes it into
the trajectory json, and a replay of a *fixed* config rebuilds the same scene without being told. An
all-default config passes nothing, so the stock recordings' metadata is unchanged.

The earlier `lighting`, `texture`, `table_material`, `distractors`, `viewpoint`, `corruption` and
`domain_randomization` kwargs, and their presets, are gone: passing one is a `TypeError`.

## Usage

```python
from custom_maniskill_tasks import make_env

env = make_env(
    "PushCube-v1.1",
    obs_mode="rgb",
    control_mode="pd_ee_delta_pos",
    camera_view="wrist",     # "default" | "focused" | "wrist"
    perturbations=None,      # see Perturbations: {"lighting": "randomize-train", ...}
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

## FR3 and the ZED mini (v1.2)

For sim-to-real: the `-v1.2` ids are the `-v1.1` tasks (same scene, dynamics, reward, success
predicate, 50-step horizon, `pd_ee_delta_pos`, and the same perturbation kwargs) on the robot and
camera the real setup has -- a Franka Research 3 with a ZED mini on the hand. Use them with
`camera_view="wrist"`:

```python
env = make_env("PickCube-v1.2", camera_view="wrist")   # 224 x 224 hand_camera, robot "fr3_zed"
```

**Robot.** ManiSkill has no FR3, so `fr3.py` builds one from Franka's `franka_description`
(Apache-2.0; URDF and meshes under `custom_maniskill_tasks/assets/fr3`, with its licence and
NOTICE; `build_fr3_assets.py` regenerates them from a checkout). It shares the Panda's seven joints,
Franka Hand, 0.1034 m TCP and rest pose, so the controllers and grasp logic are the Panda's;
the FR3's own joint limits and `fr3_*` names apply. `robot_uids` is `fr3` (arm and hand alone, what
every other camera view gets) or `fr3_zed` (the same with the camera, which `make_env` picks under
`camera_view="wrist"` and which is recorded into a trajectory's env kwargs). A `-v1.2` task refuses
another robot. The stock tasks choose their start pose by comparing `robot_uids` to `"panda"`, so
`FR3Mixin` has them see a panda while those hooks run; `tests/test_fr3.py` checks the reset TCP
and joints equal the Panda's on all six tasks and that `PickCube`'s reward takes the panda branch.

**Camera.** Pose and intrinsics are the real unit's (`ZED_CAMERA_SIM_CONFIG.md`): mounted on
`fr3_hand_tcp` at (0.0642, -0.0315, -0.0640) m, RPY (0, -20, 90) degrees in the ROS convention
(converted to sapien's; the optical axis in the TCP frame is (-0.342, 0, 0.940)), 2 mm near clip,
no mount model. The ZED SDK's rectified HD720 left image has HFOV 82.504 degrees (`fx` =
729.734 px at 1280 wide). The policy sees 224 x 224, so **the real side must take the centre
720 x 720 crop of that rectified frame and resize it to 224 x 224**; that is a square pinhole
image with the 720 rows' 52.517 degree field of view on both axes (`fx = fy` = 227 px), which is what is
rendered. Any other real-side preprocessing is not what this simulates. No photometric or
camera-jitter randomization is applied beyond what `perturbations` asks for.

**What differs from `-v1.1` in the perturbation machinery.** The camera is a `WristRig`
(`cameras.py`): `ViewpointConfig` carries the rig it shifts, so the viewpoint geometry and the
nominal camera pose follow the robot. `viewpoint.fov` is in degrees about the rig's own, so on the
ZED it runs 43.8 to 61.3 (neutral 52.517) where on the Panda it runs 90 to 126 (neutral 108); the
other viewpoint aspects are shifts relative to whichever camera it is, so the same ranges apply.
They were set for the Panda's view and have not been re-tuned for the ZED, which sees only about
16 x 21 cm of table at reset.

## Tests

No pytest in the project environment, so the files run standalone:

```bash
python tests/test_wrappers.py   # wrapper semantics against a fake env, no simulator
python tests/test_tasks.py      # the -v1.1 ids: registration, and what they do/don't change
python tests/test_perturbation_classes.py   # the class/aspect registry, resolver and splits; no simulator
python tests/test_perturbation_mixin.py     # `perturbations` on real envs: each class shows, redraws, reproduces
python tests/test_make_env.py   # real PushCube-v1 envs: resets, views, frame_skip equivalence
python tests/test_fr3.py        # the -v1.2 ids: the FR3, the ZED's pose and intrinsics, rig-aware fov
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
