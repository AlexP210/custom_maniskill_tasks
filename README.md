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

These ids also take five kwargs that change how the scene looks: `lighting`, `texture`,
`distractors`, `viewpoint` and `corruption` (see [Visual perturbations](#visual-perturbations)),
which the stock `-v1` ids do not.

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

## Visual perturbations

Six independent kwargs change how the scene looks, for evaluating a policy on a domain it was not
trained on. Each defaults to "none", which is the scene every dataset here was recorded in, and any
combination can be asked for:

| kwarg | what it changes | e.g. |
| --- | --- | --- |
| `lighting` | the lights, the table's colour, and the objects' hue: see [Lighting](#lighting) | `"very-dim+warm"`, `"object-hue-30"` |
| `texture` | patterns on the table and objects: see [Texture](#texture) | `"table-checker+object-stripes"` |
| `table_material` | the table's material, visual only: see [Table material](#table-material) | `"marble"`, `"metal"` |
| `distractors` | how many YCB objects are scattered on the table: see [Distractors](#distractors) | `3` |
| `viewpoint` | where the wrist camera is and how wide it sees: see [Viewpoint](#viewpoint) | `"back-3+pitch-up-5"`, `"fov-90"` |
| `corruption` | the camera images: blur, noise, low resolution: see [Corruption](#corruption) | `"blur-2+noise-0.05"` |

```python
env = make_env("PickCube-v1.1", camera_view="wrist",
               lighting="dim", texture="table-checker", distractors=3,
               viewpoint="back-2", corruption="blur-1")
```

Like every kwarg the tasks take, all six reach the env itself, so `gym.make` records them in
`env.spec.kwargs`, `RecordEpisode` writes them into the trajectory json, and a replay rebuilds the
same scene without being told. A default one is not passed at all, so existing recordings' metadata
is unchanged. Asking for any on a stock `-v1` id raises rather than silently doing nothing, as does
an unknown keyword. Keywords stack with `+` (except `distractors`, which is a count).

### Randomizing

Any of the six also takes `"randomize"`, which draws the condition afresh on every reset from what
the evaluation runs: for each kwarg, the unperturbed scene, every non-neutral point on its axes'
grids in `axes.py` (what `run_sweep.py` runs) and every named material or pattern in `CATEGORIES`
(what `run_ungraded.py` runs), each equally likely (`axes.randomization_pool(kwarg)`).

```python
env = make_env("LiftPegUpright-v1.1", camera_view="wrist",
               lighting="randomize", viewpoint="randomize", corruption="randomize")
env.reset(seed=3)
env.unwrapped.episode_perturbation   # {'lighting': 'warm-set-0.4', 'viewpoint': 'default', 'corruption': 'blur-0.8'}
```

The kwargs are independent, so randomizing several stacks one draw from each (the evaluation runs
one perturbation at a time); one left at a fixed value stays fixed. The draw is seeded by the reset
seed, so the same `reset(seed=...)` gives the same condition, and is shared by all parallel envs.
Everything but `corruption` changes what is built, so a randomized one makes the env reconfigure on
every reset (`reconfiguration_freq=1`, slower, and every env must reset together). `viewpoint` needs
`camera_view="wrist"`. See `randomize.py`.

`perturbation_sheet.py` renders any combination side by side, wrist camera and third-person view,
for looking at them:

```
python perturbation_sheet.py --task PickCube-v1.1 default corruption=blur-2 \
    "texture=table-checker,distractors=3" "lighting=dim,viewpoint=back-4+pitch-up-6"
```

### Graded axes

Some of these can be dialled up from "none" to "a lot", and a robustness sweep wants the knob, its
neutral value and a grid to run. They are registered in `custom_maniskill_tasks/axes.py`, which
sweep scripts take their conditions from (`conditions("corruption.blur")` gives
`[(0, "blur-0"), (0.5, "blur-0.5"), ...]`, to pass straight to `make_env`). `tests/test_axes.py`
fails if a parser keyword or mixin kwarg is on neither an axis nor the "no axis" list, if a grid
value does not parse, or if an axis's neutral value is not the default scene. The table below is
generated (`python -m custom_maniskill_tasks.axes`) and the test checks it is current. A negative
value on a signed `viewpoint` axis is the opposite keyword (`-5` on `viewpoint.pitch` is
`pitch-down-5`).

<!-- axes:start -->
| axis | kwarg | neutral | unit | default grid | notes |
| --- | --- | --- | --- | --- | --- |
| `lighting.brightness` | `lighting` | 1 | x default level | 0.25, 0.5, 1, 2.5, 4, 8 (log) | ambient 0.3k and both lights k |
| `lighting.warm` | `lighting` | 0 | 0 = 6500 K, 1 = 2700 K | 0, 0.2, 0.4, 0.6, 0.8, 1 | constant luminance; the parser allows up to 1.6 |
| `lighting.cool` | `lighting` | 0 | 0 = 6500 K, 1 = 12000 K | 0, 0.2, 0.4, 0.6, 0.8, 1 | constant luminance; the parser allows up to 1.6 |
| `lighting.side` | `lighting` | 0 | 0 = default, 1 = side | 0, 0.2, 0.4, 0.6, 0.8, 1 | key light direction |
| `lighting.table` | `lighting` | 1 | x table colour | 0.01, 0.408, 0.806, 1.204, 1.602, 2 | grey scale; the r-g-b form is not on the axis |
| `lighting.object_hue` | `lighting` | 0 | degrees | 0, 12, 24, 36, 48, 60 | cyclic; 0 and 360 are the same; 180 is the far end |
| `corruption.blur` | `corruption` | 0 | px sigma | 0, 0.2, 0.4, 0.6, 0.8, 1 |  |
| `corruption.noise` | `corruption` | 0 | std, fraction of full scale | 0, 0.1, 0.2, 0.3, 0.4, 0.5 | 0.05 is about 13 grey levels |
| `corruption.lowres` | `corruption` | 1 | x downscale factor | 1, 2, 3, 4, 5, 6 (log) |  |
| `viewpoint.lateral` | `viewpoint` | 0 | cm, + left | -6, -4.2, -2.4, -0.6, 1.2, 3 |  |
| `viewpoint.vertical` | `viewpoint` | 0 | cm, + up | -1, 0.2, 1.4, 2.6, 3.8, 5 |  |
| `viewpoint.depth` | `viewpoint` | 0 | cm, + forward | -3, -1.8, -0.6, 0.6, 1.8, 3 | large `back` shows the hand mount |
| `viewpoint.pitch` | `viewpoint` | 0 | degrees, + up | -10, -4, 2, 8, 14, 20 |  |
| `viewpoint.yaw` | `viewpoint` | 0 | degrees, + left | -20, -14, -8, -2, 4, 10 |  |
| `viewpoint.roll` | `viewpoint` | 0 | degrees, + left side down | -15, -9, -3, 3, 9, 15 |  |
| `viewpoint.fov` | `viewpoint` | 108 | degrees | 90, 97, 104, 111, 118, 125 | allowed 10 to 170 |
| `distractors.count` | `distractors` | 0 | YCB objects | 0, 1, 2, 3 | count; up to about 3 always fit |

No axis: `table_material` (categorical: seven named materials with no order between them); `texture` (categorical: pattern x target; the repeat count `n` changes spatial frequency only, and there is no value at which a pattern is no pattern); `lighting:named` (the named presets (`dim`, `side`, `shadows`...) are points, not scales; their graded relatives are the axes above).

Ungraded conditions (`CATEGORIES`, run by `run_ungraded.py`):

- `table_material` (`table_material`): `marble`, `glass`, `metal`, `gold`, `concrete`, `rubber`
- `texture.table` (`texture`): `table-checker`, `table-stripes`, `table-dots`, `table-noise`
- `texture.object` (`texture`): `object-checker`, `object-stripes`, `object-dots`, `object-noise`
<!-- axes:end -->

### Texture

`texture` is `<target>-<pattern>` or `<target>-<pattern>-<n>`, stacked with `+`. Targets: `table`,
`object` (the task's object of interest: the cube, sphere or peg), `distractors`, or any actor's
name (`cube`, `sphere`, `peg`, `bin`, `goal_region`, ...). Patterns: `checker`, `stripes`, `dots`,
`noise`; `n` is how many times it repeats across the texture (a default per pattern). A more specific
target wins whatever order they are named in, so `"object-checker+cube-dots"` dots the cube and
checks anything else `object` means. An unknown target lists the actors there are, and `object` on
PokeCube (a cube *and* a peg) asks you to name one.

The table is replaced by the pattern in a light and a dark tone. Objects keep their colour and gain
the pattern as a darkening of it (a red cube under `checker` is red and dark red), so what differs is
texture alone, not colour.

### Table material

`table_material` is `"default"` (the wood), `"marble"`, `"glass"`, `"metal"` (brushed steel), `"gold"`,
`"concrete"` or `"rubber"`. It edits only the table's render material (colour, roughness, metallic,
specular, transmission); the collision box and its friction are separate and untouched, so contact
behaviour is identical under every material. `"glass"` is a pale, glossy look, not a true
see-through table. It stacks with `texture` (the pattern replaces the colour, the material's shine
stays) and `lighting`'s table tint. Materials are defined in `table_material.MATERIALS`.

### Viewpoint

Moves the wrist camera (`camera_view="wrist"`) from where the robot mounts it. Keywords, each with a
non-negative number, stacked with `+` (there are no signs, since `+` joins keywords):

| keyword | meaning |
| --- | --- |
| `left`, `right`, `up`, `down` `-<cm>` | move the camera across its image plane |
| `forward`, `back` `-<cm>` | move it along its optical axis, which points at the table |
| `pitch-up`, `pitch-down` `-<deg>` | turn it so the view tilts up / down |
| `yaw-left`, `yaw-right` `-<deg>` | turn it so the view swings left / right |
| `roll-left`, `roll-right` `-<deg>` | roll it, left / right side down |
| `fov-<deg>` | set the field of view outright (the default is the camera's own: 108 on the `-v1.1` tasks' realsense, 52.517 on the `-v1.2` tasks' ZED) |

The same axis twice adds up (`left-2+right-2` is no shift). It acts through the camera's
`sensor_configs`, and `make_env` checks the built camera is where it was asked to be. A large `back`
puts the camera behind its own hand mount, which then fills the lower half of the image.

### Corruption

Degrades the images the env returns, on every camera, leaving `env.render()` alone. `blur-<sigma>`
(Gaussian, in pixels), `noise-<std>` (additive, as a fraction of full scale: 0.05 is about 13 grey
levels) and `lowres-<factor>` (shrink by that factor, blow back up), stacked with `+` and always
applied in that order. Images keep their shape and dtype, so `lowres` renders at full resolution
and degrades afterwards. The noise is seeded by the reset seed, so an episode is reproducible.

## Distractors

`distractors=N` scatters N YCB objects on the table, `N` from 0 to 8, on any `-v1.1` task. They are
the first N of a fixed pool of compact household objects (a soup can, an apple, a pudding box, a
potted meat can, a Rubik's cube, a mug, an orange, a mustard bottle), so the scene is the same
objects every time and only their places and yaws change. Needs the YCB assets:
`python -m mani_skill.utils.download_asset ycb`.

Each lands at a random place wholly inside the wrist camera's view (at the shifted pose if there is
a `viewpoint`) and clear, by 4 cm, of the task's own objects, of the ground the task needs (the
strip PushCube pushes its cube along, PlaceSphere's bin) and of each other. The places come from
the episode RNG, so a given reset seed gives the same scene, and they are drawn after the task's
own, so the task's spawn is unchanged. Up to about 3 always fit; with more, a crowded draw (most
often PushCube, whose path takes room) falls back to the roomiest place, which can touch a
neighbour and be pushed apart by the physics.

```python
env = make_env("LiftPegUpright-v1.1", camera_view="wrist", distractors=3)
```

They are off by default, so a bare `gym.make` still builds the scene every dataset was recorded in.
The objects are dynamic, so the robot can knock them, and they are put back on every reset. Nothing
reads them: observations (including `state`), reward and success are unchanged, which
`tests/test_tasks.py` checks. `distractors` used to be a flag for copies of the task's own object;
`distractors=True` now raises. In TSD, set `task.cfg.distractors=3` (a count).

## Usage

```python
from custom_maniskill_tasks import make_env

env = make_env(
    "PushCube-v1.1",
    obs_mode="rgb",
    control_mode="pd_ee_delta_pos",
    camera_view="wrist",     # "default" | "focused" | "wrist"
    lighting="default",      # "default" | "dim" | "bright" | "warm" | "cool" | "side" | "random"
    distractors=0,           # 0 to 8 YCB objects scattered in the wrist view
    texture="default", viewpoint="default", corruption="default",
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

`lighting` picks the condition the scene is rendered under. Unlike `camera_view` it is an ordinary
env kwarg on the `-v1.1` ids (`LightingMixin`), so a shifted recording is distinguishable from a
default-lit one on disk (see [Visual perturbations](#visual-perturbations)).

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
camera-jitter randomization is applied: the existing `lighting`, `corruption` and `viewpoint`
kwargs are the only perturbations.

**What differs from `-v1.1` in the perturbation machinery.** The camera is a `WristRig`
(`cameras.py`): `ViewpointConfig` carries the rig it was parsed for, so the viewpoint geometry, the
distractor visibility test and the nominal camera pose all follow the robot. `viewpoint.fov`'s axis
is re-centred on 52.517 (grid 42 to 62) for the ZED (`axes.RIG_AXES`, `axes.axis(name, "fr3_zed")`),
and `viewpoint="randomize"` draws from that. Two things to know:

- the ZED sees only about 16 x 21 cm of table at reset, so `distractors` are always put inside
  the view but cannot be kept clear of the task's own objects and of each other as on the Panda (one
  distractor overlapped in 4 of 10 resets, two or three in all 10, over seeds 0-9), and the placement falls back
  to the roomiest visible spot;
- the corruption grids (`blur`, `lowres`) are in pixels of the 224 x 224 image, as before, and were set for the Panda's view; they have not been re-tuned for the ZED.

## Tests

No pytest in the project environment, so the files run standalone:

```bash
python tests/test_wrappers.py   # wrapper semantics against a fake env, no simulator
python tests/test_tasks.py      # the -v1.1 ids: registration, and what they do/don't change
python tests/test_perturbations.py   # corruption, viewpoint and texture
python tests/test_axes.py       # the graded-axis registry against the parsers, no simulator
python tests/test_make_env.py   # real PushCube-v1 envs: resets, views, frame_skip equivalence
python tests/test_lighting.py   # presets: default parity, that each shift shows, per-env draws
python tests/test_fr3.py        # the -v1.2 ids: the FR3, the ZED's pose and intrinsics, rig-aware viewpoint
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
