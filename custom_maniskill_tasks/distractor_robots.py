"""Distractor robots: scripted Pandas beside the agent, each doing a task of its own.

`DistractorRobotsMixin` adds one Panda per named *station* (`STATIONS`: "left" and "right" of the
agent, as the agent sees it, i.e. +y and -y). Each station has its own cube and four pads at the
corners of a 0.25 x 0.3 m rectangle in front of its robot, and the robot spends the episode carrying
the cube from pad to pad in a fixed order (`Station.tour`: a zigzag, so two of every four carries
cross a diagonal, ~0.4 m) with a scripted pick-and-place. It is there to be seen, not to be
interacted with:

- **Not the agent.** The distractors are extra articulations, never part of `env.agent`, so the
  action space, the agent's proprioception, the reward and the success predicate are the wrapped
  task's own. They are built and stepped by the mixin alone.
- **Out of the task's way.** Each station is 0.7 m to the side of the agent, its pads at
  x in {-0.28, -0.03} and |y| in {0.64, 0.74, 0.94}: every pad 0.33-0.63 m out from its robot's
  base, inside the reach of a top-down grasp (at 0.27 m joint 4 hits its limit, at 0.68 m the hand
  bottoms out above the cube). Nothing filters collisions, though, and the
  inner pads are ~0.71 m from the agent's base, inside a Panda's reach: an agent that swings
  sideways can hit a distractor or its cube.
- **Replayable.** Robots and cubes are ordinary sim bodies, so they are in `get_state_dict()` and
  in every recorded `env_states`. The script is a function of that state, of `elapsed_steps` and
  of two per-episode constants (where in the tour the cube starts, and how many steps the robot
  idles before starting), drawn from the episode seed in a stream of their own, so the wrapped task's
  own draws are the same with and without distractors. As with the stock task's peg sizes, those
  constants are not in the state dict: `set_state_dict` restores where everything is, a seeded
  `reset` restores what the distractors will do next.
- **Cycle.** `CYCLE` lists the phases; one pick-and-place (one leg of the tour) takes
  `CYCLE_STEPS` (64) env steps, so a 100-step PegInsertionSide episode sees one carry and most of
  a second; which legs depends on the drawn start. The script is open-loop in time
  (the phase comes from the step count) and closed-loop in space (it reaches for wherever the cube
  actually is), so a missed grasp costs one cycle and the next one picks the cube up from where it
  fell.

Control is the robot's own `pd_ee_delta_pos` controller, given an action every env step from
`_before_control_step`, i.e. after the agent's own action is set and before the physics runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose

DISTRACTOR_PREFIX = "distractor_"
"""Every actor the mixin builds is named with this prefix, which `appearance._apply_object_hue`
skips: the distractors are not task objects, so perturbing the task's objects leaves them alone."""

CUBE_HALF_SIZE = 0.02
PAD_HALF_SIZE = 0.04
PAD_THICKNESS = 0.001

# rest pose: folded, hand pointing straight down 0.24 m above the pads (tcp at base + (0.32, 0, 0.24)).
# Not the agent's start pose, which holds the hand 0.6 m out at x=0 -- close to the scene camera,
# where it projects out to the edges of the frame.
REST_QPOS = np.array([0.0, -0.6, 0.0, -2.9, 0.0, 2.3, -np.pi / 4, 0.04, 0.04])

MAX_START_DELAY = 20
"""Each episode a distractor idles for 0..MAX_START_DELAY env steps before its first cycle, drawn
per station and env, so two stations (and two episodes) are not in lockstep."""


@dataclass(frozen=True)
class Station:
    """Where one distractor robot stands and where its cube goes (world frame, table top is z=0)."""

    base: tuple[float, float]
    yaw: float
    """Robot base yaw (rad); 0 faces +x, as the agent does."""
    pads: tuple[tuple[float, float], ...]
    tour: tuple[int, ...]
    """The order the cube visits `pads` in, cyclically; each episode starts at a drawn point in it."""
    cube_colour: str
    pad_colour: str


STATIONS: dict[str, Station] = {
    "left": Station(
        base=(-0.615, 0.7),
        yaw=0.0,
        # back-inner, back-outer, front-outer, front-inner ("front" is +x, towards the camera)
        pads=((-0.28, 0.64), (-0.28, 0.94), (-0.03, 0.94), (-0.03, 0.74)),
        tour=(0, 2, 1, 3),
        cube_colour="#2F6FDE",
        pad_colour="#5A5A5A",
    ),
    "right": Station(
        base=(-0.615, -0.7),
        yaw=0.0,
        pads=((-0.28, -0.64), (-0.28, -0.94), (-0.03, -0.94), (-0.03, -0.74)),
        tour=(0, 2, 1, 3),
        cube_colour="#2FA84F",
        pad_colour="#5A5A5A",
    ),
}


@dataclass(frozen=True)
class Phase:
    name: str
    steps: int
    target: str  # "cube" (track the cube's xy) or "place" (the destination pad's xy)
    z: float  # tcp height above the table top
    gripper: float  # 1 open, -1 closed
    speed: float  # max tcp travel per env step (m)


LIFT_HEIGHT = 0.22
LAG_GAIN = 2.0
"""Commanded delta / wanted delta. Measured: with gain 1 the tcp covers ~0.4 of each commanded
delta per step, so the script never got lower than 0.1 m before closing on air."""
HOVER_HEIGHT = 0.15
DESCENT_GUARD = 2.0
"""A phase's target height is raised by this times the tcp's remaining horizontal distance to its
target, up to `HOVER_HEIGHT`, so the hand only comes down below the hover once it is over it."""
CYCLE: tuple[Phase, ...] = (
    # the cube is right below after a retreat, so only the first approach of an episode travels far
    Phase("approach", 12, "cube", HOVER_HEIGHT, 1.0, 0.04),
    Phase("descend", 7, "cube", CUBE_HALF_SIZE, 1.0, 0.03),
    Phase("grasp", 5, "cube", CUBE_HALF_SIZE, -1.0, 0.02),
    Phase("lift", 7, "cube", LIFT_HEIGHT, -1.0, 0.035),
    Phase("transfer", 16, "place", LIFT_HEIGHT, -1.0, 0.035),  # a diagonal is ~0.4 m
    Phase("lower", 8, "place", CUBE_HALF_SIZE + 0.012, -1.0, 0.03),
    Phase("release", 4, "place", CUBE_HALF_SIZE + 0.012, 1.0, 0.02),
    Phase("retreat", 5, "place", HOVER_HEIGHT, 1.0, 0.04),
)
CYCLE_STEPS = sum(p.steps for p in CYCLE)


def _phase_table(device) -> dict[str, torch.Tensor]:
    """Per env-step-in-cycle lookup tables, so the script is a gather, not a python loop over envs."""
    rows = [p for p in CYCLE for _ in range(p.steps)]
    return dict(
        place=torch.tensor([p.target == "place" for p in rows], device=device),
        z=torch.tensor([p.z for p in rows], dtype=torch.float32, device=device),
        gripper=torch.tensor([p.gripper for p in rows], dtype=torch.float32, device=device),
        speed=torch.tensor([p.speed for p in rows], dtype=torch.float32, device=device),
    )


def canonical_distractors(distractor_robots) -> tuple[str, ...]:
    names = (distractor_robots,) if isinstance(distractor_robots, str) else tuple(distractor_robots)
    unknown = [n for n in names if n not in STATIONS]
    if unknown or len(set(names)) != len(names):
        raise ValueError(
            f"distractor_robots={distractor_robots!r}: expected distinct names from {sorted(STATIONS)}"
        )
    return tuple(sorted(names))


class DistractorRobotsMixin:
    """Adds the scripted distractor robots named by the `distractor_robots` kwarg (see module doc)."""

    DEFAULT_DISTRACTORS: tuple[str, ...] = ("left", "right")
    distractor_max_start_delay: int = MAX_START_DELAY
    """Per-task override of `MAX_START_DELAY` (see `distractor_scene` for when to shorten it)."""
    distractor_cube_colour: tuple[float, float, float, float] | None = None
    """Per-task override of every station's cube: RGBA in [0, 1], drawn with a plain material (only a
    base colour, as ManiSkill's `actors.build_cube` draws a task's own cube), so the distractors'
    cubes can be made indistinguishable from the task's. None keeps each station's `cube_colour`."""
    distractor_pads: bool = True
    """Per-task switch for the pads marking where each distractor puts its cube down. Purely visual
    (the script targets the pad positions either way)."""

    def __init__(self, *args, distractor_robots=None, **kwargs):
        self.distractor_names = canonical_distractors(
            self.DEFAULT_DISTRACTORS if distractor_robots is None else distractor_robots
        )
        self.distractors: dict[str, Panda] = {}
        self.distractor_cubes = {}
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ build
    def _load_agent(self, options: dict, *args, **kwargs):
        super()._load_agent(options, *args, **kwargs)
        self.distractors = {}
        for name in self.distractor_names:
            station = STATIONS[name]
            self.distractors[name] = Panda(
                self.scene,
                self._control_freq,
                "pd_ee_delta_pos",
                agent_idx=f"distractor-{name}",
                initial_pose=self._distractor_base_pose(station),
            )

    @staticmethod
    def _distractor_base_pose(station: Station) -> sapien.Pose:
        return sapien.Pose([*station.base, 0.0], euler2quat(0, 0, station.yaw))

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.distractor_cubes = {}
        for name in self.distractor_names:
            station = STATIONS[name]
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[CUBE_HALF_SIZE] * 3)
            if self.distractor_cube_colour is None:
                material = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba(station.cube_colour), roughness=0.5, specular=0.5
                )
            else:
                material = sapien.render.RenderMaterial(base_color=list(self.distractor_cube_colour))
            builder.add_box_visual(half_size=[CUBE_HALF_SIZE] * 3, material=material)
            builder.initial_pose = sapien.Pose([*station.pads[0], CUBE_HALF_SIZE])
            self.distractor_cubes[name] = builder.build(f"{DISTRACTOR_PREFIX}{name}_cube")
            for i, pad in enumerate(station.pads if self.distractor_pads else ()):
                builder = self.scene.create_actor_builder()
                builder.add_box_visual(
                    half_size=[PAD_HALF_SIZE, PAD_HALF_SIZE, PAD_THICKNESS],
                    material=sapien.render.RenderMaterial(
                        base_color=sapien_utils.hex2rgba(station.pad_colour), roughness=0.9
                    ),
                )
                builder.initial_pose = sapien.Pose([*pad, PAD_THICKNESS])
                # visual only and never moves: nothing to save or replay
                self.remove_from_state_dict_registry(builder.build_static(f"{DISTRACTOR_PREFIX}{name}_pad_{i}"))
        self._distractor_phases = _phase_table(self.device)

    # ------------------------------------------------------------------ episode
    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if not self.distractor_names:
            return
        b = len(env_idx)
        if getattr(self, "_distractor_start", None) is None or self._distractor_start.shape[1] != self.num_envs:
            shape = (len(self.distractor_names), self.num_envs)
            self._distractor_start = torch.zeros(shape, dtype=torch.long, device=self.device)
            self._distractor_delay = torch.zeros(shape, dtype=torch.long, device=self.device)
        # a stream of its own per env from the episode seed, so the wrapped task's draws are untouched
        seeds = np.asarray(self._episode_seed)[env_idx.cpu().numpy()]
        rngs = [np.random.RandomState((int(seed) ^ 0xD15C0B07) % 2**32) for seed in seeds]
        # (stations, b, 5): start position in the tour, delay, cube xy jitter, cube yaw
        draws = np.array([[(r.randint(len(STATIONS[name].tour)), r.randint(self.distractor_max_start_delay + 1),
                            *r.uniform(-0.01, 0.01, 2), r.uniform(-0.3, 0.3))
                           for r in rngs] for name in self.distractor_names], dtype=np.float64)
        draws = torch.as_tensor(draws, dtype=torch.float32, device=self.device)
        qpos = torch.as_tensor(REST_QPOS, dtype=torch.float32, device=self.device).expand(b, -1)
        for k, name in enumerate(self.distractor_names):
            station, robot = STATIONS[name], self.distractors[name]
            start = draws[k, :, 0].long()
            self._distractor_start[k, env_idx] = start
            self._distractor_delay[k, env_idx] = draws[k, :, 1].long()
            robot.robot.set_pose(Pose.create_from_pq(
                p=torch.tensor([*station.base, 0.0], device=self.device).expand(b, 3),
                q=torch.tensor(euler2quat(0, 0, station.yaw), dtype=torch.float32, device=self.device).expand(b, 4),
            ))
            robot.reset(qpos)
            tour_pads = torch.tensor([station.pads[i] for i in station.tour], dtype=torch.float32, device=self.device)
            p = torch.zeros(b, 3, device=self.device)
            p[:, :2] = tour_pads[start] + draws[k, :, 2:4]
            p[:, 2] = CUBE_HALF_SIZE
            half = draws[k, :, 4] / 2
            q = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], -1)
            self.distractor_cubes[name].set_pose(Pose.create_from_pq(p=p, q=q))
            self.distractor_cubes[name].set_linear_velocity(torch.zeros(b, 3, device=self.device))
            self.distractor_cubes[name].set_angular_velocity(torch.zeros(b, 3, device=self.device))

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        # as BaseEnv.reset does for the agent: after the poses reach the sim
        for robot in self.distractors.values():
            robot.controller.reset()
        return out

    # ------------------------------------------------------------------ script
    def distractor_actions(self) -> dict[str, torch.Tensor]:
        """Each distractor's (num_envs, 4) `pd_ee_delta_pos` action for the coming step."""
        t = self.elapsed_steps.to(self.device).long().reshape(-1)
        actions = {}
        for k, name in enumerate(self.distractor_names):
            station, robot, cube = STATIONS[name], self.distractors[name], self.distractor_cubes[name]
            local = t - self._distractor_delay[k]
            started = local >= 0
            local = local.clamp(min=0)
            cycle, step = local // CYCLE_STEPS, local % CYCLE_STEPS
            table = {key: v[step] for key, v in self._distractor_phases.items()}
            tour_pads = torch.tensor([station.pads[i] for i in station.tour], dtype=torch.float32, device=self.device)
            place_xy = tour_pads[(self._distractor_start[k] + cycle + 1) % len(station.tour)]
            xy = torch.where(table["place"][:, None], place_xy, cube.pose.p[:, :2])
            tcp = robot.tcp.pose.p
            # don't go down before being over the spot: the first approach of an episode can be
            # ~0.4 m and run late, and closing 3 cm above the cube pushes it off the pad
            raise_by = DESCENT_GUARD * (xy - tcp[:, :2]).norm(dim=-1)
            z = table["z"] + torch.minimum(raise_by, (HOVER_HEIGHT - table["z"]).clamp(min=0.0))
            target = torch.cat([xy, z[:, None]], dim=-1)
            delta = target - tcp
            norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            delta = delta * (table["speed"][:, None] / norm).clamp(max=1.0)
            # the controller's deltas are in the robot's base frame
            delta = (robot.robot.pose.inv() * Pose.create_from_pq(p=tcp + delta)).p - (
                robot.robot.pose.inv() * Pose.create_from_pq(p=tcp)
            ).p
            # pd_ee_delta_pos re-bases its target on the current pose each step and the PD only
            # covers ~40% of a commanded delta per step, so ask for more than the move wanted
            arm = (LAG_GAIN * delta / 0.1).clamp(-1, 1)  # the Panda's pd_ee_delta_pos range is +-0.1 m
            arm = torch.where(started[:, None], arm, torch.zeros_like(arm))
            gripper = torch.where(started, table["gripper"], torch.ones_like(table["gripper"]))
            actions[name] = torch.cat([arm, gripper[:, None]], dim=-1)
        return actions

    def _before_control_step(self):
        super()._before_control_step()
        if not self.distractors:
            return
        for name, action in self.distractor_actions().items():
            self.distractors[name].set_action(action)
        if self.gpu_sim_enabled:
            self.scene.px.gpu_apply_articulation_target_position()
            self.scene.px.gpu_apply_articulation_target_velocity()

    def _before_simulation_step(self):
        super()._before_simulation_step()
        for robot in self.distractors.values():
            robot.before_simulation_step()
