"""Pick-cube distractors: the scripted Pandas of `distractor_robots`, each doing PickCube for itself.

`PickCubeDistractorsMixin` gives each distractor station a PickCube episode of its own: a cube and a
goal drawn from PickCube-v1's own distributions (Panda config, `pick_cube_cfgs.PICK_CUBE_CONFIGS`),
placed relative to the distractor's base exactly as the task places them relative to the agent's:

- cube: xy uniform in +-`CUBE_SPAWN_HALF_SIZE` (0.1 m) around a centre 0.615 m in front of the base
  (the task's (0, 0) for the agent at (-0.615, 0)), resting on the table, yaw uniform over a full turn;
- goal: xy drawn the same way, independently of the cube; height 0.02 + uniform [0, `MAX_GOAL_HEIGHT`].

With probability `task_alignment_probability` (kwarg, default 0.5), drawn independently per env and
per station, a distractor instead receives the agent's own set-up: the agent's cube offset and yaw
and its goal offset, each relative to the distractor's base. The goal is a point the script carries
the cube to; like the task's, it is never drawn.

The script, per episode: idle 0..`distractor_max_start_delay` steps, then approach over the cube
(turning the hand to the cube's yaw, mod 90 degrees), descend, grasp, carry the cube to the goal and
hold it there to the end of the episode -- the task's success condition. It runs on
`pd_ee_delta_pos_yaw` (`controllers`), since the cube's yaw spans a full turn. As in
`distractor_robots`, it is open-loop in time (the phase comes from the step count) and closed-loop
in space (it reaches for wherever the cube is), and its per-episode constants (delay, alignment, goal)
come from the episode seed in a stream of their own, so the task's own draws are unchanged: the agent's
cube and goal are those of PickCube-v1.1 for the same seed. Cubes are sim bodies (in the state dict);
the goals and delays, like the cube tour's constants, are not, so a seeded reset restores them.

Selected by `distractor_task="pick_cube"`, PickCube-MultiArmDistractors' default since 2026-11-23;
`distractor_task="pick_place"` is the earlier behaviour (each distractor carrying its cube around a
tour of four spots, `DistractorRobotsMixin`), which every run trained before then saw.
"""

from __future__ import annotations

import numpy as np
import torch

from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.structs.pose import Pose

from custom_maniskill_tasks.controllers import YAW_CONTROL_MODE, tcp_yaw
from custom_maniskill_tasks.distractor_robots import (
    CUBE_HALF_SIZE,
    DESCENT_GUARD,
    HOVER_HEIGHT,
    LAG_GAIN,
    STATIONS,
    DistractorRobotsMixin,
)

PICK_CUBE_DISTRACTOR_TASKS = ("pick_cube", "pick_place")
"""The `distractor_task` kwarg's values on the PickCube distractor task; the first is the default."""

CUBE_SPAWN_HALF_SIZE = 0.1
MAX_GOAL_HEIGHT = 0.3
GOAL_THRESH = 0.025
"""PickCube-v1's Panda config (`cube_spawn_half_size`, `max_goal_height`, `goal_thresh`)."""
TASK_CENTRE_AHEAD = 0.615
"""How far in front of its robot's base a PickCube episode is centred: the task's (0, 0) for the agent at (-0.615, 0)."""

# (name, steps, gripper, max tcp travel per env step); the last phase lasts the rest of the episode
PICK_CUBE_SCRIPT = (
    ("approach", 9, 1.0, 0.05),
    ("descend", 7, 1.0, 0.03),
    ("grasp", 4, -1.0, 0.02),
    ("carry", None, -1.0, 0.04),
)


def _cube_yaw(q: torch.Tensor) -> torch.Tensor:
    R = quaternion_to_matrix(q)
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def _wrap_quarter(a: torch.Tensor) -> torch.Tensor:
    """An angle mod 90 degrees, into [-45, 45) degrees: a cube looks the same turned a quarter."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


class PickCubeDistractorsMixin(DistractorRobotsMixin):
    """`DistractorRobotsMixin` whose robots each do PickCube (module doc) unless `distractor_task="pick_place"`."""

    def __init__(self, *args, distractor_task: str = "pick_cube", task_alignment_probability: float = 0.5,
                 distractor_draw: int = 0, **kwargs):
        if distractor_task not in PICK_CUBE_DISTRACTOR_TASKS:
            raise ValueError(f"distractor_task={distractor_task!r}: expected one of {PICK_CUBE_DISTRACTOR_TASKS}")
        p = float(task_alignment_probability)
        if isinstance(task_alignment_probability, bool) or not 0.0 <= p <= 1.0:
            raise ValueError(f"task_alignment_probability={task_alignment_probability!r}: expected a probability in [0, 1]")
        if distractor_task == "pick_cube" and kwargs.get("distractor_tour_start") is not None:
            raise ValueError("distractor_tour_start pins the pick_place tour; the pick_cube distractors have none")
        if isinstance(distractor_draw, bool) or not isinstance(distractor_draw, (int, np.integer)) or distractor_draw < 0:
            raise ValueError(f"distractor_draw={distractor_draw!r}: expected a non-negative int")
        if distractor_task != "pick_cube" and distractor_draw != 0:
            raise ValueError("distractor_draw selects the pick_cube distractors' draws; pick_place takes distractor_tour_start")
        self.distractor_draw = int(distractor_draw)
        """Which of the distractors' per-episode draws a seed gives: 0 is the default stream; any other index
        is an independent one (other delays, alignment, cubes and goals), with the agent's episode and every
        other draw unchanged -- a controlled change of what the distractors do (the distractor probe's
        conditions on the pick_cube scenes)."""
        self.distractor_task = distractor_task
        self.task_alignment_probability = p
        super().__init__(*args, **kwargs)

    @property
    def _picks(self) -> bool:
        return self.distractor_task == "pick_cube"

    @property
    def distractor_control_mode(self) -> str:
        return YAW_CONTROL_MODE if self._picks else super().distractor_control_mode

    # ------------------------------------------------------------------ episode
    def _distractor_task_centre(self, name: str, env_idx) -> torch.Tensor:
        """(len(env_idx), 2) world xy distractor `name`'s PickCube episode is centred on."""
        assert STATIONS[name].yaw == 0.0  # offsets are added in the world frame, which is the base frame then
        centre = self._distractor_base_xy(name, env_idx).clone()
        centre[:, 0] += TASK_CENTRE_AHEAD
        return centre

    def _agent_task_centre(self, env_idx) -> torch.Tensor:
        """(len(env_idx), 2) world xy the agent's own episode is centred on: the table's (0, 0)."""
        return torch.zeros(len(env_idx), 2, device=self.device)

    def _initialize_distractors(self, env_idx):
        if not self._picks:
            return super()._initialize_distractors(env_idx)
        b, dev = len(env_idx), self.device
        if getattr(self, "_distractor_goal", None) is None or self._distractor_goal.shape[1] != self.num_envs:
            shape = (len(self.distractor_names), self.num_envs)
            self._distractor_delay = torch.zeros(shape, dtype=torch.long, device=dev)
            self._distractor_aligned = torch.zeros(shape, dtype=torch.bool, device=dev)
            self._distractor_goal = torch.zeros((*shape, 3), dtype=torch.float32, device=dev)
        # a stream of its own per env from the episode seed, so the task's own draws are untouched
        seeds = np.asarray(self._episode_seed)[env_idx.cpu().numpy()]
        rngs = [np.random.RandomState(((int(seed) ^ 0x91C6C0BE) + 0x9E3779B1 * self.distractor_draw) % 2**32) for seed in seeds]
        # (stations, b, 8): delay, aligned?, cube xy, cube yaw, goal xy, goal lift
        h = CUBE_SPAWN_HALF_SIZE
        draws = np.array([[(r.randint(self.distractor_max_start_delay + 1), r.uniform() < self.task_alignment_probability,
                            *r.uniform(-h, h, 2), r.uniform(0, 2 * np.pi), *r.uniform(-h, h, 2), r.uniform(0, MAX_GOAL_HEIGHT))
                           for r in rngs] for _ in self.distractor_names], dtype=np.float64)
        draws = torch.as_tensor(draws, dtype=torch.float32, device=dev)
        # the agent's own set-up, relative to its task centre
        agent_centre = self._agent_task_centre(env_idx)
        agent_cube, agent_goal = self.cube.pose, self.goal_site.pose.p[env_idx].clone()
        agent_cube_xy, agent_cube_q = agent_cube.p[env_idx, :2] - agent_centre, agent_cube.q[env_idx]
        agent_goal[:, :2] -= agent_centre
        self._reset_distractor_robots(env_idx)
        for k, name in enumerate(self.distractor_names):
            centre = self._distractor_task_centre(name, env_idx)
            aligned = draws[k, :, 1] > 0.5
            half = draws[k, :, 4] / 2
            zero = torch.zeros_like(half)
            own_q = torch.stack([torch.cos(half), zero, zero, torch.sin(half)], -1)
            cube_xy = centre + torch.where(aligned[:, None], agent_cube_xy, draws[k, :, 2:4])
            cube_q = torch.where(aligned[:, None], agent_cube_q, own_q)
            goal = torch.cat([centre + draws[k, :, 5:7], (CUBE_HALF_SIZE + draws[k, :, 7])[:, None]], -1)
            goal = torch.where(aligned[:, None], torch.cat([centre + agent_goal[:, :2], agent_goal[:, 2:]], -1), goal)
            self._distractor_delay[k, env_idx] = draws[k, :, 0].long()
            self._distractor_aligned[k, env_idx] = aligned
            self._distractor_goal[k, env_idx] = goal
            cube = self.distractor_cubes[name]
            p = torch.cat([cube_xy, torch.full((b, 1), CUBE_HALF_SIZE, device=dev)], -1)
            cube.set_pose(Pose.create_from_pq(p=p, q=cube_q))
            cube.set_linear_velocity(torch.zeros(b, 3, device=dev))
            cube.set_angular_velocity(torch.zeros(b, 3, device=dev))

    # ------------------------------------------------------------------ script
    def distractor_actions(self) -> dict[str, torch.Tensor]:
        """Each distractor's (num_envs, 5) `pd_ee_delta_pos_yaw` action for the coming step."""
        if not self._picks:
            return super().distractor_actions()
        dev = self.device
        t = self.elapsed_steps.to(dev).long().reshape(-1)
        bounds = np.cumsum([s for _, s, _, _ in PICK_CUBE_SCRIPT[:-1]])  # phase k starts at bounds[k-1]
        actions = {}
        for k, name in enumerate(self.distractor_names):
            robot, cube = self.distractors[name], self.distractor_cubes[name]
            local = t - self._distractor_delay[k]
            started = local >= 0
            phase = torch.bucketize(local.clamp(min=0), torch.as_tensor(bounds, device=dev), right=True)
            gripper = torch.as_tensor([g for _, _, g, _ in PICK_CUBE_SCRIPT], dtype=torch.float32, device=dev)[phase]
            speed = torch.as_tensor([v for _, _, _, v in PICK_CUBE_SCRIPT], dtype=torch.float32, device=dev)[phase]
            tcp = robot.tcp.pose.p
            cube_p = cube.pose.p
            hover = torch.full_like(tcp[:, :1], HOVER_HEIGHT)
            low = torch.full_like(tcp[:, :1], CUBE_HALF_SIZE)
            # don't come down before being over the cube (as the cube-tour script does)
            guarded = torch.minimum(low + DESCENT_GUARD * (cube_p[:, :2] - tcp[:, :2]).norm(dim=-1, keepdim=True), hover)
            target = torch.where((phase == 0)[:, None], torch.cat([cube_p[:, :2], hover], -1),
                                 torch.cat([cube_p[:, :2], guarded], -1))
            # carrying: move the cube (wherever it sits in the hand) onto the goal
            target = torch.where((phase == 3)[:, None], tcp + self._distractor_goal[k] - cube_p, target)
            delta = target - tcp
            delta = delta * (speed[:, None] / delta.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
            arm = (LAG_GAIN * delta / 0.1).clamp(-1, 1)  # deltas in the base frame = world (base yaw 0)
            dyaw = _wrap_quarter(_cube_yaw(cube.pose.q) - tcp_yaw(robot.tcp.pose.q))
            turn = torch.where(phase <= 1, (2.5 * dyaw / 0.1).clamp(-1, 1), torch.zeros_like(dyaw))[:, None]
            arm = torch.where(started[:, None], arm, torch.zeros_like(arm))
            turn = torch.where(started[:, None], turn, torch.zeros_like(turn))
            gripper = torch.where(started, gripper, torch.ones_like(gripper))
            actions[name] = torch.cat([arm, turn, gripper[:, None]], dim=-1)
        return actions

    def distractor_success(self) -> dict[str, torch.Tensor]:
        """Whether each distractor's cube is within the task's `goal_thresh` of its goal (the placement half of PickCube's success)."""
        return {name: (self.distractor_cubes[name].pose.p - self._distractor_goal[k]).norm(dim=-1) <= GOAL_THRESH
                for k, name in enumerate(self.distractor_names)}
