"""PickCube distractors with the three robots' positions shuffled every episode.

`ShuffledPositionsMixin` is `PickCubeDistractorsMixin` (each distractor doing PickCube for itself)
with one change: the scene still has the same three robot *slots* (`SLOT_XY`: the agent's usual
base at the table centre, and the two distractor stations 0.7 m to either side, all facing +x), but
which robot stands in which slot is a uniformly drawn permutation each episode, per env.

- **The agent moves with its slot.** Its base, cube and goal are translated together by the slot's
  offset from the table centre, so relative to its own base the agent's episode is exactly
  PickCube-v1.1's for the same seed. Reward and success are PickCube's own, which only compare
  world-frame positions (tcp, cube, goal) with each other, so they are unchanged by the translation.
  The observations are not: `tcp_pose` and `goal_pos` are world-frame, as ManiSkill reports them,
  so they say which slot the agent is in. `qpos` / `qvel` are joint-space and do not.
- **A distractor's set-up moves with it.** Each distractor's PickCube episode (cube, goal, and the
  agent's set-up when `task_alignment_probability` copies it) is placed relative to whatever slot
  it stands in this episode, and its script is world-frame, so it works from any slot.
- **Colour, not position, tells the robots apart.** The cubes are plain RGB, owned by the robot
  and moving with it: the agent's red (PickCube's own), the `left` robot's blue, the `right`
  robot's green (`DISTRACTOR_CUBE_COLOURS`). ("left"/"right" now name the robot, not where it stands.)

The permutation comes from the episode seed in a stream of its own, so the agent's set-up (in its
base frame) and every distractor draw are those of PickCube-MultiArmDistractors for the same seed;
only where things stand differs. Like the distractors' other per-episode constants, it is not in
the state dict, though the poses it produces are: `set_state_dict` restores where everything is,
a seeded `reset` restores the permutation. `robot_slots` reports it.

Only for `distractor_task="pick_cube"`: the `pick_place` cube tour runs between pads fixed to
the stations.
"""

from __future__ import annotations

import numpy as np
import torch

from mani_skill.utils.structs.pose import Pose

from custom_maniskill_tasks.distractor_robots import STATIONS
from custom_maniskill_tasks.pick_cube_distractor_robots import PickCubeDistractorsMixin

AGENT_BASE_XY = (-0.615, 0.0)
"""Where the table scene stands the agent (`TableSceneBuilder.initialize`), facing +x."""
SLOT_NAMES = ("centre", "left", "right")
SLOT_XY = torch.tensor([AGENT_BASE_XY, STATIONS["left"].base, STATIONS["right"].base], dtype=torch.float32)
"""The three base positions robots are shuffled over, indexed as `SLOT_NAMES`."""
assert all(STATIONS[n].yaw == 0.0 for n in ("left", "right")), "slots must share the agent's heading"

DISTRACTOR_CUBE_COLOURS = {"left": (0.0, 0.0, 1.0, 1.0), "right": (0.0, 1.0, 0.0, 1.0)}
"""Plain blue and green, as plain as the agent's red (1, 0, 0, 1)."""


class ShuffledPositionsMixin(PickCubeDistractorsMixin):
    """`PickCubeDistractorsMixin` with the robots' positions permuted every episode (module doc)."""

    def __init__(self, *args, **kwargs):
        if kwargs.get("distractor_task", "pick_cube") != "pick_cube":
            raise ValueError("the shuffled-position task needs distractor_task='pick_cube': the pick_place tour is tied to its station's pads")
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if not self.distractor_names:  # no `_initialize_distractors` call to shuffle the agent in
            self._shuffle(env_idx)

    def _initialize_distractors(self, env_idx):
        self._shuffle(env_idx)
        super()._initialize_distractors(env_idx)

    def _shuffle(self, env_idx):
        """Draw this episode's permutation and move the agent's set-up onto its slot."""
        b, dev = len(env_idx), self.device
        n_robots = 1 + len(self.distractor_names)
        if getattr(self, "_robot_slot", None) is None or self._robot_slot.shape[0] != self.num_envs:
            self._robot_slot = torch.zeros((self.num_envs, n_robots), dtype=torch.long, device=dev)
        # a stream of its own per env from the episode seed, so no other draw changes
        seeds = np.asarray(self._episode_seed)[env_idx.cpu().numpy()]
        perms = np.stack([np.random.RandomState((int(seed) ^ 0x5B0FF1ED) % 2**32).permutation(len(SLOT_NAMES))[:n_robots]
                          for seed in seeds])
        self._robot_slot[env_idx] = torch.as_tensor(perms, dtype=torch.long, device=dev)

        # the agent, its cube and its goal, translated together onto the agent's slot
        offset = self._agent_task_centre(env_idx)
        offset3 = torch.cat([offset, torch.zeros(b, 1, device=dev)], -1)
        robot = self.agent.robot
        robot.set_pose(Pose.create_from_pq(p=robot.pose.p[env_idx] + offset3, q=robot.pose.q[env_idx]))
        cube = self.cube.pose
        self.cube.set_pose(Pose.create_from_pq(p=cube.p[env_idx] + offset3, q=cube.q[env_idx]))
        goal = self.goal_site.pose
        self.goal_site.set_pose(Pose.create_from_pq(p=goal.p[env_idx] + offset3, q=goal.q[env_idx]))

    def _slot_xy(self, robot: int, env_idx) -> torch.Tensor:
        return SLOT_XY.to(self.device)[self._robot_slot[env_idx, robot]]

    def _distractor_base_xy(self, name: str, env_idx) -> torch.Tensor:
        return self._slot_xy(1 + self.distractor_names.index(name), env_idx)

    def _agent_task_centre(self, env_idx) -> torch.Tensor:
        return self._slot_xy(0, env_idx) - torch.tensor(AGENT_BASE_XY, device=self.device)

    def robot_slots(self) -> dict[str, torch.Tensor]:
        """Which slot (index into `SLOT_NAMES`) each robot stands in this episode, (num_envs,) per robot:
        `agent`, then each distractor by name."""
        names = ("agent", *self.distractor_names)
        return {name: self._robot_slot[:, k] for k, name in enumerate(names)}
