"""Peg-insertion distractors: the scripted Pandas of `distractor_robots`, each doing PegInsertionSide.

`PegInsertionDistractorsMixin` replaces each station's cube and pads with a peg and a box with a
hole, built like the stock task's (`_build_box_with_hole`, the same orange/white peg and sand box),
and the pick-and-place script with a peg-insertion one, repeated in a cycle (`PEG_CYCLE`): grasp the
peg where it lies, lift it to the hole's height, line it up in front of the hole, push it in, hold,
pull it out, carry it back, put it down, let go. It is selected by `distractor_task="peg_insertion"`
on the tasks built with it (`peg_insertion.PegInsertionSide-MultiArmDistractors`); the default,
"pick_place", is `DistractorRobotsMixin` unchanged.

Everything `distractor_robots` says about the robots still holds -- not the agent, out of its way
but not collision-filtered, replayable from the state dict plus a seeded reset -- except:

- **Control** is `pd_ee_delta_pos_yaw` (`controllers`), 5-D, so the script can turn the hand to the
  peg's yaw to grasp it and keep the peg lined up with the hole while carrying it.
- **Geometry is fixed**, unlike the agent's peg: half length `PEG_HALF_LENGTH`, half width
  `PEG_RADIUS`, the hole `DISTRACTOR_CLEARANCE` wider (the stock 3 mm would make an open-loop-in-time
  script jam now and then; 6 mm looks the same from the scene camera). Each episode the peg's rest
  pose is jittered (+-1 cm, +-0.15 rad) and each robot idles 0..`distractor_max_start_delay` steps.
- **Layout** (`PEG_STATIONS`): each robot's peg lies 0.31 m in front of it, pointing outwards
  (away from the agent), and its box stands beside it, 0.17 m further forward and 0.24 m further
  out, the hole facing back along the peg: insertion runs outwards, away from the agent's
  workspace. Everything stays within the camera framing `distractor_scene` chose for the pads.
- **Cycle** (`PEG_CYCLE_STEPS`, 110 steps) is longer than a 100-step episode, so an episode sees
  one insertion; whether the robot is still putting the peg back at the end depends on its delay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sapien
import torch

from mani_skill.envs.tasks.tabletop.peg_insertion_side import _build_box_with_hole
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.structs.pose import Pose

from custom_maniskill_tasks.controllers import YAW_CONTROL_MODE, tcp_yaw
from custom_maniskill_tasks.distractor_robots import (
    DISTRACTOR_PREFIX,
    LAG_GAIN,
    STATIONS,
    DistractorRobotsMixin,
)

DISTRACTOR_TASKS = ("pick_place", "peg_insertion")
"""The `distractor_task` kwarg's values."""

PEG_HALF_LENGTH = 0.1
PEG_RADIUS = 0.02
DISTRACTOR_CLEARANCE = 0.006
GRASP_BACK = 0.03
"""The grasp point, this far behind the peg's centre (towards its tail), as the scripted agent grasps."""
PRE_INSERT_GAP = 0.04
"""How far in front of the hole's entrance the peg head lines up before going in."""
HOVER = 0.15


@dataclass(frozen=True)
class PegStation:
    """One distractor's peg and box (world frame). `side` is +1 for left (+y), -1 for right."""

    peg_xy: tuple[float, float]
    peg_yaw: float
    """Yaw of the peg's head direction at rest."""
    box_xy: tuple[float, float]
    box_yaw: float
    """Yaw of the hole's axis, which the peg travels along going in."""


def _peg_station(side: int) -> PegStation:
    y = 0.7 * side
    return PegStation(
        peg_xy=(-0.305, y - 0.04 * side),
        peg_yaw=np.pi / 2 * side,
        box_xy=(-0.135, y + 0.20 * side),
        box_yaw=np.pi / 2 * side,
    )


PEG_STATIONS = {"left": _peg_station(+1), "right": _peg_station(-1)}


@dataclass(frozen=True)
class PegPhase:
    name: str
    steps: int
    gripper: float  # 1 open, -1 closed
    speed: float  # max tcp travel per env step (m)


PEG_CYCLE: tuple[PegPhase, ...] = (
    PegPhase("approach", 10, 1.0, 0.04),
    PegPhase("descend", 9, 1.0, 0.03),
    PegPhase("grasp", 5, -1.0, 0.02),
    PegPhase("lift", 9, -1.0, 0.03),
    PegPhase("align", 16, -1.0, 0.025),
    PegPhase("insert", 16, -1.0, 0.012),
    PegPhase("hold", 4, -1.0, 0.01),
    PegPhase("withdraw", 10, -1.0, 0.02),
    PegPhase("return", 12, -1.0, 0.03),
    PegPhase("lower", 9, -1.0, 0.025),
    PegPhase("release", 4, 1.0, 0.02),
    PegPhase("retreat", 6, 1.0, 0.04),
)
PEG_CYCLE_STEPS = sum(p.steps for p in PEG_CYCLE)
_PHASE_INDEX = {p.name: i for i, p in enumerate(PEG_CYCLE)}


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw / 2
    zero = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], -1)


class PegInsertionDistractorsMixin(DistractorRobotsMixin):
    """`DistractorRobotsMixin` whose robots do peg insertion when `distractor_task="peg_insertion"` (module doc)."""

    def __init__(self, *args, distractor_task: str = "pick_place", **kwargs):
        if distractor_task not in DISTRACTOR_TASKS:
            raise ValueError(f"distractor_task={distractor_task!r}: expected one of {DISTRACTOR_TASKS}")
        self.distractor_task = distractor_task
        self.distractor_pegs, self.distractor_boxes = {}, {}
        super().__init__(*args, **kwargs)

    @property
    def _pegs(self) -> bool:
        return self.distractor_task == "peg_insertion"

    @property
    def distractor_control_mode(self) -> str:
        return YAW_CONTROL_MODE if self._pegs else super().distractor_control_mode

    # ------------------------------------------------------------------ build
    def _load_distractor_props(self):
        if not self._pegs:
            return super()._load_distractor_props()
        self.distractor_cubes = {}
        self.distractor_pegs, self.distractor_boxes = {}, {}
        L, r = PEG_HALF_LENGTH, PEG_RADIUS
        for name in self.distractor_names:
            st = PEG_STATIONS[name]
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[L, r, r])
            for colour, sign in (("#EC7357", 1), ("#EDF6F9", -1)):  # head, tail: the stock peg's
                builder.add_box_visual(
                    sapien.Pose([sign * L / 2, 0, 0]), half_size=[L / 2, r, r],
                    material=sapien.render.RenderMaterial(base_color=sapien_utils.hex2rgba(colour), roughness=0.5, specular=0.5),
                )
            builder.initial_pose = sapien.Pose([*st.peg_xy, r])
            self.distractor_pegs[name] = builder.build(f"{DISTRACTOR_PREFIX}{name}_peg")
            # the stock box: outer half size and depth both the peg's half length, hole centred
            builder = _build_box_with_hole(self.scene, r + DISTRACTOR_CLEARANCE, L, L)
            builder.initial_pose = sapien.Pose([*st.box_xy, L], _yaw_quat(torch.tensor(st.box_yaw)).numpy())
            self.distractor_boxes[name] = builder.build_kinematic(f"{DISTRACTOR_PREFIX}{name}_box_with_hole")

    def _distractor_objects(self) -> dict:
        return self.distractor_pegs if self._pegs else super()._distractor_objects()

    # ------------------------------------------------------------------ episode
    def _initialize_distractors(self, env_idx):
        if not self._pegs:
            return super()._initialize_distractors(env_idx)
        b, dev = len(env_idx), self.device
        if getattr(self, "_distractor_delay", None) is None or self._distractor_delay.shape[1] != self.num_envs:
            shape = (len(self.distractor_names), self.num_envs)
            self._distractor_delay = torch.zeros(shape, dtype=torch.long, device=dev)
            self._distractor_rest = torch.zeros((*shape, 3), dtype=torch.float32, device=dev)  # x, y, yaw
        # a stream of its own per env from the episode seed (as the cube distractors draw theirs)
        seeds = np.asarray(self._episode_seed)[env_idx.cpu().numpy()]
        rngs = [np.random.RandomState((int(seed) ^ 0x9E6D15C0) % 2**32) for seed in seeds]
        draws = np.array([[(r.randint(self.distractor_max_start_delay + 1), *r.uniform(-0.01, 0.01, 2),
                            r.uniform(-0.15, 0.15)) for r in rngs] for _ in self.distractor_names], dtype=np.float64)
        draws = torch.as_tensor(draws, dtype=torch.float32, device=dev)
        self._reset_distractor_robots(env_idx)
        for k, name in enumerate(self.distractor_names):
            st = PEG_STATIONS[name]
            rest = torch.stack([st.peg_xy[0] + draws[k, :, 1], st.peg_xy[1] + draws[k, :, 2],
                                st.peg_yaw + draws[k, :, 3]], -1)
            self._distractor_delay[k, env_idx] = draws[k, :, 0].long()
            self._distractor_rest[k, env_idx] = rest
            p = torch.zeros(b, 3, device=dev)
            p[:, :2], p[:, 2] = rest[:, :2], PEG_RADIUS
            peg = self.distractor_pegs[name]
            peg.set_pose(Pose.create_from_pq(p=p, q=_yaw_quat(rest[:, 2])))
            peg.set_linear_velocity(torch.zeros(b, 3, device=dev))
            peg.set_angular_velocity(torch.zeros(b, 3, device=dev))

    # ------------------------------------------------------------------ script
    def distractor_actions(self) -> dict[str, torch.Tensor]:
        """Each distractor's (num_envs, 5) `pd_ee_delta_pos_yaw` action for the coming step."""
        if not self._pegs:
            return super().distractor_actions()
        dev = self.device
        t = self.elapsed_steps.to(dev).long().reshape(-1)
        if not hasattr(self, "_peg_phase_of_step"):
            self._peg_phase_of_step = torch.tensor([i for i, p in enumerate(PEG_CYCLE) for _ in range(p.steps)], device=dev)
            self._peg_gripper = torch.tensor([p.gripper for p in PEG_CYCLE], dtype=torch.float32, device=dev)
            self._peg_speed = torch.tensor([p.speed for p in PEG_CYCLE], dtype=torch.float32, device=dev)
        L, r = PEG_HALF_LENGTH, PEG_RADIUS
        actions = {}
        for k, name in enumerate(self.distractor_names):
            st, robot = PEG_STATIONS[name], self.distractors[name]
            local = t - self._distractor_delay[k]
            started = local >= 0
            phase = self._peg_phase_of_step[local.clamp(min=0) % PEG_CYCLE_STEPS]
            is_ = lambda *names: torch.stack([phase == _PHASE_INDEX[n] for n in names]).any(0)

            peg = self.distractor_pegs[name].pose
            head_dir = quaternion_to_matrix(peg.q)[..., :, 0]
            head_dir = torch.nn.functional.normalize(head_dir * torch.tensor([1.0, 1.0, 0.0], device=dev), dim=-1)
            head = peg.p + head_dir * L
            peg_yaw = torch.atan2(head_dir[:, 1], head_dir[:, 0])
            tcp = robot.tcp.pose.p
            hand_yaw = tcp_yaw(robot.tcp.pose.q)

            box = self.distractor_boxes[name].pose
            axis = torch.tensor([np.cos(st.box_yaw), np.sin(st.box_yaw), 0.0], dtype=torch.float32, device=dev)
            hole = box.p[:1].expand_as(tcp)  # kinematic and the same in every env
            pre = hole - axis * (L + PRE_INSERT_GAP)
            rest = self._distractor_rest[k]
            rest_dir = torch.stack([torch.cos(rest[:, 2]), torch.sin(rest[:, 2]), torch.zeros_like(rest[:, 2])], -1)
            rest_grasp = torch.cat([rest[:, :2], torch.full_like(rest[:, :1], r)], -1) - rest_dir * GRASP_BACK
            grasp = peg.p - head_dir * GRASP_BACK

            # tcp targets, per phase
            over = lambda xy, z: torch.cat([xy, z.expand(len(xy), 1) if z.dim() < 2 else z], -1)
            target = tcp.clone()
            hover = torch.tensor([[HOVER]], device=dev)
            grasp_z = torch.tensor([[r]], device=dev)
            # come down only once over the spot (as the cube script does)
            guard = lambda xy, z: torch.minimum(z + 2.0 * (xy - tcp[:, :2]).norm(dim=-1, keepdim=True), hover.expand_as(z))
            target = torch.where(is_("approach")[:, None], over(grasp[:, :2], hover), target)
            target = torch.where(is_("descend", "grasp")[:, None],
                                 over(grasp[:, :2], guard(grasp[:, :2], grasp_z.expand(len(tcp), 1))), target)
            # the carried phases steer the peg head, through wherever the peg sits in the hand
            head_goal = torch.where(is_("lift")[:, None], torch.cat([head[:, :2], hole[:, 2:] + 0.02], -1), pre)
            head_goal = torch.where(is_("insert", "hold")[:, None], hole + axis * 0.01, head_goal)
            target = torch.where(is_("lift", "align", "insert", "hold", "withdraw")[:, None], tcp + head_goal - head, target)
            target = torch.where(is_("return")[:, None], over(rest_grasp[:, :2], torch.full_like(tcp[:, :1], 0.12)), target)
            target = torch.where(is_("lower", "release")[:, None],
                                 over(rest_grasp[:, :2], guard(rest_grasp[:, :2], torch.full_like(tcp[:, :1], r + 0.004))), target)
            target = torch.where(is_("retreat")[:, None], over(rest_grasp[:, :2], hover), target)

            # yaw: square the hand to the peg (mod pi) to grasp it, then turn the peg to the hole / its rest yaw
            dyaw = _wrap(2 * (peg_yaw - hand_yaw)) / 2
            dyaw = torch.where(is_("lift", "align", "insert", "hold", "withdraw"), _wrap(st.box_yaw - peg_yaw), dyaw)
            dyaw = torch.where(is_("return", "lower"), _wrap(rest[:, 2] - peg_yaw), dyaw)
            dyaw = torch.where(is_("release", "retreat"), torch.zeros_like(dyaw), dyaw)

            delta = target - tcp
            delta = delta * (self._peg_speed[phase][:, None] / delta.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
            # the controller's deltas are in the robot's base frame; every station's base has yaw 0
            assert STATIONS[name].yaw == 0.0
            arm = (LAG_GAIN * delta / 0.1).clamp(-1, 1)
            turn = (2.5 * dyaw / 0.1).clamp(-1, 1)[:, None]
            gripper = self._peg_gripper[phase]
            arm = torch.where(started[:, None], arm, torch.zeros_like(arm))
            turn = torch.where(started[:, None], turn, torch.zeros_like(turn))
            gripper = torch.where(started, gripper, torch.ones_like(gripper))
            actions[name] = torch.cat([arm, turn, gripper[:, None]], dim=-1)
        return actions

    def distractor_inserted(self) -> dict[str, torch.Tensor]:
        """Whether each distractor's peg head is in its hole, by the stock task's predicate."""
        out = {}
        for name in self.distractor_names:
            peg, box = self.distractor_pegs[name].pose, self.distractor_boxes[name].pose
            head = peg * Pose.create_from_pq(p=torch.tensor([[PEG_HALF_LENGTH, 0.0, 0.0]], device=self.device))
            at = (box.inv() * head).p
            radius = PEG_RADIUS + DISTRACTOR_CLEARANCE
            out[name] = (at[:, 0] >= -0.015) & (at[:, 1].abs() <= radius) & (at[:, 2].abs() <= radius)
        return out
