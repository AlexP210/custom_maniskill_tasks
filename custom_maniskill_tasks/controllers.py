"""`pd_ee_delta_pos_yaw`: a 5-D end-effector control mode for the Panda (dx, dy, dz, dyaw, gripper).

ManiSkill's Panda offers `pd_ee_delta_pos` (4-D: the hand can move but never turn) and
`pd_ee_delta_pose` (7-D: any rotation, so the hand can tilt off vertical, which no tabletop task here
needs and a policy then has to learn to avoid). This mode sits between them: the hand is held
pointing straight down, and only its yaw about the vertical is controlled, which is the one
rotation a top-down grasp needs to match an object's orientation (e.g. PegInsertionSide's box,
turned up to 22.5 degrees).

Each step the target is the current TCP pose moved by (dx, dy, dz) in the robot's root frame and
turned by dyaw about the vertical, with roll and pitch reset to exactly straight down. Snapping
roll and pitch every step (rather than adding a zero delta to them, as `pd_ee_delta_pose` would)
means contact torques can't tilt the hand off vertical a bit at a time. Bounds are
`pd_ee_delta_pose`'s: 0.1 m and 0.1 rad per step at an action of 1.

Registered by adding it to `Panda._controller_configs` on import (`install`): additive, every stock
mode is unchanged, and it reaches every Panda subclass (`panda_wristcam`, `cameras.PandaHandCam`,
the distractor robots, which never use it).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController, PDEEPoseControllerConfig
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
from mani_skill.utils.structs import Pose

YAW_CONTROL_MODE = "pd_ee_delta_pos_yaw"


def tcp_yaw(q: torch.Tensor) -> torch.Tensor:
    """Yaw (rad) of a downward-pointing TCP orientation: the angle of its x axis in the root's xy plane."""
    R = quaternion_to_matrix(q)
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def downward_quat(yaw: torch.Tensor) -> torch.Tensor:
    """The TCP orientation pointing straight down (z = -world z) with its x axis at `yaw`.
    At yaw 0 this is the Panda's rest orientation, (w, x, y, z) = (0, 1, 0, 0)."""
    c, s, zero = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)
    x = torch.stack([c, s, zero], -1)
    y = torch.stack([s, -c, zero], -1)
    z = torch.stack([zero, zero, -torch.ones_like(yaw)], -1)
    return matrix_to_quaternion(torch.stack([x, y, z], -1))


class PDEEPosYawController(PDEEPoseController):
    """Translation plus yaw about the vertical, the hand held straight down (module doc)."""

    config: "PDEEPosYawControllerConfig"

    def _initialize_action_space(self):
        low = np.float32(np.r_[np.broadcast_to(self.config.pos_lower, 3), self.config.rot_lower])
        high = np.float32(np.r_[np.broadcast_to(self.config.pos_upper, 3), self.config.rot_upper])
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def _clip_and_scale_action(self, action):
        action = torch.clamp(action, -1, 1)
        low = torch.as_tensor(self.action_space_low, device=action.device)
        high = torch.as_tensor(self.action_space_high, device=action.device)
        return low + (action + 1) / 2 * (high - low)

    def compute_target_pose(self, prev_ee_pose_at_base: Pose, action):
        p = prev_ee_pose_at_base.p + action[:, :3]
        q = downward_quat(tcp_yaw(prev_ee_pose_at_base.q) + action[:, 3])
        return Pose.create_from_pq(p, q)

    def set_action(self, action):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos
        current = self.ee_pose_at_base
        # always an absolute target (the stock GPU path feeds the raw action to IK as a delta, which
        # would skip the roll/pitch snap); compute_ik turns it into a delta itself on the GPU
        self._target_pose = self.compute_target_pose(self._target_pose if self.config.use_target else current, action)
        self._target_qpos = self.kinematics.compute_ik(
            pose=self._target_pose,
            q0=self.articulation.get_qpos(),
            is_delta_pose=False,
            current_pose=current,
            solver_config=self.config.delta_solver_config,
        )
        if self._target_qpos is None:
            self._target_qpos = self._start_qpos
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)


@dataclass
class PDEEPosYawControllerConfig(PDEEPoseControllerConfig):
    controller_cls = PDEEPosYawController


def install() -> None:
    """Add `pd_ee_delta_pos_yaw` to the Panda's control modes (idempotent)."""
    stock = Panda._controller_configs
    if getattr(stock.fget, "_adds_yaw_mode", False):
        return

    def _controller_configs(self):
        configs = stock.fget(self)
        pose = configs["pd_ee_delta_pose"]
        arm = PDEEPosYawControllerConfig(**{f.name: getattr(pose["arm"], f.name) for f in fields(pose["arm"]) if f.init})
        configs[YAW_CONTROL_MODE] = dict(arm=arm, gripper=pose["gripper"])
        return configs

    _controller_configs._adds_yaw_mode = True
    Panda._controller_configs = property(_controller_configs)


install()
