"""`LiftPegUpright-v2`: stand the peg on its *marked* end, where only the marking's shape says which.

`LiftPegUpright-v1.1` can be solved from colour alone: the peg is a red/blue box on a wood table,
it always starts lying along the world x-axis, and either end may go down. v2 makes success hinge
on a visual feature colour cannot give away:

- **Markings, not colours, tell the ends apart.** The peg is one colour all over. Each end carries
  a marking: the base end one from `BASE_MARKINGS`, the other end a decoy from `TOP_MARKINGS`.
  Both markings are the same colour, so the only difference between the ends is the markings'
  shape -- a band against a double band, a plus against an X.
- **Colours are drawn per env.** The peg's body colour and the markings' colour are drawn afresh
  each time the scene is built (the markings at least 60 degrees of hue away from the body), so no
  colour means anything.
- **Which way the base end faces is drawn every reset.** `yaw="flip"` (the default) turns the peg
  by 0 or pi about the vertical, so the base end lies towards or away from the robot; `"full"`
  turns it uniformly. With "flip" the control problem is v1.1's or its mirror image, which keeps
  the difference from v1.1 perceptual.
- **Success is signed.** The base end must be the one on the table.

The markings are visual-only geometry, 1.5 mm proud of the peg's faces: the collision box, mass and
friction are v1.1's, so the dynamics are identical under every marking and colour.

The marking sets are split like every other perturbation here (`perturbation_classes.SPLITS`):

- `train` draws a base marking and a decoy from the training markings (3 x 3 pairs);
- `evaluation` pairs exactly one held-out marking (`frame` as the base, or `checker` as the decoy)
  with a training one, so a policy that has learned the training markings can still solve it by
  elimination -- what it is tested on is whether its features separate a marking it has never
  seen from the ones it has;
- `full` draws from everything (4 x 4).

`decoy=False` leaves the top end plain, which turns the task into "find the end with a marking on
it": the easier rung of the ladder, for checking the gap grows with the visual demand.

Markings and colours are chosen when the scene is built, since GPU PhysX cannot change what an
actor looks like afterwards (`Actor.hide_visual`). So, as in `PickSingleYCB`, `reconfiguration_freq`
defaults to 1 at `num_envs=1` (new markings every reset, at the cost of a scene rebuild) and to 0
above it (one draw per parallel env, held for the run; the base end's direction is still redrawn
every reset). The draws come from ManiSkill's per-env episode rng, so a seed reproduces them.

What the current episode drew is `env.unwrapped.peg_markings` -- deliberately not in any
observation. In `obs_mode="state"` the peg's pose already says where the base end points (it is
always the peg's local +x end), which is what a privileged-state expert needs; the markings
themselves are only in the pixels.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from mani_skill.envs.tasks.tabletop.lift_peg_upright import LiftPegUprightEnv
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from custom_maniskill_tasks.lighting import LightingMixin
from custom_maniskill_tasks.perturbation_classes import SPLITS, resolve_perturbations
from custom_maniskill_tasks.perturbation_mixin import PerturbationClassesMixin
from custom_maniskill_tasks.table_material import TableMaterialMixin
from custom_maniskill_tasks.tasks import CONTROL_MODE, FullHorizonMixin, _horizon

# --------------------------------------------------------------------------- #
# markings
# --------------------------------------------------------------------------- #
RELIEF = 0.0015
"""How far a marking stands proud of the peg's faces (m): enough that it never z-fights the face,
too little to change the silhouette."""
MARKED_LENGTH = 0.07
"""How far along the peg from each end its marking reaches (m), of the peg's 0.24."""


@dataclass(frozen=True)
class Bar:
    """A rectangle on a side face. `u` is the distance in from the peg's end, `v` across the face
    (0 is the face's centre line), both to the rectangle's centre; `angle` turns it in the face's
    plane (radians, 0 has `half_u` along the peg)."""

    u: float
    v: float
    half_u: float
    half_v: float
    angle: float = 0.0


@dataclass(frozen=True)
class Dot:
    u: float
    v: float
    radius: float


W = LiftPegUprightEnv.peg_half_width
_ACROSS = W + RELIEF
"""Half the width of a bar that wraps the whole face, reaching over the corner so that a band
reads as one ring rather than four strips."""

MARKINGS: dict[str, tuple[Bar | Dot, ...]] = {
    # the same shape is put on each of the peg's four side faces, so it reads the same however the
    # peg is rolled; the end faces stay plain
    "band": (Bar(0.025, 0.0, 0.010, _ACROSS),),
    "double_band": (Bar(0.015, 0.0, 0.005, _ACROSS), Bar(0.040, 0.0, 0.005, _ACROSS)),
    "stripe": (Bar(0.035, 0.0, 0.030, 0.006),),
    "plus": (Bar(0.035, 0.0, 0.020, 0.004), Bar(0.035, 0.0, 0.004, 0.018)),
    "cross": (Bar(0.035, 0.0, 0.024, 0.0035, np.pi / 4), Bar(0.035, 0.0, 0.024, 0.0035, -np.pi / 4)),
    "dots": (Dot(0.020, 0.0, 0.008), Dot(0.050, 0.0, 0.008)),
    "checker": (Bar(0.025, 0.010, 0.010, 0.010), Bar(0.045, -0.010, 0.010, 0.010)),
    "frame": (
        Bar(0.015, 0.0, 0.0025, 0.020), Bar(0.055, 0.0, 0.0025, 0.020),
        Bar(0.035, 0.0175, 0.0175, 0.0025), Bar(0.035, -0.0175, 0.0175, 0.0025),
    ),
}
"""Each marking as the primitives on one side face. They are paired across the two sets so that
telling base from decoy needs shape, not presence: a band against a double band, a plus against
the same plus turned 45 degrees, dots against a stripe down the same centre line."""

BASE_MARKINGS = {"train": ("band", "plus", "dots"), "evaluation": ("frame",)}
TOP_MARKINGS = {"train": ("double_band", "stripe", "cross"), "evaluation": ("checker",)}
"""Which markings may go on which end, and which of them training sees."""

NO_MARKING = "none"
YAWS = ("flip", "full")
COLOURS = ("random", "fixed")
FIXED_BODY_COLOUR = (0.55, 0.55, 0.55)
FIXED_MARKING_COLOUR = (0.08, 0.08, 0.08)


def marking_pairs(split: str, decoy: bool = True) -> tuple[tuple[str, str], ...]:
    """Every (base, top) marking pair `split` can draw, each equally likely."""
    if split not in SPLITS:
        raise ValueError(f"markings={split!r}: expected one of {', '.join(SPLITS)}.")
    base_all = BASE_MARKINGS["train"] + BASE_MARKINGS["evaluation"]
    top_all = TOP_MARKINGS["train"] + TOP_MARKINGS["evaluation"]
    if not decoy:
        base = base_all if split == "full" else BASE_MARKINGS[split]
        return tuple((b, NO_MARKING) for b in base)
    if split == "train":
        return tuple((b, t) for b in BASE_MARKINGS["train"] for t in TOP_MARKINGS["train"])
    if split == "full":
        return tuple((b, t) for b in base_all for t in top_all)
    # evaluation: exactly one of the two is held out, so it is solvable by elimination
    return tuple(
        (b, t) for b in base_all for t in top_all
        if (b in BASE_MARKINGS["evaluation"]) != (t in TOP_MARKINGS["evaluation"])
    )


def draw_colours(rng: np.random.RandomState) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """A body colour and a marking colour, sRGB in [0, 1]; the marking's hue is 60 to 300 degrees
    from the body's, so it always shows."""
    hue = rng.uniform()
    body = colorsys.hsv_to_rgb(hue, rng.uniform(0.45, 0.9), rng.uniform(0.35, 0.85))
    mark_hue = (hue + rng.uniform(1 / 6, 5 / 6)) % 1.0
    mark = colorsys.hsv_to_rgb(mark_hue, rng.uniform(0.45, 0.9), rng.uniform(0.3, 0.95))
    return tuple(float(c) for c in body), tuple(float(c) for c in mark)


def _quat_about(axis: str, angle: float) -> list[float]:
    """wxyz quaternion for a turn of `angle` about the x, y or z axis."""
    return euler2quat(*(angle if a == axis else 0.0 for a in "xyz"))


def _add_marking(builder, name: str, end: int, material) -> None:
    """Add `name`'s primitives to the end of the peg whose local x has sign `end`, on each of the
    four side faces. Placed on the +z face, rotated about the peg's axis onto the other three, and
    turned about z by pi for the -x end, so a marking looks the same at either end."""
    length = LiftPegUprightEnv.peg_half_length
    flip = sapien.Pose(q=_quat_about("z", 0.0 if end > 0 else np.pi))
    for primitive in MARKINGS[name]:
        local = sapien.Pose(p=[length - primitive.u, primitive.v, W + RELIEF / 2])
        for k in range(4):
            face = sapien.Pose(q=_quat_about("x", k * np.pi / 2))
            if isinstance(primitive, Bar):
                pose = flip * face * local * sapien.Pose(q=_quat_about("z", primitive.angle))
                builder.add_box_visual(
                    pose=pose, half_size=[primitive.half_u, primitive.half_v, RELIEF / 2],
                    material=material,
                )
            else:  # a cylinder's axis is its local x: turned onto the face's normal
                pose = flip * face * local * sapien.Pose(q=_quat_about("y", np.pi / 2))
                builder.add_cylinder_visual(
                    pose=pose, radius=primitive.radius, half_length=RELIEF / 2, material=material,
                )


# --------------------------------------------------------------------------- #
# the task
# --------------------------------------------------------------------------- #
UPRIGHT_TOLERANCE = 0.08
"""Radians the peg's axis may lean from vertical: the stock task's tolerance."""


class _MarkedPegScene(LiftPegUprightEnv):
    """LiftPegUprightEnv with the peg built per env, marked as `self.marking_pairs` and
    `self.colours` say. Sits at the bottom of the MRO, in place of the stock scene, so the
    appearance mixins above it edit this peg and this table like any other task's."""

    def _load_scene(self, options: dict):
        # the stock scene (its `_load_scene`, transcribed) with the two-colour peg replaced by one
        # built per env
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        pegs, self.peg_markings = [], []
        for i in range(self.num_envs):
            rng = self._batched_episode_rng[i]
            base, top = self.marking_pairs[int(rng.randint(len(self.marking_pairs)))]
            body, mark = (
                draw_colours(rng) if self.colours == "random"
                else (FIXED_BODY_COLOUR, FIXED_MARKING_COLOUR)
            )
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[self.peg_half_length, self.peg_half_width, self.peg_half_width])
            builder.add_box_visual(
                half_size=[self.peg_half_length, self.peg_half_width, self.peg_half_width],
                material=sapien.render.RenderMaterial(base_color=[*body, 1.0]),
            )
            mark_material = sapien.render.RenderMaterial(base_color=[*mark, 1.0])
            _add_marking(builder, base, +1, mark_material)
            if top != NO_MARKING:
                _add_marking(builder, top, -1, mark_material)
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
            builder.set_scene_idxs([i])
            pegs.append(builder.build(name=f"peg-{i}"))
            self.remove_from_state_dict_registry(pegs[-1])
            self.peg_markings.append(
                {"base": base, "top": top, "body_colour": body, "marking_colour": mark}
            )
        # named "peg" like v1.1's, so state dicts and replays line up
        self.peg = Actor.merge(pegs, name="peg")
        self.add_to_state_dict_registry(self.peg)


@register_env(
    "LiftPegUpright-v2",
    max_episode_steps=_horizon("LiftPegUpright-v1"),
    control_mode=CONTROL_MODE,
)
class LiftPegUprightMarkedEnv(
    FullHorizonMixin, PerturbationClassesMixin, LightingMixin, TableMaterialMixin, _MarkedPegScene
):
    """LiftPegUpright-v1.1 where the peg must stand on its marked end (see the module docstring).

    Kwargs, on top of v1.1's:
        markings: "train" | "evaluation" | "full", which marking pairs are drawn (`marking_pairs`).
        decoy: whether the top end carries a marking too (True) or is plain (False).
        yaw: "flip" (0 or pi, so the peg lies along the x-axis as in v1.1) | "full" (uniform).
        colours: "random" (body and marking colours drawn per env) | "fixed" (grey and black).
    """

    object_of_interest = "peg"

    def __init__(
        self,
        *args,
        markings: str = "train",
        decoy: bool = True,
        yaw: str = "flip",
        colours: str = "random",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        **kwargs,
    ):
        self.marking_pairs = marking_pairs(markings, decoy)  # validates `markings`
        if yaw not in YAWS:
            raise ValueError(f"yaw={yaw!r}: expected one of {', '.join(YAWS)}.")
        if colours not in COLOURS:
            raise ValueError(f"colours={colours!r}: expected one of {', '.join(COLOURS)}.")
        if not isinstance(decoy, bool):
            raise ValueError(f"decoy={decoy!r}: expected True or False.")
        self.marking_split, self.decoy, self.yaw, self.colours = markings, decoy, yaw, colours
        self.peg_markings: list[dict[str, Any]] = []
        if reconfiguration_freq is None:
            randomized = resolve_perturbations(kwargs.get("perturbations")).is_randomized
            reconfiguration_freq = 1 if (num_envs == 1 or randomized) else 0
        super().__init__(
            *args, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, **kwargs
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # v1.1's placement first, so a seed puts the peg at the same xy as in v1.1
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            if self.yaw == "flip":
                yaw = torch.randint(0, 2, (b,)).float() * np.pi
            else:
                yaw = torch.rand(b) * 2 * np.pi
            half = yaw / 2
            turn = torch.stack([torch.cos(half), torch.zeros(b), torch.zeros(b), torch.sin(half)], dim=1)
            lying = torch.tensor(euler2quat(np.pi / 2, 0, 0), dtype=torch.float32).expand(b, 4)
            q = rotation_conversions.quaternion_multiply(turn, lying)
            p = self.peg.pose.p[env_idx]
            self.peg.set_pose(Pose.create_from_pq(p=p, q=q))

    def base_axis_z(self) -> torch.Tensor:
        """The world z-component of the unit vector from the peg's centre towards its base end:
        -1 when it stands on its base, +1 on its decoy end, 0 lying flat."""
        qmats = rotation_conversions.quaternion_to_matrix(self.peg.pose.q)
        return qmats[:, 2, 0]

    def evaluate(self):
        on_base = self.base_axis_z() < -np.cos(UPRIGHT_TOLERANCE)
        close_to_table = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length) < 0.005
        return {"success": on_base & close_to_table}

    def compute_dense_reward(self, obs: Any, action, info: dict):
        """v1.1's reward with the orientation term signed: (1 - base_axis_z) / 2, which is 1 on
        the base, 0.5 lying flat and 0 on the wrong end (v1.1 gives 1 for either end)."""
        reward = (1 - self.base_axis_z()) / 2
        z_dist = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length)
        reward = reward + 1 - torch.tanh(5 * z_dist)
        to_grip_dist = torch.linalg.norm(self.peg.pose.p - self.agent.tcp.pose.p, axis=1)
        reaching = 1 - torch.tanh(5 * to_grip_dist)
        reaching[self.agent.is_grasping(self.peg)] = 1
        reward = reward + reaching / 5
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3.0


MARKED_PEG_TASKS = {"LiftPegUpright-v2": "LiftPegUpright-v1.1"}
"""Each task here and the `-v1.1` task it changes."""
