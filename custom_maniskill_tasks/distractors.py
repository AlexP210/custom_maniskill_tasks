"""Distractors: YCB objects scattered on the table, in the wrist camera's view.

`distractors=N` on a task puts N of them out, from a small pool of compact household objects
(`YCB_POOL`). They are dynamic -- the robot can knock them -- but nothing reads them: observations,
reward and success are the task's own, and a state observation is unchanged too, since tasks put
only their own objects into `_get_obs_extra`.

Where they go is random, drawn each reset from the episode rng (seeded by the reset seed, so an
episode is reproducible) after the task has made its own draws from the torch rng, which are
therefore the same as without them. Each lands wholly inside the wrist camera's view -- at the
nominal pose, or at the shifted one if the env has a `viewpoint` -- and clear of the real objects
(`distractor_avoid`), of the ground the task needs free (also `distractor_avoid`), and of each
other. Which objects they are is fixed by `distractors`: the first N of the pool, so a scene is the
same objects every time and only their places and yaws change. That keeps the scene a build-time
property, without reconfiguring on every reset.

Needs the YCB assets: `python -m mani_skill.utils.download_asset ycb`.
"""

from __future__ import annotations

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.structs.pose import Pose

from custom_maniskill_tasks.viewpoint import (
    DEFAULT_VIEWPOINT_CONFIG,
    in_wrist_view,
    visible_table_bounds,
)

YCB_POOL = (
    "005_tomato_soup_can",
    "013_apple",
    "008_pudding_box",
    "010_potted_meat_can",
    "077_rubiks_cube",
    "025_mug",
    "017_orange",
    "006_mustard_bottle",
)
"""The objects `distractors=N` draws its first N from: compact (none wider than 11 cm), each a
different shape and colour, and small enough for the gripper to close around."""
MAX_DISTRACTORS = len(YCB_POOL)

SEARCH_REGION = ((-0.30, 0.55), (-0.45, 0.50))
"""The table area, ((x_min, x_max), (y_min, y_max)), that candidate places are drawn from when no
part of the table is in the wrist view at all (see `visible_table_bounds`, which narrows it to the
part that is)."""
DISTRACTOR_MARGIN = 0.04
"""The clear gap kept between any two footprints, the real objects' included: room for the
gripper's fingers."""
DISTRACTOR_CANDIDATES = 2048
"""Places drawn for each object, of which the first usable one is taken (the roomiest if none is).
A fixed number, so every parallel env consumes the same amount of its rng."""
VISIBILITY_INFLATION = 0.02
"""How much (m) an object's footprint is grown before the visibility test: the robot's reset noise
moves the camera by about this much."""

_ASSET_INFO = ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"


def ycb_footprints() -> dict[str, dict]:
    """Per pool model: the radius of the circle round its footprint, its height, and the z its
    origin sits at when it rests on the table."""
    if not _ASSET_INFO.exists():
        raise FileNotFoundError(
            f"distractors need the YCB assets, which are not at {_ASSET_INFO.parent}. Fetch them "
            "with: python -m mani_skill.utils.download_asset ycb"
        )
    info = load_json(_ASSET_INFO)
    out = {}
    for model_id in YCB_POOL:
        meta = info[model_id]
        scale = meta.get("scales", [1.0])[0]
        low, high = np.array(meta["bbox"]["min"]) * scale, np.array(meta["bbox"]["max"]) * scale
        extent = high - low
        out[model_id] = dict(
            radius=float(0.5 * np.hypot(extent[0], extent[1])),
            height=float(extent[2]),
            rest_z=float(-low[2]),
        )
    return out


class DistractorsMixin:
    """Gives a task the `distractors` kwarg, an int: how many YCB objects to scatter.

    A task class says what the objects must stay off with `distractor_avoid`: a tuple of
    `(actor attribute, (x, y) offset, (x, y) half extent)` boxes, each centred on that actor at the
    start of the episode -- the task's own objects, and any strip it needs kept free, like the path
    PushCube pushes its cube along. A class with none has nothing to keep clear of.

    Mixed in after `LightingMixin`, so the objects are built inside its `super()._load_scene` and
    count as task objects to an `object-hue-<degrees>` shift, as the real ones do.
    """

    distractor_avoid: tuple = ()

    def __init__(self, *args, distractors: int = 0, **kwargs):
        self._num_distractors = canonical_distractors(distractors)
        super().__init__(*args, **kwargs)

    @property
    def distractors(self) -> dict:
        """The distractor actors in this scene, by name; empty when built without them."""
        return self._distractor_actors

    @property
    def distractor_models(self) -> tuple[str, ...]:
        """The YCB model of each distractor, in order."""
        return YCB_POOL[: self._num_distractors]

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._distractor_actors = {}
        if not self._num_distractors:
            return
        self._distractor_footprints = ycb_footprints()
        for i, model_id in enumerate(self.distractor_models):
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
            builder.initial_pose = sapien.Pose(p=[0.0, 0.3 * (i + 1), 0.3])
            self._distractor_actors[f"distractor_{i}"] = builder.build(name=f"distractor_{i}")

    def _draw_distractor_places(self, env_idx: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """((envs, n, 2) world xy, (envs, n) yaw): for each object the first of its
        `DISTRACTOR_CANDIDATES` random places that is in the wrist view and keeps
        `DISTRACTOR_MARGIN` clear of everything else, or the roomiest if none is."""
        n = self._num_distractors
        radii = np.array([self._distractor_footprints[m]["radius"] for m in self.distractor_models])
        heights = np.array([self._distractor_footprints[m]["height"] for m in self.distractor_models])
        rng = self._batched_episode_rng[env_idx]
        viewpoint = getattr(self, "_viewpoint", DEFAULT_VIEWPOINT_CONFIG)
        (x_lo, x_hi), (y_lo, y_hi) = visible_table_bounds(viewpoint) or SEARCH_REGION
        low, high = np.array([x_lo, y_lo]), np.array([x_hi, y_hi])
        candidates = low + rng.uniform(size=(n, DISTRACTOR_CANDIDATES, 2)) * (high - low)
        yaw = rng.uniform(0.0, 2 * np.pi, size=(n,))
        n_envs = len(env_idx)
        candidates = candidates.reshape(n_envs, n, DISTRACTOR_CANDIDATES, 2)
        yaw = yaw.reshape(n_envs, n)

        # (centre, half extent) of everything to stay clear of, in the world
        device_idx = env_idx.to(self.device)
        boxes = []
        for attr, offset, half in self.distractor_avoid:
            centre = getattr(self, attr).pose.p[device_idx, :2].cpu().numpy() + np.array(offset)
            boxes.append((centre, np.array(half)))

        placed = np.empty((n_envs, n, 2))
        for i in range(n):
            reach = radii[i] + VISIBILITY_INFLATION
            xy = candidates[:, i]  # (envs, candidates, 2)
            visible = np.ones(xy.shape[:2], dtype=bool)
            # the footprint's ring on the table, and the top of the object over its middle
            for dx, dy, z in (
                (0, 0, heights[i]), (0, 0, 0.0),
                (reach, 0, 0.0), (-reach, 0, 0.0), (0, reach, 0.0), (0, -reach, 0.0),
            ):
                point = np.concatenate(
                    [xy + np.array([dx, dy]), np.full(xy.shape[:2] + (1,), z)], axis=-1
                )
                visible &= in_wrist_view(point, viewpoint)
            # a candidate's room: its smallest gap to anything, where the gap between two
            # footprints is their larger separation along either axis (positive = not overlapping)
            room = np.full(xy.shape[:2], np.inf)
            half = np.array([radii[i], radii[i]])
            others = [(c[:, None], h) for c, h in boxes]
            others += [(placed[:, j][:, None], np.array([radii[j], radii[j]])) for j in range(i)]
            for centre, other_half in others:
                room = np.minimum(room, (np.abs(xy - centre) - (half + other_half)).max(axis=-1))
            usable = visible & (room > DISTRACTOR_MARGIN)
            # else (crowded or off-view draw, rare) the roomiest of the visible, else of all
            fallback_pool = np.where(visible.any(axis=-1, keepdims=True), visible, True)
            fallback = np.where(fallback_pool, room, -np.inf).argmax(axis=-1)
            pick = np.where(usable.any(axis=-1), usable.argmax(axis=-1), fallback)
            placed[:, i] = xy[np.arange(n_envs), pick]
        return placed, yaw

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if not self._distractor_actors:
            return
        if self.gpu_sim_enabled:
            # On the GPU simulator a pose the task has just set cannot be read back until the
            # buffers are applied and fetched, which `BaseEnv.reset` does only after this returns;
            # without it the places below would be worked out from where the objects were before
            # the reset. The same three calls `reset` makes, so they are safe to make early.
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        xy, yaw = self._draw_distractor_places(env_idx)
        n_envs = len(env_idx)
        zeros = torch.zeros((n_envs, 3), device=self.device)
        for i, (actor, model_id) in enumerate(zip(self._distractor_actors.values(), self.distractor_models)):
            z = np.full((n_envs, 1), self._distractor_footprints[model_id]["rest_z"])
            p = torch.tensor(np.concatenate([xy[:, i], z], axis=-1), device=self.device, dtype=torch.float32)
            half_yaw = torch.tensor(yaw[:, i] / 2, device=self.device, dtype=torch.float32)
            q = torch.stack(
                [torch.cos(half_yaw), torch.zeros_like(half_yaw), torch.zeros_like(half_yaw), torch.sin(half_yaw)],
                dim=-1,
            )
            actor.set_pose(Pose.create_from_pq(p=p, q=q))
            # a knocked object would otherwise still be moving after the reset
            actor.set_linear_velocity(zeros)
            actor.set_angular_velocity(zeros)


def canonical_distractors(distractors) -> int:
    """`distractors` as a count in 0 to `MAX_DISTRACTORS`, rejecting anything else early.

    A bool is refused rather than read as 0 or 1: `distractors` used to be a flag for copies of the
    task's object, and `distractors=True` quietly meaning "one YCB object" would be a very different
    experiment run under an old config's name.
    """
    if isinstance(distractors, bool) or not isinstance(distractors, (int, np.integer)):
        raise TypeError(
            f"distractors is a count of YCB objects (0 to {MAX_DISTRACTORS}), not "
            f"{distractors!r}: it used to be a flag for copies of the task's own object."
        )
    if not 0 <= distractors <= MAX_DISTRACTORS:
        raise ValueError(f"distractors={distractors} is outside 0 to {MAX_DISTRACTORS}.")
    return int(distractors)


def supports_distractors(task_name: str) -> bool:
    """Whether `task_name`'s registered class takes a `distractors` kwarg."""
    from mani_skill.utils.registration import REGISTERED_ENVS

    spec = REGISTERED_ENVS.get(task_name)
    return spec is not None and issubclass(spec.cls, DistractorsMixin)
