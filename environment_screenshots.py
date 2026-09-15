"""Render a grid of env screenshots: one column per task, and three rows -- the episode's
start state, a goal state sampled for that same episode, and the wrist camera the policy observes.

    python environment_screenshots.py [--out PATH] [--seed N]

Needs a real render backend (SAPIEN/Vulkan), so this has to run where that's available --
inside the project's apptainer container with --nv on a GPU node, not on the login node.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.structs import Pose

from custom_maniskill_tasks import WRIST_CAMERA_UID, make_env

HERE = Path(__file__).resolve().parent

# column title, task id.
TASKS = (
    ("PushCube-v1.1", "PushCube-v1.1"),
    ("LiftPegUpright-v1.1", "LiftPegUpright-v1.1"),
    ("PlaceSphere-v1.1", "PlaceSphere-v1.1"),
    ("PokeCube-v1.1", "PokeCube-v1.1"),
)

ROW_TITLES = ("Start", "Goal", "Visual Obs")
"""The top two rows come from `env.render()`, i.e. each task's own `render_camera`
(`_default_human_render_camera_configs`): the 512x512 three-quarter view ManiSkill's docs and
videos show, which no `camera_view` here touches. The bottom row is the wrist camera a policy
actually observes -- a different camera, which is why the viewpoint changes between rows."""


def _set_pose(actor, base_env, p, q=(1, 0, 0, 0)):
    """Move `actor` to xyz `p` with quaternion `q`, batched for this env's single scene."""
    actor.set_pose(
        Pose.create_from_pq(
            p=torch.tensor([p], dtype=torch.float32, device=base_env.device),
            q=torch.tensor([q], dtype=torch.float32, device=base_env.device),
        )
    )


def _push_cube_goal(base_env, rng):
    """Cube resting anywhere inside the goal disk -- the whole success set of `evaluate`.

    Uniform over the disk (hence the sqrt on the radius; without it samples bunch up in the
    middle), with the target left where the reset put it, so the goal frame is the start frame
    with only the cube moved.
    """
    goal_xy = base_env.goal_region.pose.p[0, :2].cpu().numpy()
    radius = base_env.goal_radius * np.sqrt(rng.random())
    angle = rng.uniform(0, 2 * np.pi)
    _set_pose(
        base_env.obj,
        base_env,
        [
            float(goal_xy[0] + radius * np.cos(angle)),
            float(goal_xy[1] + radius * np.sin(angle)),
            base_env.cube_half_size,
        ],
    )


def _lift_peg_upright_goal(base_env, rng):
    """Peg standing on end, at a position drawn from the region the task initializes in.

    The quaternion is not the obvious `euler2quat(0, pi/2, 0)`. That one does stand the peg up,
    but the task's success check reads the *third* XYZ euler angle, and a pure y-rotation is the
    gimbal-locked case that leaves that angle at 0 -- so `evaluate` scores a genuinely upright peg
    as a failure, and scores the flat `euler2quat(0, 0, pi/2)` peg as a success. This composition
    is upright (the peg's local x axis, its long axis, maps to -z) *and* decomposes to
    |euler_z| = pi/2, so it satisfies the predicate the datasets and success rates are scored
    against. A random yaw would re-enter the same gimbal case, so only the position is sampled.
    """
    xy = rng.uniform(-0.1, 0.1, size=2)
    _set_pose(
        base_env.peg,
        base_env,
        [float(xy[0]), float(xy[1]), base_env.peg_half_length],
        tuple(float(v) for v in euler2quat(np.pi / 2, np.pi / 2, 0)),
    )


def _place_sphere_goal(base_env, rng):
    """Sphere seated in the bin, wherever the reset put the bin.

    Nothing to sample: `evaluate` allows 5mm of slack in each axis, so the success set is a single
    pose up to that noise. `rng` is taken anyway to keep one signature across the three.
    """
    bin_p = base_env.bin.pose.p[0].cpu().numpy()
    _set_pose(
        base_env.obj,
        base_env,
        [
            float(bin_p[0]),
            float(bin_p[1]),
            float(bin_p[2]) + base_env.radius + base_env.block_half_size[0],
        ],
    )


def _poke_cube_goal(base_env, rng):
    """Cube resting anywhere inside the goal disk, drawn uniformly over it like PushCube's.

    Success here is `is_cube_placed & is_robot_static`, so unlike PushCube the arm matters: the
    frame is taken straight after a reset, where the Panda is settled and `is_static(0.2)` holds,
    and nothing is stepped before rendering. The peg is left where the reset put it -- the goal
    is a cube position, and the task says nothing about where the tool ends up.
    """
    goal_xy = base_env.goal_region.pose.p[0, :2].cpu().numpy()
    radius = base_env.goal_radius * np.sqrt(rng.random())
    angle = rng.uniform(0, 2 * np.pi)
    _set_pose(
        base_env.cube,
        base_env,
        [
            float(goal_xy[0] + radius * np.cos(angle)),
            float(goal_xy[1] + radius * np.sin(angle)),
            base_env.cube_half_size,
        ],
    )


GOAL_STATES = {
    "PushCube-v1.1": _push_cube_goal,
    "LiftPegUpright-v1.1": _lift_peg_upright_goal,
    "PlaceSphere-v1.1": _place_sphere_goal,
    "PokeCube-v1.1": _poke_cube_goal,
}
"""How to put each task into a goal state, by task id. Each entry poses the task's own objects
rather than rolling out a policy, and `render_episode` checks the result against the task's
`evaluate()`, so a pose that stops satisfying the predicate fails loudly instead of producing a
plausible-looking but wrong figure. The arm is left in its reset pose: these frames show the goal
configuration of the scene, not a state some policy reached."""


def render_episode(task_id, seed, rng):
    """The (start, goal) render-camera frames of one episode, as (H, W, 3) uint8 arrays.

    Both come from one env, so the goal frame is the start frame with only the task's objects
    moved -- same table, same target/bin placement, same arm pose. `obs_mode="none"` because
    neither frame is read from an observation camera.
    """
    env = make_env(task_id, obs_mode="none", camera_view="default", num_envs=1)
    try:
        env.reset(seed=seed)
        start = env.render()[0].cpu().numpy()

        base_env = env.unwrapped
        GOAL_STATES[task_id](base_env, rng)
        evaluation = base_env.evaluate()
        if not bool(evaluation["success"][0]):
            raise AssertionError(
                f"the goal state built for {task_id} does not satisfy the task's own success "
                f"predicate: { {k: v.tolist() for k, v in evaluation.items()} }"
            )
        return start, env.render()[0].cpu().numpy()
    finally:
        env.close()


def render_wrist_frame(task_id, seed):
    """The wrist-camera observation at the start of the same episode, as an (H, W, 3) uint8."""
    env = make_env(task_id, obs_mode="rgb", camera_view="wrist", num_envs=1)
    try:
        obs, _ = env.reset(seed=seed)
        return obs["sensor_data"][WRIST_CAMERA_UID]["rgb"][0].cpu().numpy()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=HERE / "environment_screenshots.png",
        help="path to write the grid image to",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="reset seed, shared across every task/camera so all nine frames show the same "
        "starting configuration; also seeds the goal-state sampling",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    frames = {}
    # every non-wrist env is built before the first wrist one: registering the wrist camera on the
    # `panda` uid is process-wide and permanent, and a "default" env built afterwards would carry
    # a hand_camera too (see `cameras.camera_view_applied`)
    for col, (_, task_id) in enumerate(TASKS):
        frames[0, col], frames[1, col] = render_episode(task_id, args.seed, rng)
    for col, (_, task_id) in enumerate(TASKS):
        frames[2, col] = render_wrist_frame(task_id, args.seed)

    fig, axes = plt.subplots(
        len(ROW_TITLES), len(TASKS), figsize=(4 * len(TASKS), 4 * len(ROW_TITLES))
    )

    for row, row_title in enumerate(ROW_TITLES):
        for col, (title, _) in enumerate(TASKS):
            ax = axes[row, col]
            ax.imshow(frames[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=24)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=24)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
