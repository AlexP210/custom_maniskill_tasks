"""Render a 2x3 grid of env screenshots: one column per task, env camera on top and
wrist camera on the bottom.

    python environment_screenshots.py [--out PATH] [--seed N]

Needs a real render backend (SAPIEN/Vulkan), so this has to run where that's available --
inside the project's apptainer container with --nv on a GPU node, not on the login node.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from custom_maniskill_tasks import FOCUSED_CAMERA_UID, WRIST_CAMERA_UID, make_env

HERE = Path(__file__).resolve().parent

# column title, task id.
TASKS = (
    ("PushCube-v1.1", "PushCube-v1.1"),
    ("LiftPegUpright-v1.1", "LiftPegUpright-v1.1"),
    ("PlaceSphere-v1.1", "PlaceSphere-v1.1"),
)

# row title, camera_view, sensor uid that view renders under -- see cameras.py.
ROWS = (
    ("Env Camera", "default", FOCUSED_CAMERA_UID),
    ("Wrist Camera", "wrist", WRIST_CAMERA_UID),
)


def render_frame(task_id, camera_view, sensor_uid, seed):
    """The single rgb frame `task_id` renders under `camera_view`, as an (H, W, 3) uint8 array."""
    env = make_env(task_id, obs_mode="rgb", camera_view=camera_view, num_envs=1)
    try:
        obs, _ = env.reset(seed=seed)
        return obs["sensor_data"][sensor_uid]["rgb"][0].cpu().numpy()
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
        help="reset seed, shared across every task/camera so all six frames show the same "
        "starting configuration",
    )
    args = parser.parse_args()

    fig, axes = plt.subplots(len(ROWS), len(TASKS), figsize=(4 * len(TASKS), 4 * len(ROWS)))

    for col, (title, task_id) in enumerate(TASKS):
        for row, (row_title, camera_view, sensor_uid) in enumerate(ROWS):
            frame = render_frame(task_id, camera_view, sensor_uid, args.seed)
            ax = axes[row, col]
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=14)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=13)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
