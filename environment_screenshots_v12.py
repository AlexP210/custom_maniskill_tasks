"""The `-v1.2` (FR3 + ZED mini) counterpart of `environment_screenshots.py`: one row per task, and
five columns -- the episode's start state, the ZED wrist camera the policy observes at that start,
and the same start state under the three visual perturbations evaluated (brighter lights, warmer
lights, the task objects' colours shifted round the hue wheel).

There is no "Goal" column: that one comes from rolling out a trained visual PPO expert, and none
exists for the `-v1.2` ids.

    python environment_screenshots_v12.py [--out PATH] [--seed N]

Needs a real render backend (SAPIEN/Vulkan).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from custom_maniskill_tasks import make_env

HERE = Path(__file__).resolve().parent

TASKS = tuple(f"{name}-v1.2" for name in ("PushCube", "LiftPegUpright", "PickCube"))

# column title, lighting preset; the same ones as environment_screenshots.py
LIGHTING_COLUMNS = (
    ("Bright", "bright-set-0.75-2.5"),
    ("Warm", "warm-set-1.1"),
    ("Colour-Shift", "object-hue-30"),
)
FONT_SIZE = 38
COLUMN_TITLES = ("Start", "Visual Obs") + tuple(title for title, _ in LIGHTING_COLUMNS)


def render_start(task_id, seed, **kwargs):
    """(render-camera frame, ZED wrist frame) at the start of the seeded episode."""
    env = make_env(task_id, obs_mode="rgb", camera_view="wrist", sim_backend="physx_cpu", **kwargs)
    try:
        obs, _ = env.reset(seed=seed)
        wrist = np.asarray(obs["sensor_data"]["hand_camera"]["rgb"].cpu())[0]
        return np.asarray(env.render().cpu())[0], wrist
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=HERE / "environment_screenshots_v12.png",
                        help="path to write the grid image to; a PDF copy is written alongside it")
    parser.add_argument("--seed", type=int, default=0,
                        help="reset seed, shared by every cell so each row shows one configuration")
    args = parser.parse_args()

    frames = {}
    for row, task_id in enumerate(TASKS):
        frames[row, 0], frames[row, 1] = render_start(task_id, args.seed)
        for col, (_, lighting) in enumerate(LIGHTING_COLUMNS, start=2):
            frames[row, col], _ = render_start(task_id, args.seed, lighting=lighting)

    fig, axes = plt.subplots(len(TASKS), len(COLUMN_TITLES),
                             figsize=(4 * len(COLUMN_TITLES), 4 * len(TASKS)))
    for row, task_id in enumerate(TASKS):
        for col, title in enumerate(COLUMN_TITLES):
            ax = axes[row, col]
            ax.imshow(frames[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=FONT_SIZE)
            if col == 0:
                ax.set_ylabel(task_id.replace("-v", "\n-v"), fontsize=FONT_SIZE)

    fig.tight_layout()
    for path in (args.out, args.out.with_suffix(".pdf")):
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
