"""Every (base, decoy) marking pair of `LiftPegUpright-v2`, as the wrist camera sees it.

    python base_marking_renders/marking_matrix.py [--colours fixed|random] [--resolution 224] [--seed N]

Rows are base markings, columns decoy markings; the held-out (evaluation) markings are starred.
Each frame is the policy's observation at `--resolution`. Every cell is reset with its own seed (so
with `--colours random` each gets its own colour draw), then has the robot put back in the pose a
reset with `--seed` gave and the peg moved to the middle of the wrist camera's view, lying along x
with its base end at +x (the far end in the image): the camera pose and peg pose are the same in
every cell, and only the markings (and colours) differ.
Needs a real render backend (pin `CUDA_VISIBLE_DEVICES` to a GPU with free memory).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transforms3d.euler import euler2quat

from custom_maniskill_tasks import BASE_MARKINGS, TOP_MARKINGS, WRIST_CAMERA_UID, make_env

HERE = Path(__file__).resolve().parent
PEG_XY = (0.0, 0.0)
"""Where the peg is put for every frame: the middle of the stock spawn region."""


def robot_state(env, seed: int) -> dict:
    """The robot's articulation state after a reset with `seed`."""
    env.reset(seed=seed, options={"reconfigure": True})
    return {k: v.clone() for k, v in env.unwrapped.get_state_dict()["articulations"].items()}


def wrist_frame(env, base: str, top: str, seed: int, robot: dict) -> np.ndarray:
    task = env.unwrapped
    task.marking_pairs = ((base, top),)
    env.reset(seed=seed, options={"reconfigure": True})
    assert (task.peg_markings[0]["base"], task.peg_markings[0]["top"]) == (base, top)
    state = task.get_state_dict()
    state["articulations"] = {k: v.clone() for k, v in robot.items()}
    peg = state["actors"]["peg"]
    peg[..., :3] = torch.tensor([*PEG_XY, task.peg_half_width])
    peg[..., 3:7] = torch.tensor(euler2quat(np.pi / 2, 0, 0), dtype=torch.float32)  # base end at +x
    peg[..., 7:] = 0
    task.set_state_dict(state)
    rgb = task.get_obs()["sensor_data"][WRIST_CAMERA_UID]["rgb"]
    return rgb[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--colours", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    bases = BASE_MARKINGS["train"] + BASE_MARKINGS["evaluation"]
    tops = TOP_MARKINGS["train"] + TOP_MARKINGS["evaluation"]
    held_out = set(BASE_MARKINGS["evaluation"] + TOP_MARKINGS["evaluation"])
    label = lambda m: f"{m} *" if m in held_out else m

    env = make_env(
        "LiftPegUpright-v2", obs_mode="rgb", camera_view="wrist", camera_resolution=args.resolution,
        frame_skip=1, n_frames=1, sim_backend="physx_cpu", markings="full", colours=args.colours,
    )
    robot = robot_state(env, args.seed)
    fig, axes = plt.subplots(len(bases), len(tops), figsize=(2.6 * len(tops), 2.75 * len(bases)))
    for r, base in enumerate(bases):
        for c, top in enumerate(tops):
            ax = axes[r, c]
            ax.imshow(wrist_frame(env, base, top, args.seed + 1 + r * len(tops) + c, robot))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"decoy: {label(top)}", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"base: {label(base)}", fontsize=11)
    env.close()
    fig.suptitle(
        f"LiftPegUpright-v2: every base x decoy marking pair, wrist camera at {args.resolution} px "
        f"({args.colours} colours)\nbase end at +x in every frame; * = held out for evaluation",
        fontsize=12,
    )
    fig.tight_layout()
    out = args.out or HERE / f"marking_matrix_wrist_{args.colours}_{args.resolution}px.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
