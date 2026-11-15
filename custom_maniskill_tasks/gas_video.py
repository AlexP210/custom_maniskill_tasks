"""DBC's "ideal gas" distractor, rendered as a video to play on the table top (`table_video`).

    python -m custom_maniskill_tasks.gas_video --out /data/.../ideal_gas.mp4 [--frames 400] [--seed 0]

Deep Bisimulation for Control (Zhang et al., 2021; agents/deep_bisim4control/distractors) plays a
pre-rendered video of `IdealGas` -- `num_bodies` non-interacting particles moving in straight lines
in a box and bouncing off its walls -- behind the agent. This renders the same dynamics directly to
an mp4 sized for the table top, for `table_video` to lay over the table:

- the box is the frame, `width` by `height` pixels (2:1 by default, the table top's aspect; the
  video's width runs along the table's long side);
- particles start uniformly in the box with velocity components uniform in [-1, 1] times
  `speed` (box *heights* per frame), as DBC's `INIT_MAX_VEL * uniform(-1, 1)`, and bounce
  elastically off the walls. With no interaction the motion is exact, so there is no ODE solver;
- one frame per env step: write it at the env's control rate (20 Hz on ManiSkill) and
  `table_video` shows frame k at step k;
- each particle is a filled, anti-aliased disc of radius `radius` (box heights) in its own colour,
  drawn from the viridis map as DBC's scatter plot does, on a black background.

Unlike DBC the colours stay RGB (DBC loads its videos as greyscale) -- blend it into the wood with
`table_video`'s `alpha` to soften it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULTS = dict(num_particles=10, frames=400, fps=20, width=1024, height=512, speed=0.02, radius=0.04, seed=0)


def ideal_gas_positions(num_particles, frames, aspect, speed, seed):
    """(frames, num_particles, 2) positions in a box [0, aspect] x [0, 1] (x along the width)."""
    rng = np.random.RandomState(seed)
    box = np.array([aspect, 1.0])
    position = rng.uniform(0.0, 1.0, size=(num_particles, 2)) * box
    velocity = speed * rng.uniform(-1.0, 1.0, size=(num_particles, 2))
    out = np.empty((frames, num_particles, 2))
    for k in range(frames):
        out[k] = position
        position = position + velocity
        # reflect off the walls (DBC's rule: mirror the overshoot, flip the velocity)
        below, above = position < 0.0, position > box
        position = np.where(below, -position, position)
        position = np.where(above, 2 * box - position, position)
        velocity = np.where(below | above, -velocity, velocity)
    return out


def render_ideal_gas(out_path, num_particles=10, frames=400, fps=20, width=1024, height=512, speed=0.02,
                     radius=0.04, seed=0) -> Path:
    """Write the ideal-gas video to `out_path` (mp4) and return its path."""
    import cv2
    from matplotlib import colormaps

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    positions = ideal_gas_positions(num_particles, frames, width / height, speed, seed)
    colours = (np.asarray(colormaps["viridis"](np.random.RandomState(seed + 1).uniform(size=num_particles)))[:, :3] * 255)
    bgr = colours[:, ::-1].astype(int).tolist()
    pixel_radius = max(1, int(round(radius * height)))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open a video writer for {out_path}")
    for k in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for (x, y), colour in zip(positions[k] * height, bgr):
            cv2.circle(frame, (int(round(x)), int(round(height - y))), pixel_radius, colour, thickness=-1, lineType=cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    args = vars(parser.parse_args())
    out = args.pop("out")
    print(f"wrote {render_ideal_gas(out, **args)}")


if __name__ == "__main__":
    main()
