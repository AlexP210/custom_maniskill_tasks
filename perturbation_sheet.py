"""A contact sheet of a task under any combination of the appearance kwargs.

    python perturbation_sheet.py [--task ID] [--seed N] [--out PATH] [--hand-only [--cols N]] SPEC ...

Each SPEC is one column: "default", or comma-separated `kwarg=value` pairs for `make_env`, e.g.

    python perturbation_sheet.py default corruption=blur-2 "texture=table-checker,distractors=3" \
        "lighting=dim,viewpoint=back-4+pitch-up-6"

With `--hand-only`, one frame per SPEC (the wrist camera's observation) in a grid `--cols` wide,
titled with just the value when every SPEC sets the same single kwarg. Otherwise two rows: the wrist camera's observation (which is what a policy sees, corruption included) and
the task's three-quarter `render_camera` view (which shows the scene, textures and distractors,
but not `corruption`, since that acts on the observations). Needs a real render backend.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from custom_maniskill_tasks import make_env

HERE = Path(__file__).resolve().parent


def parse_spec(spec: str) -> dict:
    if spec == "default":
        return {}
    kwargs = {}
    for pair in spec.split(","):
        key, _, value = pair.partition("=")
        kwargs[key] = int(value) if key == "distractors" else value
    return kwargs


def render(task: str, kwargs: dict, seed: int):
    env = make_env(task, obs_mode="rgb", camera_view="wrist", sim_backend="physx_cpu", **kwargs)
    obs, _ = env.reset(seed=seed)
    hand = np.asarray(obs["sensor_data"]["hand_camera"]["rgb"].cpu())[0]
    view = np.asarray(env.render().cpu())[0]
    env.close()
    return hand, view


def titles_for(specs: list[str]) -> tuple[list[str], str | None]:
    """Column titles, and the kwarg they all share if there is exactly one such."""
    keys = {pair.partition("=")[0] for spec in specs if spec != "default" for pair in spec.split(",")}
    if len(keys) == 1 and all("," not in spec for spec in specs):
        key = next(iter(keys))
        return [spec.partition("=")[2] or "default" for spec in specs], key
    return [spec.replace(",", "\n") for spec in specs], None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--task", default="PushCube-v1.1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=HERE / "perturbation_sheet.png")
    parser.add_argument("--hand-only", action="store_true")
    parser.add_argument("--cols", type=int, default=5)
    args = parser.parse_args()

    frames = [render(args.task, parse_spec(spec), args.seed) for spec in args.specs]
    titles, shared = titles_for(args.specs)
    if args.hand_only:
        cols = min(args.cols, len(frames))
        rows = -(-len(frames) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, title, (hand, _) in zip(axes.flat, titles, frames):
            ax.imshow(hand)
            ax.set_title(title, fontsize=9)
        fig.suptitle(f"{args.task}, {shared}" if shared else args.task, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(args.out, dpi=100)
        print(f"wrote {args.out}")
        return
    fig, axes = plt.subplots(2, len(frames), figsize=(3 * len(frames), 6.4), squeeze=False)
    for col, (title, pair) in enumerate(zip(titles, frames)):
        for row, frame in enumerate(pair):
            axes[row, col].imshow(frame)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[0, col].set_title(title, fontsize=8)
    axes[0, 0].set_ylabel("wrist camera")
    axes[1, 0].set_ylabel("render camera")
    fig.suptitle(args.task)
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
