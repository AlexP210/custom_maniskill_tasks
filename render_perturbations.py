"""Renders of every perturbation class at the ends of its training and evaluation ranges.

    python render_perturbations.py [--task LiftPegUpright-v1.1] [--seed 0] [--classes lighting ...]
        [--out-dir perturbation_renders]

For every class in `perturbation_classes.CLASSES`, two PNGs -- `wrist_<class>.png` (the wrist
camera's observation, what a policy sees) and `scene_<class>.png` (the task's three-quarter
`render_camera`). One row per aspect, with the class's other aspects neutral, at four values: the
low and high ends of the training range, and the low and high ends of the evaluation range. For a
two-sided range those are the edges of the central 80% and the range's own ends; for a one-sided
one (`lighting.direction`) the training range starts at the neutral value and its top end is the
evaluation range's bottom end. The categorical `table_material.material` has no ends, so its one
row is every material, training's then evaluation's. Evaluation values are framed in red.

Each frame is its own env, built by `make_env` with the values fixed, so it goes through the same
validation and scene-building code a training episode does, and all frames share the reset seed.
`viewpoint` moves only the wrist camera, so its scene sheet shows the same frame throughout.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from custom_maniskill_tasks import make_env
from custom_maniskill_tasks.perturbation_classes import CLASSES, Range, registry
from custom_maniskill_tasks.viewpoint import rig_for_task

HERE = Path(__file__).resolve().parent

CAMERAS = ("wrist", "scene")
SCENE_CELL = 336
LABEL_WIDTH = 170
HEADER = 24
CAPTION = 16
BACKGROUND = (30, 30, 30)
TEXT = (235, 235, 235)
RED = (230, 60, 60)


def value_at(r: Range, u: float) -> float:
    """The value a fraction `u` of the way along `r`, in the range's own (log or linear) scale."""
    if r.log:
        return float(np.exp(np.log(r.low) + u * (np.log(r.high) - np.log(r.low))))
    return r.low + u * (r.high - r.low)


def row_cells(aspect) -> list[tuple[str, float | str, bool]]:
    """[(column label, value, is evaluation), ...] for one aspect's row."""
    if aspect.choices is not None:
        return [("train", m, False) for m in aspect.choices["train"]] + [
            ("evaluation", m, True) for m in aspect.choices["evaluation"]
        ]
    r = aspect.range
    (train_low, train_high), = r.bands("train")
    evaluation = r.bands("evaluation")
    return [
        ("train min", value_at(r, train_low), False),
        ("train max", value_at(r, train_high), False),
        ("evaluation min", value_at(r, evaluation[0][0]), True),
        ("evaluation max", value_at(r, evaluation[-1][1]), True),
    ]


def render(task: str, perturbations: dict, seed: int) -> dict[str, np.ndarray]:
    env = make_env(task, obs_mode="rgb", camera_view="wrist", sim_backend="physx_cpu", robot_init_qpos_noise=0.0,
                   perturbations=perturbations)
    obs, _ = env.reset(seed=seed)
    frames = {
        "wrist": np.asarray(obs["sensor_data"]["hand_camera"]["rgb"].cpu())[0].copy(),
        "scene": np.asarray(env.render().cpu())[0].copy(),
    }
    env.close()
    return frames


def render_class(task: str, cls: str, seed: int):
    """{row label: [(column label, value, is evaluation, {camera: frame}), ...]} for `cls`."""
    aspects = registry(float(np.degrees(rig_for_task(task).fov)))[cls]
    rows = {}
    for name, aspect in aspects.items():
        label = name + (f"\n({aspect.unit})" if aspect.unit else "")
        rows[label] = []
        for column, value, evaluation in row_cells(aspect):
            print(f"  {cls}.{name}={value}")
            frames = render(task, {cls: {name: value}}, seed)
            rows[label].append((column, value, evaluation, frames))
    return rows


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def wrap(text: str, width: int) -> str:
    """`text` broken onto lines of at most `width` characters, at spaces."""
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return "\n".join(lines + [line])


def sheet(rows, camera: str, title: str) -> Image.Image:
    first = next(iter(rows.values()))[0][3][camera]
    cell = SCENE_CELL if camera == "scene" else first.shape[0]
    columns = max(len(cells) for cells in rows.values())
    out = Image.new("RGB", (LABEL_WIDTH + columns * cell, HEADER + len(rows) * (cell + CAPTION)), BACKGROUND)
    draw = ImageDraw.Draw(out)
    draw.text((6, 5), title, fill=TEXT, font=font(12))
    for row, (label, cells) in enumerate(rows.items()):
        top = HEADER + row * (cell + CAPTION)
        name, _, unit = label.partition("\n")
        draw.text((6, top + CAPTION + cell // 2 - 20), name, fill=(200, 210, 230), font=font(13))
        draw.multiline_text((6, top + CAPTION + cell // 2), wrap(unit, 26), fill=(150, 155, 170), font=font(10))
        for column, (heading, value, evaluation, frames) in enumerate(cells):
            left = LABEL_WIDTH + column * cell
            image = Image.fromarray(frames[camera])
            if image.size != (cell, cell):
                image = image.resize((cell, cell), Image.LANCZOS)
            out.paste(image, (left, top + CAPTION))
            shown = value if isinstance(value, str) else f"{value:.3g}"
            draw.text((left + 4, top + 2), f"{heading}: {shown}", fill=RED if evaluation else TEXT, font=font(11))
            if evaluation:
                draw.rectangle([left, top + CAPTION, left + cell - 1, top + CAPTION + cell - 1], outline=RED, width=3)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", default="LiftPegUpright-v1.1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classes", nargs="+", choices=CLASSES, default=list(CLASSES))
    parser.add_argument("--out-dir", type=Path, default=HERE / "perturbation_renders")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cls in args.classes:
        print(cls)
        rows = render_class(args.task, cls, args.seed)
        for camera in CAMERAS:
            title = (f"{args.task} / {cls} / {camera} camera. One aspect per row, the rest neutral.  "
                     f"red = evaluation range")
            path = args.out_dir / f"{camera}_{cls}.png"
            sheet(rows, camera, title).save(path)
            print(f"wrote {path.resolve()}")


if __name__ == "__main__":
    main()
