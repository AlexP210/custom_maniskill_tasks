"""Contact sheets of one `domain_randomization` axis: what each of its parameters does, and where
the training / held-out split falls.

    python domain_randomization_sheet.py table_visual_material object_visual_material \
        [--task LiftPegUpright-v1.1] [--seed 0] [--out-dir .]

For every axis named, two sheets -- `<task>_dr_scene_<axis>.png` (the task's three-quarter
`render_camera`) and `<task>_dr_wrist_<axis>.png` (the wrist camera a policy sees). One row per
parameter, swept with the axis's other parameters held at their neutral (stock-scene) values, at
five points along its range: for a range with its neutral value in the middle, the two ends, the
two edges of the training band and the neutral value; for one that starts at its neutral value,
the neutral value, the middle and top edge of the training band, and two points of the held-out
top. Held-out values are framed in red.

The env is built with the axis switched on, and each frame's condition is set rather than drawn,
by standing in for the env's draw, so the frames go through exactly the code a training episode
does.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from custom_maniskill_tasks import make_env
from custom_maniskill_tasks.domain_randomization import RANGES, Range

HERE = Path(__file__).resolve().parent

TWO_SIDED_POINTS = (0.0, 0.1, 0.5, 0.9, 1.0)
ONE_SIDED_POINTS = (0.0, 0.4, 0.8, 0.9, 1.0)
SCENE_CELL = 336
LABEL_WIDTH = 160
HEADER = 20
CAPTION = 14
BACKGROUND = (30, 30, 30)
RED = (230, 60, 60)
# A parameter that does nothing at the others' neutral values is swept with these held instead,
# and its row label says so: the table's specular shows only on wood made less metallic.
ROW_CONTEXT = {("table_visual_material", "specular"): {"metallic_reduction": 0.8}}
SHORT_NAMES ={"LiftPegUpright-v1.1": "lift_peg", "PushCube-v1.1": "push_cube", "PushCube-v1.2": "push_cube_fr3",
               "PickCube-v1.1": "pick_cube", "PlaceSphere-v1.1": "place_sphere"}


def value_at(r: Range, u: float) -> float:
    """The value a fraction `u` of the way along `r`, in the range's own (log or linear) scale."""
    if r.log:
        return float(np.exp(np.log(r.low) + u * (np.log(r.high) - np.log(r.low))))
    return r.low + u * (r.high - r.low)


def neutral(r: Range) -> float:
    return value_at(r, 0.5) if r.two_sided else r.low


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_axis(task: str, axis: str, seed: int):
    """{parameter: [(value, held_out, scene_frame, wrist_frame), ...]} for every parameter of `axis`."""
    env = make_env(task, obs_mode="rgb", camera_view="wrist", sim_backend="physx_cpu", robot_init_qpos_noise=0.0,
                   domain_randomization=[axis])
    base = env.unwrapped
    ranges = RANGES[axis]
    rows = {}
    for name, r in ranges.items():
        context = ROW_CONTEXT.get((axis, name), {})
        label = name + "".join(f"\n({k}={v:g})" for k, v in context.items())
        rows[label] = []
        for u in TWO_SIDED_POINTS if r.two_sided else ONE_SIDED_POINTS:
            condition = {k: neutral(other) for k, other in ranges.items()}
            condition.update(ROW_CONTEXT.get((axis, name), {}))
            condition[name] = value_at(r, u)
            base._draw = lambda _axis, condition=condition: dict(condition)
            obs, _ = env.reset(seed=seed)
            wrist = np.asarray(obs["sensor_data"]["hand_camera"]["rgb"].cpu())[0].copy()
            scene = np.asarray(env.render().cpu())[0].copy()
            held_out = r.contains(condition[name], "heldout") and not r.contains(condition[name], "train")
            rows[label].append((name, condition[name], held_out, scene, wrist))
    env.close()
    return rows


def sheet(rows, camera: str, title: str) -> Image.Image:
    index = 3 if camera == "scene" else 4
    first = next(iter(rows.values()))[0][index]
    cell = SCENE_CELL if camera == "scene" else first.shape[0]
    columns = max(len(v) for v in rows.values())
    width = LABEL_WIDTH + columns * cell
    height = HEADER + len(rows) * (cell + CAPTION)
    out = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(out)
    draw.text((6, 4), title, fill=(230, 230, 230), font=font(11))
    for row, (label, frames) in enumerate(rows.items()):
        top = HEADER + row * (cell + CAPTION)
        draw.multiline_text((6, top + CAPTION + cell // 2 - 6), label, fill=(200, 210, 230), font=font(11))
        for column, (name, value, held_out, *images) in enumerate(frames):
            left = LABEL_WIDTH + column * cell
            image = Image.fromarray(images[index - 3]).resize((cell, cell), Image.LANCZOS)
            draw.rectangle([left, top, left + cell - 1, top + CAPTION - 1], fill=(0, 0, 0))
            out.paste(image, (left, top + CAPTION))
            colour = RED if held_out else (235, 235, 235)
            draw.text((left + 4, top + 1), f"{name}={value:.2f}" + ("  held out" if held_out else ""),
                      fill=colour, font=font(10))
            if held_out:
                draw.rectangle([left, top, left + cell - 1, top + CAPTION + cell - 1], outline=RED, width=3)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("axes", nargs="+", choices=sorted(RANGES))
    parser.add_argument("--task", default="LiftPegUpright-v1.1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()

    short = SHORT_NAMES.get(args.task, args.task)
    for axis in args.axes:
        rows = render_axis(args.task, axis, args.seed)
        for camera, label in (("scene", "scene camera"), ("wrist", "wrist camera")):
            title = (f"{args.task} / {axis} / {label}. One parameter at a time, the rest neutral.  "
                     f"red = held out of training")
            path = args.out_dir / f"{short}_dr_{camera}_{axis}.png"
            sheet(rows, camera, title).save(path)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
