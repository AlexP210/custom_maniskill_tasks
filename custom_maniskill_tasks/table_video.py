"""A video playing on the table top: the `table_video` env kwarg.

    make_env("LiftPegUpright-v1.1", ..., table_video={"path": "/data/.../clip.mp4"})

The video covers the whole table top in place of the wood, playing in real time from `start_s`
seconds in: the frame shown with an observation is the one at `start_s + step / control_freq`,
the same clip in every episode -- or, with `random_start`, a loop of `frames` frames that each env
enters at its own random point every episode (seeded by that env's episode seed), so the video
differs from episode to episode as DBC's does, for training. `alpha` below 1 blends it with the table's own wood: each pixel is
`alpha` of the video and `1 - alpha` of the wood the table has at that spot (mixed in linear light,
laid out in the table's own texture space and drawn with the table's own material, so 0 is the
plain table again, sheen and all). It is purely visual -- nothing on the table touches it -- and it
works on physx_cpu and physx_cuda alike.

Why a stack of quads and not a texture that changes: once a scene has rendered, neither a new
texture nor new UVs reach SAPIEN's renderer (on either backend), but actor poses do. So the frames
an episode needs (`steps + 1` of them: the reset's and one per step) are decoded once, each is put
on its own kinematic, visual-only quad the size of the table top, and before every step the quad
for that step is raised to just above the table while every other one waits inside the table top,
out of sight. An episode longer than `steps` loops the clip.

Mapped so the video stands upright to a camera at the robot looking out over the table: the top of
the frame at the table's far edge (+x), its left edge on the robot's left (+y). The frame is centre-
cropped to the table top's aspect ratio (about 2:1), so nothing is stretched.

`table_video` is an env kwarg rather than a perturbation class because it names a file: the
perturbation classes are numeric or categorical aspects, validated without a simulator.
"""

from __future__ import annotations

import atexit
import collections
import functools
import hashlib
import os
from typing import Any, Mapping

import numpy as np

from custom_maniskill_tasks.appearance import _table_top_uv_map

TABLE_VIDEO_LIFT = 5e-4
"""How far above the table top the playing frame lies, in metres."""
TABLE_VIDEO_PARKED_DEPTH = 0.01
"""How far below the table top the waiting frames are parked: inside the top's 4.5 cm slab."""
DEFAULTS = {"start_s": 0.0, "steps": 50, "width": 1024, "alpha": 1.0, "random_start": False, "frames": None}


def resolve_table_video(spec: Mapping[str, Any] | None) -> dict | None:
    """`spec` with its defaults filled in and checked, or None for no video."""
    if spec is None:
        return None
    spec = dict(spec)
    unknown = sorted(set(spec) - {"path", *DEFAULTS})
    if unknown or "path" not in spec:
        raise ValueError(f"table_video takes 'path' and optionally {sorted(DEFAULTS)}; got {sorted(spec)}")
    out = {**DEFAULTS, **spec}
    out["start_s"], out["steps"], out["width"] = float(out["start_s"]), int(out["steps"]), int(out["width"])
    out["alpha"] = float(out["alpha"])
    out["random_start"] = bool(out["random_start"])
    if out["random_start"]:
        if out["frames"] is None:
            raise ValueError("table_video random_start needs 'frames': how many frames the loop it starts in has")
        out["frames"] = int(out["frames"])
    elif out["frames"] is not None:
        raise ValueError("table_video 'frames' only applies with random_start (otherwise it plays steps + 1 frames)")
    if out["steps"] < 1 or out["width"] < 16 or out["start_s"] < 0 or not 0.0 <= out["alpha"] <= 1.0 \
            or (out["random_start"] and out["frames"] < 2):
        raise ValueError(f"table_video={out} is out of range")
    return out


@functools.lru_cache(maxsize=4)
def _decode_frames(path: str, times: tuple[float, ...], width: int, aspect: float) -> np.ndarray:
    """(len(times), h, width, 3) uint8 RGB frames at `times` seconds, centre-cropped to `aspect`
    (width / height) and resized to `width` across. Cached per process: every env build in a run
    plays the same clip."""
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"table_video could not open {path!r}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = [int(round(t * fps)) for t in times]
    if max(wanted) >= count:
        raise ValueError(
            f"table_video needs frame {max(wanted)} ({max(times):.2f} s) but {path!r} has {count} "
            f"({count / fps:.2f} s); lower start_s or steps"
        )
    frames, index = {}, 0
    for target in sorted(set(wanted)):
        # read forward rather than seek: seeking an h264 stream lands on keyframes
        while index <= target:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"table_video: {path!r} ended at frame {index}, before {target}")
            index += 1
        frames[target] = frame
    capture.release()

    def prepare(bgr):
        h, w = bgr.shape[:2]
        if w / h > aspect:  # too wide: crop the sides
            cw = int(round(h * aspect))
            bgr = bgr[:, (w - cw) // 2:(w - cw) // 2 + cw]
        else:  # too tall: crop top and bottom
            ch = int(round(w / aspect))
            bgr = bgr[(h - ch) // 2:(h - ch) // 2 + ch]
        height = int(round(width / aspect))
        return cv2.cvtColor(cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

    return np.stack([prepare(frames[i]) for i in wanted])


def _frames_in_table_uv_space(env, frames: np.ndarray, alpha: float, extent) -> list:
    """Each of `frames` laid out in the table texture's own UV space, blended `alpha` : `1 - alpha`
    with the wood in linear light: (h_t, w_t, 4) uint8 images the size of the table's texture.

    The table top spans just under one repeat of its texture, so every texel (i, j) of a texture-
    sized image stands for one point of the top: its UV is (j + 0.5) / w_t, (i + 0.5) / h_t, moved by
    whole repeats into the range the top covers, and the affine table UV map turns that back into
    a world (x, y). The video pixel there is sampled bilinearly (frame row = distance from the far
    edge, column = distance from the robot's left edge) and mixed with the wood texel itself, so at
    alpha 0 the image is the table's texture exactly. The sliver of texels past the top's edge
    (about 2% of each axis) takes the nearest edge pixel of the frame."""
    import cv2

    from custom_maniskill_tasks.lighting import _linear_to_srgb, _srgb_to_linear

    x0, x1, y0, y1 = extent
    uv_map, texture = _table_top_uv_map(env)[0], _table_material(env).base_color_texture
    wood = texture.download()
    th, tw = wood.shape[:2]
    corners = np.array([[x, y, 1.0] for x in (x0, x1) for y in (y0, y1)]) @ uv_map.T
    u_min, v_min = corners.min(axis=0)
    jj, ii = np.meshgrid(np.arange(tw) + 0.5, np.arange(th) + 0.5)
    u = u_min + np.mod(jj / tw - u_min, 1.0)
    v = v_min + np.mod(ii / th - v_min, 1.0)
    xy = np.linalg.solve(uv_map[:, :2], np.stack([u - uv_map[0, 2], v - uv_map[1, 2]]).reshape(2, -1))
    x, y = xy.reshape(2, th, tw)
    h, w = frames.shape[1:3]
    rows = ((x1 - x) / (x1 - x0) * h - 0.5).astype(np.float32)
    cols = ((y1 - y) / (y1 - y0) * w - 0.5).astype(np.float32)
    decode = _srgb_to_linear if texture.is_srgb else (lambda c: c)
    encode = _linear_to_srgb if texture.is_srgb else (lambda c: c)
    wood_linear = decode(wood[..., :3].astype(np.float64) / 255.0)
    out = []
    for frame in frames:
        video = cv2.remap(frame, cols, rows, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mixed = encode(alpha * _srgb_to_linear(video.astype(np.float64) / 255.0) + (1.0 - alpha) * wood_linear)
        image = wood.copy()
        image[..., :3] = np.clip(np.round(mixed * 255.0), 0, 255).astype(np.uint8)
        out.append(image)
    return out


def _table_material(env):
    import sapien

    body = env.table_scene.table._objs[0].find_component_by_type(sapien.render.RenderBodyComponent)
    return next(part.material for shape in body.render_shapes for part in shape.parts
                if part.material.base_color_texture is not None)


def _material_like(source, base_color_texture):
    """A new material with every property and map of `source` but its colour image."""
    import sapien

    material = sapien.render.RenderMaterial(
        emission=list(source.emission), base_color=list(source.base_color), specular=source.specular,
        roughness=source.roughness, metallic=source.metallic, transmission=source.transmission,
        ior=source.ior, transmission_roughness=source.transmission_roughness,
    )
    for name in ("normal_texture", "roughness_texture", "metallic_texture", "emission_texture", "transmission_texture"):
        texture = getattr(source, name)
        if texture is not None:
            setattr(material, name, texture)
    material.base_color_texture = base_color_texture
    return material


_FRAME_TEXTURES: "collections.OrderedDict[tuple, list]" = collections.OrderedDict()
_FRAME_TEXTURES_KEPT = 4
_FRAME_TEXTURES_RELEASE_REGISTERED = False
"""Uploaded frame textures, shared by every env instance in the process that plays the same frames.

A run builds several envs -- the trainer's and one per evaluator -- and each used to upload its
own copy: 200 frames of 1024 x 1024 RGBA (with mipmaps, ~5.6 MB each) is ~1.1 GB of GPU memory per
env instance. sapien's render textures belong to the process, not to a scene (which is also why an
edited texture leaks into later envs; see lighting._remapped_texture), so one upload can be shared.
The key is everything the pixels depend on; the most recent `_FRAME_TEXTURES_KEPT` sets are kept
for the life of the process."""


def _frame_textures(env, spec: dict, times: tuple, extent, uv_map, template) -> list:
    """The frames at `times`, laid out in the table texture's UV space and blended per `spec`, as
    sapien textures like `template` (the table's colour texture) -- from the process-wide cache when
    another env instance has already uploaded the same ones."""
    import sapien

    wood = template.download()
    key = (
        spec["path"], os.path.getmtime(spec["path"]), times, spec["width"], spec["alpha"],
        hashlib.sha1(wood.tobytes()).hexdigest(), tuple(np.round(uv_map, 9).ravel()), tuple(np.round(extent, 9)),
        template.format, template.mipmap_levels, template.filter_mode, template.address_mode, template.is_srgb,
    )
    if key in _FRAME_TEXTURES:
        _FRAME_TEXTURES.move_to_end(key)
        return _FRAME_TEXTURES[key]
    x0, x1, y0, y1 = extent
    frames = _decode_frames(spec["path"], times, spec["width"], (y1 - y0) / (x1 - x0))
    images = _frames_in_table_uv_space(env, frames, spec["alpha"], extent)
    textures = [
        sapien.render.RenderTexture2D(
            image, template.format, template.mipmap_levels, template.filter_mode, template.address_mode,
            template.is_srgb,
        )
        for image in images
    ]
    global _FRAME_TEXTURES_RELEASE_REGISTERED
    if not _FRAME_TEXTURES_RELEASE_REGISTERED:
        # Drop the textures at exit before sapien's own teardown (atexit runs last-registered first,
        # and this registers after sapien is imported): freed after the renderer they segfault.
        atexit.register(_FRAME_TEXTURES.clear)
        _FRAME_TEXTURES_RELEASE_REGISTERED = True
    _FRAME_TEXTURES[key] = textures
    while len(_FRAME_TEXTURES) > _FRAME_TEXTURES_KEPT:
        _FRAME_TEXTURES.popitem(last=False)
    return textures


def build_table_video(env, spec: dict):
    """The frame quads for `env`'s parallel envs, merged into one `Actor` (frame-major: all envs'
    quads for frame 0, then frame 1, ...), with the centre, the shown and parked heights and the
    frame count. Built from `_load_scene`, so the table is already in place.

    Each quad is the table top's own rectangle with the table's own UVs and material -- normal,
    roughness and metallic maps included -- and only the colour image replaced by the frame laid out
    in that UV space (`_frames_in_table_uv_space`), so the top keeps the wood's sheen and, at alpha
    0, renders as the plain table. Loaded from an OBJ so sapien computes the tangents the normal map
    needs (`appearance.build_table_spin_disc` explains)."""
    import sapien
    from mani_skill.utils.structs.actor import Actor

    from custom_maniskill_tasks.appearance import _write_obj

    _, top_z, ((x0, x1), (y0, y1)), normal = _table_top_uv_map(env)
    depth, length = x1 - x0, y1 - y0  # the video's height runs along x, its width along y
    control_interval = 1.0 / float(env.control_freq)
    count = spec["frames"] if spec["random_start"] else spec["steps"] + 1
    times = tuple(spec["start_s"] + k * control_interval for k in range(count))
    uv_map = _table_top_uv_map(env)[0]
    source = _table_material(env)
    textures = _frame_textures(env, spec, times, (x0, x1, y0, y1), uv_map, source.base_color_texture)

    centre = ((x0 + x1) / 2, (y0 + y1) / 2)
    world = np.array([[x1, y1], [x1, y0], [x0, y0], [x0, y1]])
    vertices = np.c_[world - centre, np.zeros(4)].astype(np.float32)
    uvs = (np.c_[world, np.ones(4)] @ uv_map.T).astype(np.float32)
    triangles = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32)  # counter-clockwise seen from above
    normals = np.tile(normal, (4, 1)).astype(np.float32)
    path = _write_obj(vertices, triangles, normals, uvs)

    quads = []
    for k, texture in enumerate(textures):
        material = _material_like(source, texture)  # per env instance; cheap, and points at the shared texture
        for i in range(env.num_envs):
            builder = env.scene.create_actor_builder()
            builder._procedural_shapes.append(
                sapien.render.RenderShapeTriangleMesh(path, np.ones(3, dtype=np.float32), material)
            )
            builder.set_scene_idxs([i])
            builder.initial_pose = sapien.Pose([centre[0], centre[1], top_z - TABLE_VIDEO_PARKED_DEPTH])
            quads.append(builder.build_kinematic(name=f"table_video_{k}_{i}"))
            # Kept out of the sim state: the quads are visual-only and their poses follow from the
            # step counter, while `get_state_dict()` (which the S2P task wrapper calls on every
            # reset and step) snapshots every registered actor through its own
            # `cuda_rigid_body_data.torch()` view, each leaking ~2 KB of host memory in SAPIEN.
            # At 256 envs x 200 frames that is 51k views per step: ~6 GB/h of training, and most
            # of a step's time.
            env.remove_from_state_dict_registry(quads[-1])
    actor = Actor.merge(quads, name="table_video")
    shown_z, parked_z = top_z + TABLE_VIDEO_LIFT, top_z - TABLE_VIDEO_PARKED_DEPTH
    return actor, centre, shown_z, parked_z, len(textures)
