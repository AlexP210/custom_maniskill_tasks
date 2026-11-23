"""A video playing on a wall behind the scene: the `background_video` env kwarg.

    make_env("PickCube-MultiArmDistractors", background_video="/data/.../factory_backdrop.mp4")
    make_env("LiftPegUpright-MultiArmDistractors", background_video={"path": ..., "frames": 100, "brightness": 0.8})

`BackdropVideoMixin` stands a 4 x 2.25 m (16:9) screen upright on the floor behind the distractor
stations (`BACKDROP_X`, facing +x, i.e. towards the scene camera), filling the scene camera's view
above the table where it would otherwise see the empty floor grid. The clip loops; every env
enters the loop at its own frame each episode (drawn from that env's episode seed, in a stream of
its own, so the task's draws are unchanged) and advances one frame per env step. Write the clip at
the env's control rate (20 Hz on ManiSkill) so it plays in real time. Purely visual: the screen has
no collision shape and is kept out of the state dict, like `table_video`'s quads.

The mechanism is `table_video`'s, for the same reason: once a scene has rendered, a changed
texture never reaches SAPIEN's renderer, but actor poses do. So each frame of the loop is its own
kinematic quad, and before every step the one each env should show is moved onto the wall while
the rest wait `BACKDROP_PARKED_DROP` metres below the floor, out of sight of every camera. The
frame textures are uploaded once per process and shared by every env instance (`_frame_textures`).
Cost: frames x num_envs actors (160 x 256 = 41k for the factory clip), the same order as
`table_video` with `random_start`; pass `frames` to loop a shorter prefix.

The screen is lit like any other surface, plus an emission of `brightness` times the frame itself,
so it reads as a bright scene behind the workcell under every lighting preset rather than going
dark when the key light faces away from it.

The factory clip (`/data/AlexPleava/datasets/distractor_videos/factory_backdrop.mp4`) is a 2x2
mosaic of four Pexels clips, listed with their licence in `factory_src/SOURCES.md` beside it, built
with:

    ffmpeg -i 6450803-hd_1920_1080_25fps.mp4 -i 4727768-hd_1920_1080_30fps.mp4 \\
           -i 4468754-hd_1920_1080_24fps.mp4 -i 4291725-hd_1920_1080_25fps.mp4 -filter_complex \\
      "[0:v]fps=20,scale=512:288,trim=duration=8,setpts=PTS-STARTPTS[a];
       [1:v]fps=20,scale=512:288,trim=start=5:duration=8,setpts=PTS-STARTPTS[b];
       [2:v]fps=20,scale=512:288,trim=duration=8,setpts=PTS-STARTPTS[c];
       [3:v]fps=20,scale=512:288,trim=start=2:duration=8,setpts=PTS-STARTPTS[d];
       [a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" \\
      -map "[v]" -r 20 -c:v libx264 -pix_fmt yuv420p -crf 18 factory_backdrop.mp4
"""

from __future__ import annotations

import atexit
import collections
import os
from typing import Any, Mapping

import numpy as np

BACKDROP_X = -1.25
"""The screen's plane: 0.6 m behind the distractor robots' bases (x = -0.615), clear of every link."""
BACKDROP_WIDTH, BACKDROP_HEIGHT = 4.0, 2.25
BACKDROP_BOTTOM = -0.92
"""Standing on the floor of ManiSkill's table scene (its ground plane is at z = -0.92)."""
BACKDROP_PARKED_DROP = 10.0
"""How far below its place a waiting frame is parked: under the floor, out of every camera's view."""
DEFAULTS = {"frames": None, "brightness": 0.8, "width": 1024}


def resolve_background_video(spec: str | Mapping[str, Any] | None) -> dict | None:
    """`spec` (a path, or a dict with 'path') with its defaults filled in and checked; None for none."""
    if spec is None or spec is False:
        return None
    spec = {"path": spec} if isinstance(spec, (str, os.PathLike)) else dict(spec)
    unknown = sorted(set(spec) - {"path", *DEFAULTS})
    if unknown or "path" not in spec:
        raise ValueError(f"background_video takes a path, or 'path' and optionally {sorted(DEFAULTS)}; got {sorted(spec)}")
    out = {**DEFAULTS, **spec}
    out["path"] = os.fspath(out["path"])
    if not os.path.isfile(out["path"]):
        raise FileNotFoundError(f"background_video: no file at {out['path']!r}")
    out["width"], out["brightness"] = int(out["width"]), float(out["brightness"])
    if out["frames"] is not None:
        out["frames"] = int(out["frames"])
        if out["frames"] < 2:
            raise ValueError("background_video 'frames' must be at least 2")
    if out["width"] < 16 or out["brightness"] < 0:
        raise ValueError(f"background_video={out} is out of range")
    return out


def _clip_frame_count(path: str, fps: float) -> int:
    """How many frames of `path` there are at `fps` (the env's control rate)."""
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"background_video could not open {path!r}")
    count, rate = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return int(np.floor((count - 1) / rate * fps)) + 1


_FRAME_TEXTURES: "collections.OrderedDict[tuple, list]" = collections.OrderedDict()
_FRAME_TEXTURES_KEPT = 2
_RELEASE_REGISTERED = False


def _frame_textures(spec: dict, fps: float) -> list:
    """The loop's frames as sapien textures, shared process-wide by every env instance that plays
    the same clip (see `table_video._FRAME_TEXTURES` for why one upload can be shared)."""
    import sapien

    from custom_maniskill_tasks.table_video import _decode_frames

    count = _clip_frame_count(spec["path"], fps)
    if spec["frames"] is not None:
        if spec["frames"] > count:
            raise ValueError(f"background_video wants {spec['frames']} frames but {spec['path']!r} has {count} at {fps} Hz")
        count = spec["frames"]
    key = (spec["path"], os.path.getmtime(spec["path"]), count, spec["width"], fps)
    if key in _FRAME_TEXTURES:
        _FRAME_TEXTURES.move_to_end(key)
        return _FRAME_TEXTURES[key]
    times = tuple(k / fps for k in range(count))
    frames = _decode_frames(spec["path"], times, spec["width"], BACKDROP_WIDTH / BACKDROP_HEIGHT)
    textures = [
        sapien.render.RenderTexture2D(
            np.concatenate([f, np.full(f.shape[:2] + (1,), 255, np.uint8)], axis=-1),
            "R8G8B8A8Unorm", 1, "linear", "edge", True,
        )
        for f in frames
    ]
    global _RELEASE_REGISTERED
    if not _RELEASE_REGISTERED:
        atexit.register(_FRAME_TEXTURES.clear)  # before sapien's teardown, as in table_video
        _RELEASE_REGISTERED = True
    _FRAME_TEXTURES[key] = textures
    while len(_FRAME_TEXTURES) > _FRAME_TEXTURES_KEPT:
        _FRAME_TEXTURES.popitem(last=False)
    return textures


def build_backdrop(env, spec: dict):
    """The frame quads for `env`'s parallel envs, merged into one `Actor` (frame-major: all envs'
    quads for frame 0, then frame 1, ...), and the frame count. Called from `_load_scene`."""
    import sapien
    from mani_skill.utils.structs.actor import Actor

    from custom_maniskill_tasks.appearance import _write_obj

    textures = _frame_textures(spec, float(env.control_freq))
    half_w, half_h = BACKDROP_WIDTH / 2, BACKDROP_HEIGHT / 2
    # in the quad's frame, centred: y runs along the wall, z up. Facing +x, the camera's right is
    # +y, so the image's left edge is at -y and its top at +z
    vertices = np.array([[0, -half_w, half_h], [0, half_w, half_h], [0, half_w, -half_h], [0, -half_w, -half_h]],
                        dtype=np.float32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    triangles = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32)  # counter-clockwise seen from +x
    normals = np.tile([1.0, 0.0, 0.0], (4, 1)).astype(np.float32)
    path = _write_obj(vertices, triangles, normals, uvs)
    glow = spec["brightness"]

    quads = []
    for k, texture in enumerate(textures):
        material = sapien.render.RenderMaterial(
            base_color=[1.0, 1.0, 1.0, 1.0], emission=[glow, glow, glow, 1.0], roughness=1.0, metallic=0.0, specular=0.0
        )
        material.base_color_texture = texture
        material.emission_texture = texture
        for i in range(env.num_envs):
            builder = env.scene.create_actor_builder()
            builder._procedural_shapes.append(
                sapien.render.RenderShapeTriangleMesh(path, np.ones(3, dtype=np.float32), material)
            )
            builder.set_scene_idxs([i])
            builder.initial_pose = sapien.Pose(backdrop_centre(parked=True))
            quads.append(builder.build_kinematic(name=f"backdrop_video_{k}_{i}"))
            # visual only, and its pose follows from the step counter: out of the state dict, which
            # would otherwise snapshot frames x num_envs actors on every get_state_dict() (see
            # table_video for the host-memory leak that caused)
            env.remove_from_state_dict_registry(quads[-1])
    return Actor.merge(quads, name="backdrop_video"), len(textures)


def backdrop_centre(parked: bool = False) -> list[float]:
    z = BACKDROP_BOTTOM + BACKDROP_HEIGHT / 2 - (BACKDROP_PARKED_DROP if parked else 0.0)
    return [BACKDROP_X, 0.0, z]


class BackdropVideoMixin:
    """Adds the video wall named by the `background_video` kwarg (see module doc); None for none."""

    def __init__(self, *args, background_video: str | Mapping[str, Any] | None = None, **kwargs):
        self._backdrop_spec = resolve_background_video(background_video)
        self._backdrop = None
        self._backdrop_offset = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._backdrop = None if self._backdrop_spec is None else build_backdrop(self, self._backdrop_spec)

    def _show_backdrop_frames(self, frame, env_idx=None) -> None:
        """Put frame `frame[j]` of each env in `env_idx` (all if None) on the wall and park the rest."""
        import torch
        from mani_skill.utils.structs.pose import Pose

        actor, count = self._backdrop
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        frame = torch.as_tensor(frame, device=self.device).long().reshape(-1).expand(len(env_idx))
        # rows are frame-major; Actor.set_pose writes the rows of the envs in `scene._reset_mask`
        # (the ones being reset inside `_initialize_episode`, every env during a step)
        shown = torch.arange(count, device=self.device)[:, None] == frame[None, :]
        on, off = backdrop_centre(), backdrop_centre(parked=True)
        z = torch.where(shown, on[2], off[2]).reshape(-1)
        p = torch.stack([torch.full_like(z, on[0]), torch.full_like(z, on[1]), z], dim=-1)
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(len(z), 4)
        actor.set_pose(Pose.create_from_pq(p, q))

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if self._backdrop is None:
            return
        import torch

        count = self._backdrop[1]
        if self._backdrop_offset is None or len(self._backdrop_offset) != self.num_envs:
            self._backdrop_offset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        seeds = np.asarray(self._episode_seed)[env_idx.cpu().numpy()]
        starts = [np.random.RandomState((int(seed) ^ 0xBAC4D209) % 2**32).randint(count) for seed in seeds]
        self._backdrop_offset[env_idx] = torch.as_tensor(starts, dtype=torch.long, device=self.device)
        self._show_backdrop_frames(self._backdrop_offset[env_idx], env_idx)

    def step(self, action):
        if self._backdrop is not None:
            # the frame this step returns is rendered after it, at elapsed step + 1; loops past the end
            self._show_backdrop_frames((self._backdrop_offset + self.elapsed_steps + 1) % self._backdrop[1])
            if self.scene.gpu_sim_enabled:
                # BaseEnv.step never applies rigid-body data (only reset does); see
                # PerturbationClassesMixin._apply_mid_episode_poses
                self.scene.px.gpu_apply_rigid_dynamic_data()
                self.scene.px.gpu_fetch_rigid_dynamic_data()
        return super().step(action)
