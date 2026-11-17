"""How the `object_colour`, `object_hue` and `table_texture` perturbation classes change the scene,
and the table tint `table_colour` stands for. The classes themselves -- their aspects, ranges and splits -- are
in `perturbation_classes`; `PerturbationClassesMixin` calls these.

Both appliers edit render materials the scene has already built, so they run from `_load_scene`,
before the scene first renders: a material edited after that is stored but never reaches the
renderer.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from custom_maniskill_tasks.lighting import _linear_to_srgb, _remapped_texture, _srgb_to_linear
from custom_maniskill_tasks.texture import _render_parts


def table_tint(condition: dict[str, float]) -> tuple[float, float, float]:
    """The `table_tint` that `table_colour`'s drawn `condition` stands for."""
    return tuple(float(condition[c] * condition["brightness"]) for c in "rgb")


def _apply_object_colour(env, offsets: Sequence[float]) -> None:
    """Add `offsets` (r, g, b) to the colour of every render part of `env`'s object of interest,
    clipped to [0, 1]. Every original colour is read before any is written and each result is a
    function of its original, so a material shared between parallel envs is shifted once, not once
    per env. A textured object (a YCB model) has its texture shifted the same way, into a new one
    (see `_remapped_texture`)."""
    attribute = getattr(env, "object_of_interest", None)
    if attribute is None:
        raise ValueError(
            f"{type(env).__name__} has no single object of interest, so 'object_colour' has no "
            "object to recolour."
        )
    offsets = np.asarray(offsets, dtype=np.float64)
    parts = _render_parts(getattr(env, attribute))
    edits = [(part.material, list(part.material.base_color), part.material.base_color_texture) for part in parts]
    if not edits:
        raise ValueError(f"the object {attribute!r} has no render parts, so there is nothing to recolour.")
    shifted = [
        (
            material,
            [*np.clip(np.asarray(base_color[:3]) + offsets, 0.0, 1.0).tolist(), base_color[3]],
            None if texture is None else _remapped_texture(
                texture, lambda c: _srgb_to_linear(np.clip(_linear_to_srgb(c) + offsets, 0.0, 1.0))
            ),
        )
        for material, base_color, texture in edits
    ]
    for material, base_color, texture in shifted:
        material.base_color = base_color
        if texture is not None:
            material.base_color_texture = texture


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """`colorsys.rgb_to_hsv` over an (..., 3) array, hue in [0, 1)."""
    r, g, b = np.moveaxis(rgb, -1, 0)
    value = rgb.max(axis=-1)
    delta = value - rgb.min(axis=-1)
    safe = np.where(delta > 0, delta, 1.0)
    hue = np.select(
        [delta == 0, value == r, value == g],
        [0.0, ((g - b) / safe) % 6.0, (b - r) / safe + 2.0],
        default=(r - g) / safe + 4.0,
    ) / 6.0
    saturation = np.where(value > 0, delta / np.where(value > 0, value, 1.0), 0.0)
    return np.stack([hue % 1.0, saturation, value], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """`colorsys.hsv_to_rgb` over an (..., 3) array."""
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    sector = np.floor(hue * 6.0)
    f = hue * 6.0 - sector
    p, q, t = value * (1 - saturation), value * (1 - saturation * f), value * (1 - saturation * (1 - f))
    sector = sector.astype(int) % 6
    channels = [
        np.choose(sector, [value, q, p, p, t, value]),
        np.choose(sector, [t, value, value, q, p, p]),
        np.choose(sector, [p, p, t, value, value, q]),
    ]
    return np.stack(channels, axis=-1)


def _rotate_hue(linear_rgb: np.ndarray, degrees: float) -> np.ndarray:
    """`linear_rgb` (..., 3) turned `degrees` round the HSV hue wheel of its sRGB encoding, the
    space "hue" is ordinarily meant in (red -> yellow -> green -> blue at 0 -> 60 -> 120 -> 240).
    Saturation and value are kept, so greys stay grey."""
    hsv = _rgb_to_hsv(_linear_to_srgb(np.clip(linear_rgb, 0.0, 1.0)))
    hsv[..., 0] = (hsv[..., 0] + degrees / 360.0) % 1.0
    return _srgb_to_linear(_hsv_to_rgb(hsv))


def _apply_object_hue(env, degrees: float) -> None:
    """Turn the colours of every task object in `env` `degrees` round the hue wheel: each scene
    actor but the table scene's own (table and ground), the lighting props and the distractor
    robots' props (`distractor_robots`), so the cube, the
    goal target and site, the peg -- whatever the task built. The robot is an articulation, not an
    actor, and is untouched.

    Every original colour is read before any is written, and each result is a function of its
    original, so a material shared between parallel envs is rotated once, not once per env; a
    textured object gets a new, rotated texture (see `_remapped_texture`), not an in-place upload.
    """
    import sapien

    if degrees % 360.0 == 0.0:
        return
    table_scene = getattr(env, "table_scene", None)
    fixtures = set() if table_scene is None else {obj.name for obj in table_scene.scene_objects}
    edits = []
    for name, actor in env.scene.actors.items():
        # distractor_robots' cubes and pads are scenery, not task objects
        if name in fixtures or name.startswith(("lighting_prop_", "distractor_")):
            continue
        for entity in actor._objs:
            body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
            if body is None:
                continue
            for shape in body.render_shapes:
                for part in shape.parts:
                    material = part.material
                    edits.append((material, list(material.base_color), material.base_color_texture))
    if not edits:
        raise ValueError(
            f"{type(env).__name__} has no task objects to recolour, so an object_hue of {degrees} "
            "would render the unshifted scene."
        )
    rotated = [
        (
            material,
            [*_rotate_hue(np.asarray(base_color[:3]), degrees).tolist(), base_color[3]],
            None if texture is None else _remapped_texture(texture, lambda c: _rotate_hue(c, degrees)),
        )
        for material, base_color, texture in edits
    ]
    for material, base_color, texture in rotated:
        material.base_color = base_color
        if texture is not None:
            material.base_color_texture = texture


def transform_pixels(
    pixels: np.ndarray, scale: float, shear: float, rotation: float, shift_x: float, shift_y: float
) -> np.ndarray:
    """`pixels` (h, w, c uint8) tiled infinitely and then scaled by `scale` about its centre,
    sheared in x by `shear`, rotated by `rotation` degrees and shifted by (`shift_x`, `shift_y`)
    periods, sampled bilinearly back onto the same grid. The identity at (1, 0, 0, 0, 0)."""
    h, w = pixels.shape[:2]
    theta = np.radians(rotation)
    rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    forward = rotate @ np.array([[1.0, shear], [0.0, 1.0]]) @ np.diag([scale, scale])
    inverse = np.linalg.inv(forward)
    x, y = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    source = (np.stack([x - 0.5 - shift_x, y - 0.5 - shift_y], axis=-1) @ inverse.T) + 0.5
    fx, fy = source[..., 0] * w - 0.5, source[..., 1] * h - 0.5
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = (fx - x0)[..., None], (fy - y0)[..., None]
    at = lambda yy, xx: pixels[yy % h, xx % w].astype(np.float64)
    top = at(y0, x0) * (1 - tx) + at(y0, x0 + 1) * tx
    bottom = at(y0 + 1, x0) * (1 - tx) + at(y0 + 1, x0 + 1) * tx
    return np.clip(np.round(top * (1 - ty) + bottom * ty), 0, 255).astype(np.uint8)


def _apply_table_texture(env, condition: dict[str, float]) -> None:
    """Move the table's textures as `condition` says. Every parallel env's table is the same model,
    so each texture is worked out once, from the first env's, and assigned to all of them (as
    `lighting._tint_table` does), which also keeps the transform from compounding on a texture
    they share. A part with no texture (a plain-coloured material) has nothing to move."""
    import sapien

    table_scene = getattr(env, "table_scene", None)
    if table_scene is None:
        raise ValueError(f"{type(env).__name__} has no `table_scene`, so there is no table texture to move.")
    entities = table_scene.table._objs

    def parts(entity):
        body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        return [part for shape in body.render_shapes for part in shape.parts]

    template = parts(entities[0])
    moved = []
    for part in template:
        texture = part.material.base_color_texture
        if texture is None:
            moved.append(None)
            continue
        pixels = texture.download()
        out = pixels.copy()
        out[..., :3] = transform_pixels(pixels[..., :3], **condition)
        moved.append(sapien.render.RenderTexture2D(
            out, texture.format, texture.mipmap_levels, texture.filter_mode,
            texture.address_mode, texture.is_srgb,
        ))
    for entity in entities:
        for part, texture in zip(parts(entity), moved):
            if texture is not None:
                part.material.base_color_texture = texture


TABLE_SPIN_LIFT = 5e-4
"""How far above the table top the spinning disc lies, in metres: enough to win the depth test,
far too little to see."""


def _table_top_uv_map(env):
    """(affine uv map, top height, (x range, y range), normal) of `env`'s table top, in world frame.

    The map is the 2x3 matrix taking world (x, y, 1) to the texture coordinates the table's own mesh
    gives its top face there, fitted exactly on the largest triangle of that face (the top's UVs
    are affine in position, so three points fix them). Read off the first env's table: every
    parallel env's table is the same model at the same pose relative to its own scene."""
    import sapien

    table_scene = getattr(env, "table_scene", None)
    if table_scene is None:
        raise ValueError(f"{type(env).__name__} has no `table_scene`, so there is no table top to spin.")
    entity = table_scene.table._objs[0]
    body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
    textured = [(shape, part) for shape in body.render_shapes for part in shape.parts
                if part.material.base_color_texture is not None]
    if len(textured) != 1:
        raise AssertionError(f"expected the table to have one textured part, found {len(textured)}")
    shape, part = textured[0]
    to_world = (entity.pose * shape.local_pose).to_transformation_matrix()
    local = part.get_vertices() * np.asarray(shape.scale)
    world = local @ to_world[:3, :3].T + to_world[:3, 3]
    uv = part.get_vertex_uv()
    top_z = world[:, 2].max()
    on_top = np.abs(world[:, 2] - top_z) < 1e-5
    triangles = [t for t in part.get_triangles() if on_top[t].all()]
    def area(t):
        (ax, ay), (bx, by) = world[t[1], :2] - world[t[0], :2], world[t[2], :2] - world[t[0], :2]
        return abs(ax * by - ay * bx)

    largest = max(triangles, key=area)
    xy1 = np.c_[world[largest, :2], np.ones(3)]
    uv_map = np.linalg.solve(xy1, uv[largest]).T  # (2, 3): uv = uv_map @ (x, y, 1)
    # The model's top-face normals are not straight up (they lean, in world frame), and its shading
    # -- the highlight in particular -- comes from them, so the disc copies them rather than using +z.
    normals = part.get_vertex_normal()[largest] @ to_world[:3, :3].T
    normal = normals.mean(axis=0)
    normal /= np.linalg.norm(normal)
    top = world[on_top]
    extent = ((top[:, 0].min(), top[:, 0].max()), (top[:, 1].min(), top[:, 1].max()))
    return uv_map, float(top_z), extent, normal


def _write_obj(vertices, triangles, normals, uvs) -> str:
    """The mesh as a Wavefront OBJ in a cache file named for its contents, so every env and every
    rebuild reuses one file."""
    import hashlib
    import tempfile
    from pathlib import Path

    lines = [f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in vertices]
    lines += [f"vt {u:.7f} {1.0 - v:.7f}" for u, v in uvs]  # OBJ v runs bottom-up; the loader flips it back
    lines += [f"vn {x:.7f} {y:.7f} {z:.7f}" for x, y, z in normals]
    lines += ["f " + " ".join(f"{i + 1}/{i + 1}/{i + 1}" for i in t) for t in triangles]
    text = "\n".join(lines) + "\n"
    path = Path(tempfile.gettempdir()) / f"table_spin_disc_{hashlib.sha1(text.encode()).hexdigest()[:16]}.obj"
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
    return str(path)


def build_table_spin_disc(env, segments: int = 256):
    """A disc lying on `env`'s table top, painted with the table's own texture at exactly the
    texture coordinates the table has there, so at angle 0 it is invisible; turning it about its
    centre turns the table's texture. One kinematic, visual-only actor per parallel env (no
    collision shapes, so nothing on the table touches or is dragged by it), merged into one
    `Actor`, posed at angle 0.

    Its centre is the middle of the table top in x and y, and its radius half the top's shorter
    side, so it reaches the table's near and far edges and never overhangs them; beyond it in the
    long direction the table's own, still texture shows.

    A disc rather than spinning the texture itself: once a scene has rendered, neither a new texture
    nor new UVs reach the renderer ("mesh already on device"), on physx_cpu or physx_cuda. A moving
    actor's pose does, on both.
    """
    import sapien
    from mani_skill.utils.structs.actor import Actor

    uv_map, top_z, ((x0, x1), (y0, y1)), normal = _table_top_uv_map(env)
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    radius = min(x1 - x0, y1 - y0) / 2 - 1e-3
    angles = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    ring = radius * np.c_[np.cos(angles), np.sin(angles)]
    local_xy = np.vstack([[0.0, 0.0], ring])
    vertices = np.c_[local_xy, np.zeros(len(local_xy))].astype(np.float32)
    triangles = np.array([[0, 1 + i, 1 + (i + 1) % segments] for i in range(segments)], dtype=np.uint32)
    normals = np.tile(normal, (len(local_xy), 1)).astype(np.float32)
    uvs = (np.c_[local_xy + centre, np.ones(len(local_xy))] @ uv_map.T).astype(np.float32)

    entity = env.table_scene.table._objs[0]
    body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
    material = next(part.material for shape in body.render_shapes for part in shape.parts
                    if part.material.base_color_texture is not None)

    # Loaded from a file rather than built from the arrays: sapien works out the tangents a mesh's
    # normal map needs only when it loads one (the table's material has a normal map), and without
    # them the disc shades visibly darker than the table it covers.
    path = _write_obj(vertices, triangles, normals, uvs)
    pose = sapien.Pose([centre[0], centre[1], top_z + TABLE_SPIN_LIFT])
    discs = []
    for i in range(env.num_envs):
        builder = env.scene.create_actor_builder()
        builder._procedural_shapes.append(
            sapien.render.RenderShapeTriangleMesh(path, np.ones(3, dtype=np.float32), material)
        )
        builder.set_scene_idxs([i])
        builder.initial_pose = pose
        discs.append(builder.build_kinematic(name=f"table_spin_disc_{i}"))
    return Actor.merge(discs, name="table_spin_disc"), pose


OBJECT_TEXTURE_SIZE = 256


def _tileable_noise(octaves, size=OBJECT_TEXTURE_SIZE):
    from custom_maniskill_tasks.texture import pattern_mask

    total = sum(pattern_mask("noise", n, size) / (i + 1) for i, n in enumerate(octaves))
    return (total - total.min()) / max(total.max() - total.min(), 1e-9)


# Every face of a box part maps the whole texture once, and a peg face is only some 30 x 70 pixels
# in the 224 px wrist camera, so these patterns are drawn far coarser than the real thing (real
# print layers are ~0.2 mm): fine detail would be averaged away by the texture filtering and the
# camera would see the plain colour.


def _wood_grain(size=OBJECT_TEXTURE_SIZE):
    """Four or five wavering grain bands across the texture's rows, plus a little fibre."""
    axis = (np.arange(size) + 0.5) / size
    _, y = np.meshgrid(axis, axis)
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * (4 * y + 1.0 * _tileable_noise((2, 4), size)))
    fibre = _tileable_noise((16, 32), size)
    return 0.5 + 0.42 * rings ** 2 + 0.08 * fibre


def _paper_fibre(size=OBJECT_TEXTURE_SIZE):
    """Blotchy, matte mottling, the look of crumpled paper or card."""
    return 0.62 + 0.28 * _tileable_noise((4, 8, 16), size) + 0.1 * _tileable_noise((32,), size)


def _print_layers(size=OBJECT_TEXTURE_SIZE):
    """3D-print layer lines: eight rounded ridges per face, dark in their valleys."""
    axis = (np.arange(size) + 0.5) / size
    _, y = np.meshgrid(axis, axis)
    ridge = np.abs(np.sin(np.pi * 8 * y)) ** 0.5
    return 0.55 + 0.45 * ridge


OBJECT_TEXTURES = {
    # name: (brightness pattern, roughness, specular)
    "wood": (_wood_grain, 0.6, 0.3),
    "paper": (_paper_fibre, 1.0, 0.1),
    "printed_plastic": (_print_layers, 0.35, 0.6),
}
"""Patterns for the `object_texture` class. Each is a field of reflectance multipliers (about 0.5 to
1) that `_apply_object_texture` multiplies into every render part's own colour, so a red-and-blue
object stays red and blue, now with grain, fibre or layer lines; plus the surface's roughness and
specular."""


def _apply_object_texture(env, name: str) -> None:
    """Give every render part of `env`'s object of interest the `name` pattern in its own colour.

    Each part's colour (its `base_color`, linear) becomes a texture: colour times the pattern, in
    linear light, sRGB-encoded. The material's `base_color` is set to white, since sapien would
    otherwise multiply it in again. The pattern repeats once per face of a box part (sapien maps
    each face to the whole texture). Applied from `_load_scene`, before the scene first renders,
    after `object_colour` and `object_hue` have set the colours. A part that already has a texture
    (a YCB model) is refused rather than silently overwritten."""
    import sapien

    attribute = getattr(env, "object_of_interest", None)
    if attribute is None:
        raise ValueError(f"{type(env).__name__} has no single object of interest, so 'object_texture' has no object.")
    pattern, roughness, specular = OBJECT_TEXTURES[name]
    field = pattern()
    parts = _render_parts(getattr(env, attribute))
    if not parts:
        raise ValueError(f"the object {attribute!r} has no render parts, so there is nothing to texture.")
    textures = {}  # one texture per distinct colour, shared across parallel envs
    for part in parts:
        material = part.material
        if material.base_color_texture is not None:
            raise ValueError(f"{attribute!r} already has a texture; object_texture only patterns plain-coloured parts")
        colour = tuple(round(c, 6) for c in material.base_color[:3])
        if colour not in textures:
            rgb = _linear_to_srgb(np.clip(field[..., None] * np.asarray(colour), 0.0, 1.0))
            pixels = np.empty(field.shape + (4,), dtype=np.uint8)
            pixels[..., :3] = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
            pixels[..., 3] = 255
            textures[colour] = sapien.render.RenderTexture2D(pixels, "R8G8B8A8Unorm", 4, "linear", "repeat", True)
    for part in parts:
        material = part.material
        texture = textures.get(tuple(round(c, 6) for c in material.base_color[:3]))
        if texture is None:  # a material shared with a part already done
            continue
        material.base_color_texture = texture
        material.base_color = [1.0, 1.0, 1.0, material.base_color[3]]
        material.roughness = roughness
        material.specular = specular
