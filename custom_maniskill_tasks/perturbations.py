"""The scene and camera perturbations a task can take, as one mixin.

Alongside `lighting` (`LightingMixin`), a task takes:

- `texture` -- patterns on the table and objects (`texture`);
- `table_material` -- the table made to look like marble, glass, metal... (`table_material`);
- `distractors` -- YCB objects scattered in the wrist camera's view (`distractors`);
- `viewpoint` -- the wrist camera moved, turned or zoomed (`viewpoint`);
- `corruption` -- the camera images blurred, noised or degraded (`corruption`).

Each is an independent env kwarg with its own "default" that changes nothing, so any combination
can be asked for, and each is recorded into trajectories collected under it.

The order matters. `LightingMixin` goes ahead of this in a task's bases, since its `_load_scene`
recolours what the mixins after it have built (an `object-hue-<degrees>` shift turns the
distractors and the pattern under them too). Within this, `TextureMixin` goes ahead of
`DistractorsMixin` for the same reason: the distractors have to exist to be textured. It also goes
ahead of `TableMaterialMixin`, so a pattern is drawn over the material's colour, not under it.
"""

from __future__ import annotations

from custom_maniskill_tasks.corruption import CorruptionMixin
from custom_maniskill_tasks.distractors import DistractorsMixin
from custom_maniskill_tasks.table_material import TableMaterialMixin
from custom_maniskill_tasks.texture import TextureMixin
from custom_maniskill_tasks.viewpoint import ViewpointMixin


class PerturbationsMixin(TextureMixin, TableMaterialMixin, DistractorsMixin, ViewpointMixin, CorruptionMixin):
    """`texture`, `table_material`, `distractors`, `viewpoint` and `corruption`, in that order."""
