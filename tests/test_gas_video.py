"""DBC's ideal-gas distractor video, and `table_video`'s random-start loop (no simulator needed)."""

import numpy as np
import pytest

from custom_maniskill_tasks.gas_video import ideal_gas_positions, render_ideal_gas
from custom_maniskill_tasks.table_video import resolve_table_video


def test_particles_stay_in_the_box_and_move_in_straight_lines_between_bounces():
    pos = ideal_gas_positions(num_particles=10, frames=500, aspect=2.0, speed=0.03, seed=0)
    assert pos.shape == (500, 10, 2)
    assert (pos[..., 0] >= 0).all() and (pos[..., 0] <= 2.0).all()
    assert (pos[..., 1] >= 0).all() and (pos[..., 1] <= 1.0).all()
    step = np.abs(np.diff(pos, axis=0))
    assert step.max() <= 0.03 + 1e-9  # never faster than `speed` per axis
    # speed is conserved through bounces (elastic): per-axis |step| stays constant away from walls
    interior = (pos[1:-1] > 0.05).all(-1) & (pos[1:-1, :, 0] < 1.95) & (pos[1:-1, :, 1] < 0.95)
    assert np.allclose(step[1:][interior], step[:-1][interior])


def test_seeded():
    a = ideal_gas_positions(5, 50, 2.0, 0.02, seed=3)
    assert np.array_equal(a, ideal_gas_positions(5, 50, 2.0, 0.02, seed=3))
    assert not np.array_equal(a, ideal_gas_positions(5, 50, 2.0, 0.02, seed=4))


def test_render_writes_a_readable_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = render_ideal_gas(tmp_path / "gas.mp4", frames=12, width=128, height=64)
    capture = cv2.VideoCapture(str(path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(20)
    ok, frame = capture.read()
    assert ok and frame.shape == (64, 128, 3) and frame.max() > 0  # some particle drawn on black


def test_random_start_needs_a_loop_length():
    with pytest.raises(ValueError, match="frames"):
        resolve_table_video({"path": "x.mp4", "random_start": True})
    with pytest.raises(ValueError, match="random_start"):
        resolve_table_video({"path": "x.mp4", "frames": 100})
    spec = resolve_table_video({"path": "x.mp4", "random_start": True, "frames": 100, "alpha": 0.5})
    assert spec["random_start"] is True and spec["frames"] == 100
    assert resolve_table_video({"path": "x.mp4"})["random_start"] is False
