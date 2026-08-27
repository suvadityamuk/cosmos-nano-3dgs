import numpy as np

from cosmos3_gsplat.camera_diagnostics import SHALLOW_YAW_ROT6D, _action_metrics


def test_shallow_arc_actions_remain_inside_reference_envelope() -> None:
    lower = np.asarray(
        [-1.2, -0.21, -0.09, 0.99998, -0.0015, -0.0062, -0.0015, 0.99999, -0.0012],
        dtype=np.float32,
    )
    upper = np.asarray(
        [-0.58, -0.01, 0.43, 1.0, 0.0015, -0.0032, 0.0015, 1.0, 0.0012],
        dtype=np.float32,
    )
    reference = np.stack([lower, upper] * 30)
    shallow = reference.copy()
    shallow[:, 3:] = SHALLOW_YAW_ROT6D
    outside = (shallow < reference.min(axis=0)) | (shallow > reference.max(axis=0))
    metrics = _action_metrics(shallow)
    assert not outside.any()
    np.testing.assert_array_equal(shallow[:, :3], reference[:, :3])
    assert 14.9 < metrics["rotation_deg_total"] < 15.1
    assert 0.249 < metrics["rotation_deg_mean"] < 0.251
