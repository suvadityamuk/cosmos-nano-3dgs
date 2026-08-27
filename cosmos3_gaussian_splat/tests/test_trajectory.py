import numpy as np

from cosmos3_gsplat.config import TrajectoryConfig
from cosmos3_gsplat.trajectory import (
    actions_to_poses,
    make_closed_helical_trajectory,
    poses_to_actions,
    rot6d_to_rotation_matrix,
    rotation_matrix_to_rot6d,
)


def test_closed_trajectory_and_expected_shape() -> None:
    trajectory = make_closed_helical_trajectory(TrajectoryConfig())
    assert trajectory.poses_c2w.shape == (61, 4, 4)
    assert trajectory.raw_actions.shape == (60, 9)
    np.testing.assert_allclose(trajectory.poses_c2w[0], trajectory.poses_c2w[-1], atol=1e-5)
    np.testing.assert_allclose(trajectory.elevation_deg[[0, -1]], 0.0, atol=1e-5)
    assert trajectory.elevation_deg.max() > 11.0
    assert trajectory.elevation_deg.min() < -11.0


def test_pose_action_round_trip() -> None:
    trajectory = make_closed_helical_trajectory(TrajectoryConfig(radius_m=2.0, num_actions=24))
    reconstructed = actions_to_poses(trajectory.raw_actions, trajectory.poses_c2w[0])
    np.testing.assert_allclose(reconstructed, trajectory.poses_c2w, atol=2e-5)
    np.testing.assert_allclose(poses_to_actions(reconstructed), trajectory.raw_actions, atol=2e-5)


def test_rot6d_column_convention_round_trip() -> None:
    trajectory = make_closed_helical_trajectory(TrajectoryConfig(num_actions=8))
    rotations = trajectory.poses_c2w[:, :3, :3]
    decoded = rot6d_to_rotation_matrix(rotation_matrix_to_rot6d(rotations))
    np.testing.assert_allclose(decoded, rotations, atol=1e-6)
    determinants = np.linalg.det(decoded)
    np.testing.assert_allclose(determinants, 1.0, atol=1e-6)
