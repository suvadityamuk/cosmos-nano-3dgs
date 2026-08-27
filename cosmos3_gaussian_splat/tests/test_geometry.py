from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from cosmos3_gsplat.geometry import (
    invert_poses,
    robust_similarity_alignment,
    rotation_geodesic_degrees,
    select_keyframes,
)


def test_robust_similarity_recovers_transform_with_outlier() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(20, 3))
    rotation = Rotation.from_euler("xyz", [0.2, -0.4, 0.1]).as_matrix()
    scale = 2.25
    translation = np.array([4.0, -1.5, 0.25])
    target = scale * (source @ rotation.T) + translation
    target += rng.normal(scale=1e-4, size=target.shape)
    target[-1] += 10.0

    transform = robust_similarity_alignment(source, target, residual_threshold=0.01, max_trials=512)

    assert transform.inliers.sum() == 19
    assert abs(transform.scale - scale) < 1e-3
    np.testing.assert_allclose(transform.rotation, rotation, atol=1e-3)
    np.testing.assert_allclose(transform.translation, translation, atol=1e-3)


def test_rotation_geodesic() -> None:
    identity = np.eye(3)[None]
    quarter_turn = Rotation.from_euler("z", 90, degrees=True).as_matrix()[None]
    np.testing.assert_allclose(rotation_geodesic_degrees(identity, quarter_turn), [90.0], atol=1e-6)


def test_invert_vggt_three_by_four_extrinsics() -> None:
    w2c = np.eye(4)[None]
    w2c[0, :3, 3] = [1.0, 2.0, 3.0]
    c2w = invert_poses(w2c[:, :3])
    assert c2w.shape == (1, 4, 4)
    np.testing.assert_allclose(c2w[0, :3, 3], [-1.0, -2.0, -3.0])


def test_keyframes_keep_closure_and_filter_blurry_images(tmp_path: Path) -> None:
    paths = []
    rng = np.random.default_rng(0)
    for index in range(10):
        values = (
            np.full((64, 64), 128, dtype=np.uint8) if index == 5 else rng.integers(0, 255, (64, 64), dtype=np.uint8)
        )
        path = tmp_path / f"{index:02d}.png"
        Image.fromarray(values).save(path)
        paths.append(path)

    indices, scores = select_keyframes(paths, count=6, blur_threshold=1.0)

    assert indices[0] == 0
    assert indices[-1] == 9
    assert 5 not in indices
    assert scores[5] == 0.0
