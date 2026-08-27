from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from cosmos3_gsplat.config import GeometryConfig
from cosmos3_gsplat.reconstruction import ReconstructionResult, build_reconstruction
from cosmos3_gsplat.vggt_backend import VGGTGeometryResult, _pycolmap_image_extrinsic


def _synthetic_geometry(tmp_path: Path) -> VGGTGeometryResult:
    image_paths = []
    for index in range(3):
        path = tmp_path / "source" / f"{index}.png"
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (32, 32), (index * 30, 100, 150)).save(path)
        image_paths.append(path)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    poses[:, 0, 3] = [-1.0, 0.0, 1.0]
    intrinsics = np.repeat(np.array([[[20, 0, 16], [0, 20, 16], [0, 0, 1]]], dtype=np.float32), 3, axis=0)
    result = VGGTGeometryResult(
        keyframe_indices=np.arange(3),
        keyframe_paths=tuple(image_paths),
        training_image_paths=tuple(image_paths),
        accepted_mask=np.array([True, False, True]),
        commanded_poses_c2w=poses,
        predicted_poses_c2w=poses,
        aligned_poses_c2w=poses,
        intrinsics=intrinsics,
        depths=np.ones((3, 32, 32), dtype=np.float32),
        depth_confidence=np.ones((3, 32, 32), dtype=np.float32),
        points=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        colors=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
        metrics={"sim3_scale": 1.0, "depth_confidence_threshold": 0.5},
        root=tmp_path / "geometry",
    )
    result.write()
    return result


def test_geometry_persistence_and_colmap_export(tmp_path: Path) -> None:
    geometry = _synthetic_geometry(tmp_path)
    loaded = VGGTGeometryResult.read(geometry.root)
    np.testing.assert_allclose(loaded.aligned_poses_c2w, geometry.aligned_poses_c2w)
    assert loaded.training_image_paths == geometry.training_image_paths

    config = replace(GeometryConfig(), run_colmap_diagnostic=False)
    reconstruction = build_reconstruction(loaded, config, tmp_path / "reconstruction")
    assert len(reconstruction.image_paths) == 2
    assert (reconstruction.colmap_dir / "cameras.txt").is_file()
    assert (reconstruction.colmap_dir / "images.txt").is_file()
    assert (reconstruction.colmap_dir / "points3D.txt").is_file()

    reloaded = ReconstructionResult.read(reconstruction.root)
    np.testing.assert_allclose(reloaded.poses_c2w, reconstruction.poses_c2w)


def test_pycolmap_transform_property_and_method_compatibility() -> None:
    matrix = np.eye(4)
    property_image = SimpleNamespace(cam_from_world=SimpleNamespace(matrix=matrix))
    method_image = SimpleNamespace(cam_from_world=lambda: SimpleNamespace(matrix=lambda: matrix))
    np.testing.assert_allclose(_pycolmap_image_extrinsic(property_image), matrix[:3])
    np.testing.assert_allclose(_pycolmap_image_extrinsic(method_image), matrix[:3])
