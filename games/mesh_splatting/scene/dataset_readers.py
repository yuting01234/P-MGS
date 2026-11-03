#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

import os
import numpy as np
import trimesh
import torch

from games.mesh_splatting.utils.graphics_utils import MeshPointCloud
from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    readCamerasFromTransforms,
    getNerfppNorm,
    SceneInfo,
    storePly,
)
from utils.sh_utils import SH2RGB
from games.mesh_splatting.init import build_mesh_guided_point_cloud

softmax = torch.nn.Softmax(dim=2)


def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices


def readNerfSyntheticMeshInfo(
        path, white_background, eval, num_splats, extension=".png"
):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    print("Reading Mesh object")
    mesh_scene = trimesh.load(f'{path}/mesh.obj', force='mesh')
    vertices = mesh_scene.vertices
    vertices = transform_vertices_function(
        torch.tensor(vertices),
    )
    faces = mesh_scene.faces
    triangles = vertices[torch.tensor(mesh_scene.faces).long()].float()

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    if True:
        # Since this data set has no colmap data, we start with random points
        num_pts_each_triangle = num_splats
        num_pts = num_pts_each_triangle * triangles.shape[0]
        print(
            f"Generating random point cloud ({num_pts})..."
        )

        # We create random points inside the bounds traingles
        alpha = torch.rand(
            triangles.shape[0],
            num_pts_each_triangle,
            3
        )

        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0

        pcd = MeshPointCloud(
            alpha=alpha,
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            transform_vertices_function=transform_vertices_function,
            triangles=triangles.cuda()
        )

        storePly(ply_path, pcd.points, SH2RGB(shs) * 255)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfSyntheticMeshGuidedInfo(
        path, white_background, eval, extension=".png",
        *,
        n_seed=100000,
        num_per_face=8,
        complexity_power=1.0,
        noise_scale_ratio=0.02, base_size=0.001,
        sobel_weight=0.5, var_weight=0.5, var_kernel=5,
        visibility_power=1.0,
        seed=0,
        sampler="adaptive",
        visual_n_seed=300000,
        patch2d_n_seed=300000,
        patch2d_patch_size=5,
        patch2d_complexity_power=1.0,
        patch2d_visual_area_power=0.5,
        patch2d_min_coverage=3,
        patch2d_stage1_extra_factor=2.0,
        use_comp_scale_shrink=True,
        comp_shrink_alpha=0.5,
        scale_jitter_lo=None,
        scale_jitter_hi=None,
        weight_mode="power",
        weight_alpha=0.5,
        rotation_mode="face_tangent",
        jitter_mode="tangent",
):
    """Mesh-guided Blendeɾ initialization (complexity-aware)."""
    is_dtu = os.path.exists(os.path.join(path, "sfm_scene.json"))
    if is_dtu:
        from scene.dtu_dataset import readDTUCameras
        train_cam_infos, test_cam_infos = readDTUCameras(path, white_background)
    else:
        print("[MeshGuided] Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
        print("[MeshGuided] Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    print("[MeshGuided] Loading Mesh object")
    mesh_file = "mesh_world.ply" if is_dtu else "mesh.obj"
    mesh_scene = trimesh.load(os.path.join(path, mesh_file), force="mesh")
    vertices = mesh_scene.vertices
    # Prepared DTU meshes are already in the calibrated IDR world frame.
    vertices = torch.tensor(vertices) if is_dtu else transform_vertices_function(torch.tensor(vertices))
    faces    = mesh_scene.faces
    triangles = vertices[torch.tensor(mesh_scene.faces).long()].float()

    mesh_mv = mesh_scene.copy()
    mesh_mv.vertices = vertices.detach().cpu().numpy()

    init_dict = build_mesh_guided_point_cloud(
        mesh_mv, train_cam_infos,
        n_seed=n_seed,
        num_per_face=num_per_face,
        complexity_power=complexity_power,
        noise_scale_ratio=noise_scale_ratio,
        base_size=base_size,
        sobel_weight=sobel_weight,
        var_weight=var_weight,
        var_kernel=var_kernel,
        visibility_power=visibility_power,
        sampler=sampler,
        visual_n_seed=visual_n_seed,
        patch2d_n_seed=patch2d_n_seed,
        patch2d_patch_size=patch2d_patch_size,
        patch2d_complexity_power=patch2d_complexity_power,
        patch2d_visual_area_power=patch2d_visual_area_power,
        patch2d_min_coverage=patch2d_min_coverage,
        patch2d_stage1_extra_factor=patch2d_stage1_extra_factor,
        use_comp_scale_shrink=use_comp_scale_shrink,
        comp_shrink_alpha=comp_shrink_alpha,
        scale_jitter_lo=scale_jitter_lo,
        scale_jitter_hi=scale_jitter_hi,
        weight_mode=weight_mode,
        weight_alpha=weight_alpha,
        rotation_mode=rotation_mode,
        jitter_mode=jitter_mode,
        seed=seed,
    )

    ply_path = os.path.join(path, "points3d.ply")
    storePly(ply_path, init_dict["xyz"].cpu().numpy(),
             (init_dict["colors"] * 255.0).astype(np.uint8))

    # Legacy MeshPointCloud placeholder (only `points`, `colors`, `normals`,
    # `vertices`, `faces`, `triangles` are actually used downstream).
    pcd = MeshPointCloud(
        alpha=-1,
        points=init_dict["xyz"],
        colors=init_dict["colors"],
        normals=init_dict["normal"].numpy(),
        vertices=init_dict["vertices"],
        faces=init_dict["faces"].cpu().numpy(),
        transform_vertices_function=None if is_dtu else transform_vertices_function,
        triangles=triangles.cuda(),
    )

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_mesh_guided=True,
    )

    # Stash the init_dict on a module-level registry. The downstream Scene
    # constructor retrieves it via the same key (ply_path) so we never need to
    # bolt arbitrary attributes onto a NamedTuple.
    _mesh_guided_registry[ply_path] = init_dict
    return scene_info


# Module-level registry so we can hand the full mesh-guided init dict from
# `readNerfSyntheticMeshGuidedInfo` to `Scene.__init__` without changing the
# `SceneInfo` NamedTuple schema.
_mesh_guided_registry: dict = {}


def take_mesh_guided_init(ply_path):
    return _mesh_guided_registry.pop(ply_path, None)


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Blender_Mesh": readNerfSyntheticMeshInfo,
    "Blender_Mesh_Guided": readNerfSyntheticMeshGuidedInfo,
}
