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
# --------------------------------------------------------------------------
#  mesh-guided initialization extension
# --------------------------------------------------------------------------
#  Keeps the original `MeshPointCloud` 100% backwards-compatible. New fields
#  used by the mesh-guided initialization are stored on a separate
#  `MeshGuidedPointCloud` NamedTuple.
# --------------------------------------------------------------------------

import numpy as np
import torch
from typing import NamedTuple, Optional


class MeshPointCloud(NamedTuple):
    """Original GaMeS point cloud container. Unchanged for backward compat."""
    alpha: torch.Tensor
    points: torch.Tensor
    colors: np.array
    normals: np.array
    vertices: np.array
    faces: np.array
    transform_vertices_function: object
    triangles: torch.Tensor


class MeshGuidedPointCloud(NamedTuple):
    """
    Extended container produced by `mesh_guided_init.build_mesh_guided_point_cloud`.

    Fields
    ------
    points           : (M, 3) torch.FloatTensor        -- xyz of Gaussians
    colors           : (M, 3) numpy float32 in [0,1]  -- initial RGB
    normals          : (M, 3) torch.FloatTensor        -- surface normals
    face_id          : (M,)   torch.LongTensor         -- nearest face id
    barycentric      : (M, 3) torch.FloatTensor        -- face barycentric (seed)
    complexity       : (M,)   torch.FloatTensor in[0,1]-- per-Gaussian appearance complexity
    scale_init       : (M, 3) torch.FloatTensor        -- adaptive initial 3D scale
    rotation_init    : (M, 4) torch.FloatTensor        -- initial rotation quat (w,x,y,z)
    vertices         : (V, 3) torch.FloatTensor        -- the mesh itself (kept for ref)
    faces            : (F, 3) torch.LongTensor         -- mesh faces
    triangles        : (F, 3, 3) torch.FloatTensor     -- mesh triangle vertices
    seed_idx         : (M,)   torch.LongTensor         -- which seed point produced each Gaussian
    """
    points: torch.Tensor
    colors: np.array
    normals: torch.Tensor
    face_id: torch.Tensor
    barycentric: torch.Tensor
    complexity: torch.Tensor
    scale_init: torch.Tensor
    rotation_init: torch.Tensor
    vertices: torch.Tensor
    faces: torch.Tensor
    triangles: torch.Tensor
    seed_idx: torch.Tensor

    def __len__(self):
        return self.points.shape[0]
