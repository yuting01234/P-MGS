#
# Mesh-guided Gaussian Splatting - Initialization module
# 2D-pixel-space adaptive Gaussian sampling
#
# Sampling in 2D pixel space (one Gaussian per visible patch):
#   1. Sample a dense candidate set of surface points on the mesh.
#   2. Project every candidate into every training view.
#   3. For each (seed, view) pair compute
#         - pixel coordinates (u, v)
#         - per-pixel complexity
#         - visual footprint area = 1px * depth^2 / focal^2
#         - visibility weight (cos <n, view>)
#   4. Across ALL (seed, view) pairs, draw `target_M` pairs weighted by
#          visual_area * pixel_complexity * visibility_weight
#      (optionally raised to a power).
#   5. Each drawn pair becomes a Gaussian at the seed's 3D position with
#      a scale proportional to the visual footprint, and a rotation that
#      aligns the Gaussian's plane to the local mesh + view.
#

import numpy as np
import torch

from .complexity_extractor import compute_image_complexity
from .complexity_fusion import fuse_complexity_to_surface
from .projector import (
    camera_info_intrinsic,
    camera_info_center,
    world_to_camera,
    project_to_pixel,
    sample_complexity_at_uv,
    visibility_weight,
)
from .adaptive_sampler import compute_face_tangent_scales


def _all_projections(points_world, normals_world, cam_infos, cmap_dict):
    """
    For every (seed, view) pair, compute
        uv          (N, V, 2)
        depth       (N, V)
        pix_complex (N, V)
        vis_weight  (N, V)

    Out-of-frame pairs get weight 0 and uv = (-1, -1).

    Returns four arrays of shape (N, V).
    """
    N = points_world.shape[0]
    V = len(cam_infos)
    uv_all        = np.full((N, V, 2), -1.0, dtype=np.float32)
    depth_all     = np.zeros((N, V), dtype=np.float32)
    pix_complex   = np.zeros((N, V), dtype=np.float32)
    vis_weight    = np.zeros((N, V), dtype=np.float32)

    for v, cam in enumerate(cam_infos):
        K  = camera_info_intrinsic(cam)
        cw = camera_info_center(cam)

        pts_cam = world_to_camera(points_world, cam.R, cam.T)
        uv, depth = project_to_pixel(pts_cam, K)

        cmap = cmap_dict.get(cam.image_name)
        if cmap is None:
            continue

        comp, valid = sample_complexity_at_uv(cmap, uv)
        w = visibility_weight(normals_world, points_world, cw, depth, valid)

        uv_all[:, v]      = uv
        depth_all[:, v]   = depth
        pix_complex[:, v] = comp
        vis_weight[:, v]  = w
    return uv_all, depth_all, pix_complex, vis_weight


def visual_footprint_area(depth, fx, fy):
    """
    The world-space area 'covered' by a single image pixel at a given depth.

    A 1x1 pixel on the image plane subtends
        dA = (depth / fx) * (depth / fy)
    in world units^2 on the focal plane.

    Returns
    -------
    (N,) float32
    """
    return (depth / fx) * (depth / fy)


def visual_gaussian_sampling(
    mesh,
    points_world,                      # (N_seed, 3) numpy
    normals_world,                     # (N_seed, 3) numpy
    face_id_per_seed,                  # (N_seed,)
    complexity_per_seed,               # (N_seed,)  in [0, 1]
    train_cam_infos,
    cmap_dict,
    *,
    target_M,                          # total Gaussians to produce
    complexity_power=1.0,
    footprint_floor=1e-7,
    seed=0,
):
    """
    Sample exactly `target_M` Gaussians in the 2D pixel space and lift each
    back to a 3D surface point.

    The sampling probability for each (seed, view) pair is
        w  =  clip(visual_area, floor, inf)
            * clip(pixel_complexity, 1e-3, 1.0) ^ complexity_power
            * clip(visibility_weight, 0, inf)
    Normalized across all pairs.

    Each resulting Gaussian has:
        xyz       = mesh surface point at that seed (already 3D)
        normal    = per-face normal of seed's face
        face_id   = seed's face id
        scale_init = (s_normal, s_tangent_v, s_tangent_u) where
                     s_normal is extremely small (disk-like) and
                     s_tangent is the face's (s1, s2) tangent scales.
        rotation_init = quaternion aligning the Gaussian's axes to the face.

    Returns
    -------
    dict like adaptive_gaussian_sampling's return.
    """
    rng = np.random.default_rng(seed)

    points_world  = np.asarray(points_world,  dtype=np.float32)
    normals_world = np.asarray(normals_world, dtype=np.float32)
    N = points_world.shape[0]
    V = len(train_cam_infos)

    # ---- per-(seed, view) features ----
    uv_all, depth_all, pix_complex, vis_w = _all_projections(
        points_world, normals_world, train_cam_infos, cmap_dict
    )

    # ---- visual footprint area per pair ----
    # pick the first view's fx,fy just to get a reference focal; we use the
    # per-view intrinsic so this is exact:
    footprint = np.zeros((N, V), dtype=np.float32)
    for v, cam in enumerate(train_cam_infos):
        K  = camera_info_intrinsic(cam)
        fx = float(K[0, 0]); fy = float(K[1, 1])
        footprint[:, v] = visual_footprint_area(depth_all[:, v], fx, fy)

    # ---- sampling weights per (seed, view) pair ----
    area   = np.maximum(footprint, footprint_floor)
    comp   = np.clip(pix_complex, 1e-3, 1.0) ** complexity_power
    weight = (area * comp * vis_w).astype(np.float32).flatten()
    if weight.sum() <= 0:
        # fall back to uniform
        weight = np.ones_like(weight)
    prob   = weight / weight.sum()

    # ---- multinomial draw ----
    N_pairs = N * V
    flat_idx = rng.choice(N_pairs, size=target_M, p=prob)
    seed_idx, view_idx = np.unravel_index(flat_idx, (N, V))

    # ---- back to 3D ----
    xyz = points_world[seed_idx]
    normals_out = normals_world[seed_idx]
    face_id_out = np.asarray(face_id_per_seed, dtype=np.int64)[seed_idx]
    complexity_out = complexity_per_seed[seed_idx]

    # ---- per-Gaussian face tangent scales (from the seed's face) ----
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces    = np.asarray(mesh.faces,    dtype=np.int64)
    s1_face, s2_face = compute_face_tangent_scales(vertices, faces)
    s1 = s1_face[face_id_out]
    s2 = s2_face[face_id_out]
    # shrink slightly with complexity (textured regions → smaller, denser)
    comp_factor = (1.0 - 0.5 * complexity_out).astype(np.float32)
    s1 = np.maximum(s1 * comp_factor, 1e-6).astype(np.float32)
    s2 = np.maximum(s2 * comp_factor, 1e-6).astype(np.float32)
    s0 = np.full(target_M, 1e-4, dtype=np.float32)
    scale_init = np.stack([s0, s1, s2], axis=1).astype(np.float32)

    # ---- rotation: align the Gaussian's frame to the face normal ----
    chosen_tri = vertices[faces[face_id_out]]
    e1 = chosen_tri[:, 1] - chosen_tri[:, 0]
    e2 = chosen_tri[:, 2] - chosen_tri[:, 0]
    n_xyz = np.cross(e1, e2)
    n_xyz = n_xyz / (np.linalg.norm(n_xyz, axis=-1, keepdims=True) + 1e-12)
    centroid = chosen_tri.mean(axis=1)
    v1_init = chosen_tri[:, 1] - centroid
    v1_unit = v1_init / (np.linalg.norm(v1_init, axis=-1, keepdims=True) + 1e-12)
    v2_init = chosen_tri[:, 2] - centroid
    v2_init = v2_init - (v2_init * n_xyz).sum(-1, keepdims=True) * n_xyz
    v2_unit = v2_init / (np.linalg.norm(v2_init, axis=-1, keepdims=True) + 1e-12)
    R = np.stack([v1_unit, v2_unit, n_xyz], axis=-1).astype(np.float32)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    sq = np.sqrt(np.clip(1.0 + trace, 1e-12, None))
    rot_init = np.empty((target_M, 4), dtype=np.float32)
    rot_init[:, 0] = 0.5 * sq
    rot_init[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (2.0 * sq)
    rot_init[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (2.0 * sq)
    rot_init[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (2.0 * sq)
    rn = np.linalg.norm(rot_init, axis=-1, keepdims=True) + 1e-12
    rot_init = (rot_init / rn).astype(np.float32)

    # ---- seed_idx (first seed on the same face, for color/feature lookup) ----
    first_seed = -np.ones(faces.shape[0], dtype=np.int64)
    seen = np.zeros(faces.shape[0], dtype=bool)
    for i, f in enumerate(face_id_per_seed):
        if not seen[f]:
            first_seed[f] = i
            seen[f] = True
    seed_idx_out = first_seed[face_id_out]

    # barycentric of the seed itself (kept for legacy callers)
    bary_all = None  # not used downstream in mesh_guided_init

    return {
        "xyz":          torch.from_numpy(xyz.astype(np.float32)),
        "normal":       torch.from_numpy(normals_out.astype(np.float32)),
        "face_id":      torch.from_numpy(face_id_out.astype(np.int64)),
        "barycentric":  bary_all,
        "seed_idx":     torch.from_numpy(seed_idx_out.astype(np.int64)),
        "complexity":   torch.from_numpy(complexity_out.astype(np.float32)),
        "scale_init":   torch.from_numpy(scale_init),
        "rotation_init":torch.from_numpy(rot_init),
    }