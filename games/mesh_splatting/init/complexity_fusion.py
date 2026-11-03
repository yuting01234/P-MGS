#
# Mesh-guided Gaussian Splatting - Initialization module
# Multi-view complexity fusion
#

import numpy as np
from .projector import (
    camera_info_intrinsic,
    camera_info_center,
    world_to_camera,
    project_to_pixel,
    sample_complexity_at_uv,
    visibility_weight,
)


def fuse_complexity_to_surface(
    points_world, normals_world,
    train_cam_infos, per_view_complexity,
    power=1.0, eps=1e-8,
):
    """
    Sample per-view appearance complexity at every surface point and combine
    via the visibility-aware cosine weight.

    Parameters
    ----------
    points_world       : (N, 3) numpy
    normals_world      : (N, 3) numpy
    train_cam_infos    : list of GS CameraInfo (must expose .R, .T, .FovX,
                         .FovY, .image, .width, .height, .image_name)
    per_view_complexity: dict  { image_name -> (H, W) float32 complexity map }
    power              : exponent applied to the per-view weight
    eps                : safe-denominator

    Returns
    -------
    complexity   : (N,) float32 in [0, 1]
    contributions: (N, V) float32   per-view sampled complexity (0 = invalid)
    weights      : (N, V) float32   per-view weight     (0 = invalid)
    """
    points_world  = np.asarray(points_world,  dtype=np.float32)
    normals_world = np.asarray(normals_world, dtype=np.float32)

    N = points_world.shape[0]
    V = len(train_cam_infos)
    contributions = np.zeros((N, V), dtype=np.float32)
    weights       = np.zeros((N, V), dtype=np.float32)

    for v, cam in enumerate(train_cam_infos):
        K  = camera_info_intrinsic(cam)
        cw = camera_info_center(cam)

        pts_cam = world_to_camera(points_world, cam.R, cam.T)
        uv, depth = project_to_pixel(pts_cam, K)

        cmap = per_view_complexity.get(cam.image_name, None)
        if cmap is None:
            continue

        comp, valid = sample_complexity_at_uv(cmap, uv)
        w = visibility_weight(normals_world, points_world, cw, depth, valid)
        if power != 1.0:
            w = np.power(np.clip(w, 0.0, None), power)

        contributions[:, v] = comp
        weights[:, v]       = w

    sum_w = weights.sum(axis=1)
    fused = np.where(sum_w > eps,
                     (weights * contributions).sum(axis=1) / np.maximum(sum_w, eps),
                     0.0)
    return np.clip(fused, 0.0, 1.0).astype(np.float32), contributions, weights
