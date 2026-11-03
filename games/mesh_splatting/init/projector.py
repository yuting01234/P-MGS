#
# Mesh-guided Gaussian Splatting - Initialization module
# Camera utilities & projection helpers
#

import numpy as np


def fov_to_intrinsic(fov_x, fov_y, width, height):
    """
    Convert (FovX, FovY, W, H) to a 3x3 OpenCV-style intrinsic matrix
    (pixel coords, origin = top-left, +x right, +y down).
    """
    fx = (width  / 2.0) / np.tan(fov_x / 2.0)
    fy = (height / 2.0) / np.tan(fov_y / 2.0)
    K = np.array([[fx, 0.0, width  / 2.0],
                  [0.0, fy, height / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def camera_center_world(R_gs, T_gs):
    """
    Optical center of a Gaussian-Splatting style camera in WORLD coords.
    Camera convention: pc = R @ pw + T  =>  pw_cam_center = -R^T T.
    """
    R = np.asarray(R_gs, dtype=np.float32)
    T = np.asarray(T_gs, dtype=np.float32)
    return -R.T @ T


def camera_info_intrinsic(cam):
    """Use calibrated DTU intrinsics; retain historical FoV cameras otherwise."""
    intrinsic = getattr(cam, "intrinsics", None)
    if intrinsic is None:
        return fov_to_intrinsic(cam.FovX, cam.FovY, cam.width, cam.height)
    fx, fy, cx, cy = intrinsic
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def camera_info_center(cam):
    """DTU CameraInfo.R stores the transpose of the OpenCV w2c rotation."""
    if getattr(cam, "intrinsics", None) is None:
        return camera_center_world(cam.R, cam.T)
    return -np.asarray(cam.R, dtype=np.float32) @ np.asarray(cam.T, dtype=np.float32)


def world_to_camera(points_w, R_gs, T_gs):
    """Apply pc = R @ pw + T to (N, 3) world points -> camera frame."""
    pts = np.asarray(points_w, dtype=np.float32)
    return (pts @ np.asarray(R_gs, dtype=np.float32)
            + np.asarray(T_gs, dtype=np.float32))


def project_to_pixel(points_cam, K):
    """
    Standard pinhole projection.  Returns pixel coords (u, v) and depth (z).
    points_cam: (N, 3)  with z > 0 = in front of the camera.
    """
    z = points_cam[..., 2]
    z_safe = z + 1e-8
    fx = K[0, 0];  fy = K[1, 1]
    cx = K[0, 2];  cy = K[1, 2]
    u = fx * points_cam[..., 0] / z_safe + cx
    v = fy * points_cam[..., 1] / z_safe + cy
    uv = np.stack([u, v], axis=-1).astype(np.float32)
    return uv, z.astype(np.float32)


def sample_complexity_at_uv(complexity_map, uv, default=0.0):
    """
    Bilinear sample at integer pixel locations.
    Pixels outside the image (or just outside) receive `default`.

    Returns (samples, valid_mask).
    """
    H, W = complexity_map.shape
    u = uv[..., 0]
    v = uv[..., 1]

    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1

    valid = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)

    u0c = np.clip(u0, 0, W - 1);  v0c = np.clip(v0, 0, H - 1)
    u1c = np.clip(u1, 0, W - 1);  v1c = np.clip(v1, 0, H - 1)

    c00 = complexity_map[v0c, u0c].astype(np.float32)
    c10 = complexity_map[v0c, u1c].astype(np.float32)
    c01 = complexity_map[v1c, u0c].astype(np.float32)
    c11 = complexity_map[v1c, u1c].astype(np.float32)

    du = (u - u0c).astype(np.float32)
    dv = (v - v0c).astype(np.float32)

    samples = (c00 * (1 - du) * (1 - dv) +
               c10 * du       * (1 - dv) +
               c01 * (1 - du) * dv       +
               c11 * du       * dv).astype(np.float32)
    samples[~valid] = default
    return samples, valid


def visibility_weight(normals_world, points_world, cam_center_world, depth,
                      visibility_mask):
    """
    Visibility-aware cosine weight:
        w = max( <n, view_dir>, 0 ) * visibility_mask * front_face

    view_dir = (point_w - cam_center_w) / ||.||
    """
    nw = np.asarray(normals_world,     dtype=np.float32)
    pw = np.asarray(points_world,      dtype=np.float32)
    cw = np.asarray(cam_center_world,  dtype=np.float32)

    view  = pw - cw                                                # (N, 3)
    nrm   = np.linalg.norm(view, axis=-1, keepdims=True) + 1e-8
    v_dir = view / nrm

    cos = np.einsum("ij,ij->i", nw, v_dir)
    w   = np.clip(cos, 0.0, None).astype(np.float32)
    w  *= np.asarray(visibility_mask, dtype=np.float32)
    w  *= (depth > 0.0).astype(np.float32)
    return w
