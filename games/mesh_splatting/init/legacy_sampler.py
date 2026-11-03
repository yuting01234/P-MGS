#
# Mesh-guided Gaussian Splatting - Initialization module
# "Ablation" sampler: pure-legacy GaMeS-style random initialization.
#
# This intentionally drops ALL three MeshGS-init components:
#   * no complexity_extractor / complexity_fusion (no image Sobel / var)
#   * no adaptive face weighting (faces sampled uniformly wrt area only)
#   * no patch2d sampler (no per-pixel / per-face visibility reasoning)
#   * no normal-aligned jitter
# Scale is sampled randomly per Gaussian, rotation is the identity
# quaternion.  Color stays random.
#
# Implementation kept minimal so the ablation is easy to read side-by-side
# with `adaptive_sampler.adaptive_gaussian_sampling`.
#

import numpy as np
import torch


def legacy_random_sampling(
    mesh,
    points_world,                      # unused, kept for API compatibility
    normals_world,                     # unused
    face_id_per_seed,                  # unused
    complexity_per_seed,               # unused
    num_per_face=8,
    base_size=0.02,
    noise_scale_ratio=0.0,             # explicitly OFF in ablation
    seed=None,
    position_lr_scale=1.0,             # unused (no jitter)
):
    """
    Ablation-mode initialization:
      M   = num_faces * num_per_face  (same as adaptive_sampler)
      per-face sampling weight = face_area only (no complexity multiplier)
      per-Gaussian position   = uniform barycentric on each sampled face
      per-Gaussian rotation   = identity quaternion (w=1, x=y=z=0)
      per-Gaussian scale      = random (uniform in [base_size, 3*base_size])
                                along each of the three axes; flatness by
                                accident (min scale may not be smallest)
      no along-normal jitter, no complexity-aware shrinking, no image I/O

    Returns the same dict schema as adaptive_gaussian_sampling so the
    downstream pipeline does not need to branch on sampler type.
    """
    del points_world, normals_world, face_id_per_seed, complexity_per_seed
    del noise_scale_ratio, position_lr_scale  # explicitly disabled in ablation

    rng = np.random.default_rng(seed)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces    = np.asarray(mesh.faces,    dtype=np.int64)
    num_faces = faces.shape[0]

    # ---- 1. area-weighted face sampling (no complexity term) ----
    face_area = 0.5 * np.linalg.norm(
        np.cross(vertices[faces][:, 1] - vertices[faces][:, 0],
                 vertices[faces][:, 2] - vertices[faces][:, 0]),
        axis=1,
    ).astype(np.float32)
    face_w = face_area / (face_area.sum() + 1e-12)

    M = num_faces * num_per_face
    sampled_face = rng.choice(num_faces, size=M, p=face_w)

    # ---- 2. uniform barycentric on each sampled face ----
    u   = rng.random(M)
    v   = rng.random(M)
    bad = (u + v) > 1.0
    u[bad] = 1.0 - u[bad]
    v[bad] = 1.0 - v[bad]
    w   = 1.0 - u - v
    barycentric = np.stack([u, v, w], axis=1).astype(np.float32)

    # ---- 3. surface positions ----
    chosen_tri = vertices[faces][sampled_face]
    xyz = (barycentric[:, :, None] * chosen_tri).sum(axis=1).astype(np.float32)

    # ---- 4. normals (per-face, used downstream only as a buffer attr) ----
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
    normal = face_normals[sampled_face]

    # ---- 5. complexity = zero for every Gaussian (legacy has none) ----
    gauss_complexity = np.zeros(M, dtype=np.float32)

    # ---- 6. random scale (ablation: no per-face anisotropic geometry) ----
    lo, hi = base_size, 3.0 * base_size
    scale_init = rng.uniform(lo, hi, size=(M, 3)).astype(np.float32)

    # ---- 7. identity quaternion ----
    rotation_init = np.zeros((M, 4), dtype=np.float32)
    rotation_init[:, 0] = 1.0   # w=1, (x,y,z)=(0,0,0)

    # ---- 8. seed_idx : trivial (no complexity to point back to) ----
    seed_idx = np.zeros(M, dtype=np.int64)

    return {
        "xyz":           torch.from_numpy(xyz),
        "normal":        torch.from_numpy(normal),
        "face_id":       torch.from_numpy(sampled_face.astype(np.int64)),
        "seed_idx":      torch.from_numpy(seed_idx),
        "complexity":    torch.from_numpy(gauss_complexity),
        "scale_init":    torch.from_numpy(scale_init),
        "rotation_init": torch.from_numpy(rotation_init),
    }
