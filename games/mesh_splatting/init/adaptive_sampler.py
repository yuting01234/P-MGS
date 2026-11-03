#
# Mesh-guided Gaussian Splatting - Initialization module
# Adaptive Gaussian sampling
#
# Simplified v2 design:
#   1. Target total Gaussian count = num_faces * num_per_face
#   2. Each face gets a sampling weight proportional to
#          area * complexity(face) ^ complexity_power
#   3. We Multinomial-sample M = total_count face IDs from this distribution.
#   4. For each sampled face, draw a uniform random barycentric coordinate
#      and place exactly one Gaussian there.
#   5. The Gaussian is bound to its face (face_id).
#
# Per-Gaussian scale is adapted to the **actual triangle geometry**:
#   s_tangent  = sqrt(area) * tangent_scale   (in-plane size)
#   s_normal   = max(eps, s_tangent * flatness) (along face normal)
# where `flatness` follows the original GaMeS disk-like covariance.

import numpy as np
import torch


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def compute_face_complexity(face_id, complexity, num_faces):
    """
    Aggregate per-seed complexity back to per-face (mean over all seed
    samples that landed on each face).  Returns (F,) float32 in [0, 1].
    """
    sums   = np.zeros(num_faces, dtype=np.float32)
    counts = np.zeros(num_faces, dtype=np.float32)
    np.add.at(sums,   face_id, complexity)
    np.add.at(counts, face_id, 1.0)
    return np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)


def compute_face_area(vertices, faces):
    """Per-face area vector (F,)."""
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    ).astype(np.float32)


def compute_face_tangent_scales(vertices, faces):
    """
    Per-face in-plane size (s1, s2) from the **exact same**
    Gram-Schmidt decomposition used in the original GaMeS
    `prepare_scaling_rot`.

    This ensures a single Gaussian placed at a uniform barycentric point
    of a triangle fully covers that triangle's area, matching the legacy
    PSNR floor.

    Returns
    -------
    s1, s2 : (F,) float32,  in-plane axes (>= eps).
    """
    tri = vertices[faces]
    # triangle normal (v0 direction)
    n   = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # (F, 3)
    n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
    v0  = n / (n_norm + 1e-12)                                    # (F, 3) unit normal

    centroid = tri.mean(axis=1)
    # v1 = vertex 1 - centroid, normalized
    v1_init  = tri[:, 1] - centroid
    v1_norm  = np.linalg.norm(v1_init, axis=-1, keepdims=True)     # (F, 1)
    v1_unit  = v1_init / (v1_norm + 1e-12)
    # v2 = vertex 2 - centroid, orthogonalized against v0 and v1
    v2_init  = tri[:, 2] - centroid
    coef_v0  = (v2_init * v0).sum(axis=-1, keepdims=True)
    coef_v1  = (v2_init * v1_unit).sum(axis=-1, keepdims=True)
    v2       = v2_init - coef_v0 * v0 - coef_v1 * v1_unit
    v2_norm  = np.linalg.norm(v2, axis=-1, keepdims=True)
    v2_unit  = v2 / (v2_norm + 1e-12)

    # exactly match prepare_scaling_rot() in gaussian_mesh_model.py
    s1 = (v1_norm.squeeze(-1) / 2.0).astype(np.float32)
    s2 = ((v2_init * v2_unit).sum(axis=-1) / 2.0).astype(np.float32)
    # NB: do NOT floor here; very degenerate triangles legitimately have
    # s2 → 0 (line-like).  If we need a minimum, it should be applied to
    # the *combined* disk size, not to s2 alone.
    s1 = np.clip(s1, 1e-6, None)
    s2 = np.clip(s2, 1e-6, None)
    return s1, s2


def compute_face_sampling_weights(face_area, face_complexity,
                                   complexity_power=1.0,
                                   min_weight=1e-3,
                                   weight_mode="power",
                                   weight_alpha=0.5):
    """
    Sampling probability per face.

        power  mode (default, original):
            w_i = area_i * clip(complexity_i, min_weight, 1)^complexity_power

        smooth mode (alternative):
            w_i = area_i * (1 + alpha * clip(complexity_i, 0, 1))

    Normalized to sum to 1.  The `min_weight` clip in power mode
    guarantees that flat faces (complexity 0) still receive a small share
    of Gaussians.  In smooth mode, complexity only nudges the area
    weight up, never down -> much more uniform face coverage.
    """
    comp = np.asarray(face_complexity, dtype=np.float32)
    if weight_mode == "smooth":
        comp = np.clip(comp, 0.0, 1.0)
        w = np.asarray(face_area, dtype=np.float32) * (1.0 + weight_alpha * comp)
    else:  # default 'power' mode
        comp = np.clip(comp, min_weight, 1.0)
        w = np.asarray(face_area, dtype=np.float32) * np.power(comp, complexity_power)
    w = w / (w.sum() + 1e-12)
    return w.astype(np.float32)


def _first_seed_index_per_face(face_id, num_faces):
    """For each face, return the index of the first seed that landed on it."""
    first = np.full(num_faces, -1, dtype=np.int64)
    seen = np.zeros(num_faces, dtype=bool)
    for i, f in enumerate(face_id):
        if not seen[f]:
            first[f] = i
            seen[f] = True
    return first


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def adaptive_gaussian_sampling(
    mesh,
    points_world,                      # (N_seed, 3)  kept for API compat
    normals_world,                     # (N_seed, 3)
    face_id_per_seed,                  # (N_seed,)
    complexity_per_seed,               # (N_seed,)
    num_per_face=8,                    # Gaussians per face *total*
    complexity_power=1.0,
    base_size=0.02,                    # NOTE: now treated as a *floor*,
                                       # not the absolute scale.
    noise_scale_ratio=0.02,
    seed=None,
    position_lr_scale=1.0,
    use_comp_scale_shrink=True,        # if False, comp_factor=1.0 (ablation:
                                       # CCC affects placement only, not scale)
    comp_shrink_alpha=0.5,              # how aggressively complexity shrinks
                                       # the Gaussian scale (0=no shrink,
                                       # 0.5=default original)
    scale_jitter_lo=None,              # if not None, multiply s1,s2 by
    scale_jitter_hi=None,              # uniform[jitter_lo, jitter_hi] per-Gaussian
                                       # (legacy-like diversity)
    weight_mode="power",               # 'power' (area*comp^power) or
                                       # 'smooth' (area*(1+alpha*comp))
    weight_alpha=0.5,                  # alpha used by 'smooth' weight mode
    rotation_mode="face_tangent",      # 'face_tangent' (default) or 'identity'
                                       # identity matches legacy ablation
    jitter_mode="tangent",             # 'tangent' (only along surface) or
                                       # 'full' (3D noise, legacy-like)
):
    """
    Sample exactly `num_faces * num_per_face` Gaussians on the mesh.

    Per-Gaussian scale is **adaptive to the actual triangle geometry**:
    each Gaussian inherits its face's (s1, s2) tangent scales (so that an
    ellipse of axes (s1, s2) covers the triangle), then optionally scaled
    down by a complexity factor on textured regions.

    Parameters
    ----------
    mesh            : trimesh.Trimesh
    points_world    : (N_seed, 3) numpy – surface candidates
    normals_world   : (N_seed, 3) numpy – unused
    face_id_per_seed: (N_seed,) numpy – face index of each seed
    complexity_per_seed: (N_seed,) numpy – per-seed complexity in [0, 1]
    num_per_face    : int  – Gaussians per face on average
    complexity_power: float – > 1 emphasizes complex regions more
    base_size       : float – minimum in-plane scale (mesh world units);
                        used as a lower bound to prevent tiny sliver faces
                        from collapsing into degenerate Gaussians.
    noise_scale_ratio: float – jitter as fraction of in-plane scale
    seed            : int   – RNG seed
    position_lr_scale: float

    Returns
    -------
    dict of tensors on CPU:
        xyz          (M, 3)
        normal       (M, 3)
        face_id      (M,)
        seed_idx     (M,)   (first seed on the same face)
        complexity   (M,)   (face-level complexity)
        scale_init   (M, 3)
        rotation_init(M, 4)
    """
    rng = np.random.default_rng(seed)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces    = np.asarray(mesh.faces,    dtype=np.int64)
    num_faces = faces.shape[0]

    # ---- 1. face areas & per-face tangent scales ----
    face_area = compute_face_area(vertices, faces)
    s1_face, s2_face = compute_face_tangent_scales(vertices, faces)

    # ---- 2. face complexity (mean complexity of seeds on each face) ----
    face_complexity = compute_face_complexity(
        face_id_per_seed, complexity_per_seed, num_faces
    )

    # ---- 3. sampling weights & multinomial sample ----
    face_w = compute_face_sampling_weights(
        face_area, face_complexity,
        complexity_power=complexity_power,
        weight_mode=weight_mode,
        weight_alpha=weight_alpha,
    )

    M = num_faces * num_per_face
    sampled_face = rng.choice(num_faces, size=M, p=face_w)

    # ---- 4. uniform barycentric on each sampled face ----
    u   = rng.random(M)
    v   = rng.random(M)
    bad = (u + v) > 1.0
    u[bad] = 1.0 - u[bad]
    v[bad] = 1.0 - v[bad]
    w   = 1.0 - u - v
    barycentric = np.stack([u, v, w], axis=1).astype(np.float32)

    # ---- 5. surface positions ----
    tri = vertices[faces]
    chosen_tri = tri[sampled_face]
    xyz = (barycentric[:, :, None] * chosen_tri).sum(axis=1).astype(np.float32)

    # ---- 6. normals (per-face) ----
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
    normal = face_normals[sampled_face]

    # ---- 7. per-Gaussian complexity = face complexity ----
    gauss_complexity = face_complexity[sampled_face]

    # ---- 8. adaptive anisotropic scale (matches triangle geometry) ----
    s1 = s1_face[sampled_face]
    s2 = s2_face[sampled_face]

    # complexity shrinks scales on textured regions: small Gaussians are
    # denser per area and can resolve fine texture better
    # comp_shrink_alpha controls how aggressively complexity shrinks the
    # scale. 0.0 = no shrink (matches legacy), 0.5 = original.
    if use_comp_scale_shrink:
        comp_factor = (1.0 - comp_shrink_alpha * gauss_complexity).astype(np.float32)
    else:
        # Ablation mode: CCC affects placement only, NOT scale.
        # Use neutral comp_factor so scale follows triangle geometry purely.
        comp_factor = np.ones_like(gauss_complexity, dtype=np.float32)
    s1 = np.maximum(s1 * comp_factor, base_size).astype(np.float32)
    s2 = np.maximum(s2 * comp_factor, base_size).astype(np.float32)

    # optional per-Gaussian scale jitter (legacy-like diversity)
    if scale_jitter_lo is not None and scale_jitter_hi is not None:
        jitter = rng.uniform(scale_jitter_lo, scale_jitter_hi, size=M).astype(np.float32)
        s1 = (s1 * jitter).astype(np.float32)
        s2 = (s2 * jitter).astype(np.float32)
    s0 = np.full(M, 1e-4, dtype=np.float32)              # disk-like
    scale_init = np.stack([s0, s1, s2], axis=1).astype(np.float32)

    # ---- 9. jitter ----
    sigma = (noise_scale_ratio * s1 * position_lr_scale).astype(np.float32)
    jitter = rng.normal(0.0, 1.0, size=(M, 3)).astype(np.float32)
    if jitter_mode == "tangent":
        # only jitter along tangent plane (use normal mask)
        normal_unit = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)
        proj_jitter = jitter - (jitter * normal_unit).sum(axis=-1, keepdims=True) * normal_unit
        xyz = xyz + proj_jitter * sigma[:, None]
    else:
        # legacy-like: full 3D jitter (also moves off the surface)
        xyz = xyz + jitter * sigma[:, None]

    # ---- 10. build rotation matching the face tangent frame ----
    if rotation_mode == "face_tangent":
        # v0 (normal) -> rotation col 2
        # v1 (first tangent) -> rotation col 0
        # v2 (second tangent) -> rotation col 1
        tri_world = chosen_tri
        n_xyz = np.cross(tri_world[:, 1] - tri_world[:, 0],
                         tri_world[:, 2] - tri_world[:, 0])
        n_xyz = n_xyz / (np.linalg.norm(n_xyz, axis=-1, keepdims=True) + 1e-12)
        centroid = tri_world.mean(axis=1)
        v1_init = tri_world[:, 1] - centroid
        v1_unit = v1_init / (np.linalg.norm(v1_init, axis=-1, keepdims=True) + 1e-12)
        v2_init = tri_world[:, 2] - centroid
        v2_init = v2_init - (v2_init * n_xyz).sum(-1, keepdims=True) * n_xyz
        v2_unit = v2_init / (np.linalg.norm(v2_init, axis=-1, keepdims=True) + 1e-12)
        # Rotation matrix per Gaussian, columns are basis vectors in world frame
        R = np.stack([v1_unit, v2_unit, n_xyz], axis=-1).astype(np.float32)

        # Rotation matrix -> quaternion (w, x, y, z), positive branch.
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        sq = np.sqrt(np.clip(1.0 + trace, 1e-12, None))
        rot_init = np.empty((M, 4), dtype=np.float32)
        rot_init[:, 0] = 0.5 * sq
        rot_init[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (2.0 * sq)
        rot_init[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (2.0 * sq)
        rot_init[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (2.0 * sq)
        rot_norm = np.linalg.norm(rot_init, axis=-1, keepdims=True) + 1e-12
        rot_init = (rot_init / rot_norm).astype(np.float32)
    else:  # 'identity' rotation (legacy-style, free to rotate during training)
        rot_init = np.zeros((M, 4), dtype=np.float32)
        rot_init[:, 0] = 1.0  # w=1, (x,y,z)=(0,0,0)

    # ---- 11. seed_idx : first seed on each face ----
    first_seed = _first_seed_index_per_face(face_id_per_seed, num_faces)
    seed_idx = first_seed[sampled_face].astype(np.int64)

    return {
        "xyz":          torch.from_numpy(xyz),
        "normal":       torch.from_numpy(normal),
        "face_id":      torch.from_numpy(sampled_face.astype(np.int64)),
        "seed_idx":     torch.from_numpy(seed_idx),
        "complexity":   torch.from_numpy(gauss_complexity),
        "scale_init":   torch.from_numpy(scale_init),
        "rotation_init":torch.from_numpy(rot_init),
    }


def assign_initial_color(gaussians, mode="random"):
    """Tiny random initial color (SH model refines it)."""
    M = gaussians["xyz"].shape[0]
    if mode == "random":
        return np.random.random((M, 3)).astype(np.float32)
    raise ValueError(f"unknown color init mode: {mode}")