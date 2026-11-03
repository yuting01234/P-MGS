#
# Mesh-guided Gaussian Splatting - Initialization module
# 2D-patch-aware sampling via projected mesh surface points + real image patches.
#
# Algorithm:
#   1. Sample N_surface mesh surface points (trimesh area-weighted barycentric).
#   2. Project all points into every training view.
#      For each (seed, view) pair we get: uv, depth, visibility.
#   3. For each pair, query the REAL training image at uv with a local patch
#      to compute texture complexity (Sobel + local variance on the real RGB).
#   4. Sampling weight = visual_footprint * patch_complexity * visibility.
#      visual_footprint = (depth/fx) * (depth/fy)  [world area of 1 image pixel]
#   5. Multinomial draw of N_stage1 pairs (weighted by complexity), back-projected to 3D mesh surface.
#   6. Coverage guarantee: any mesh face with < min_coverage Gaussians gets
#      a uniform fallback sample.
#

import numpy as np
import torch
import cv2
from scipy.ndimage import maximum_filter

from .mesh_sampler import sample_mesh_surface
from .projector import (
    camera_info_intrinsic,
    camera_info_center,
    world_to_camera,
    project_to_pixel,
    visibility_weight,
)


def _sobel_magnitude(gray01):
    """3x3 Sobel magnitude on a [0,1] grayscale image."""
    g1 = cv2.getGaussianKernel(3, 1.0).flatten().astype(np.float32)
    gx = np.outer(g1, np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    gy = np.outer(np.array([-1.0, 0.0, 1.0], dtype=np.float32), g1)
    sx = cv2.filter2D(gray01, cv2.CV_32F, gx)
    sy = cv2.filter2D(gray01, cv2.CV_32F, gy)
    return np.sqrt(sx * sx + sy * sy)


def _local_variance(gray01, kernel_size):
    mu  = cv2.boxFilter(gray01,            cv2.CV_32F, (kernel_size, kernel_size))
    mu2 = cv2.boxFilter(gray01 * gray01,  cv2.CV_32F, (kernel_size, kernel_size))
    return np.maximum(mu2 - mu * mu, 0.0)


def _patch_complexity_at_uv_fast(rgb01, uv, patch_size=5,
                                  sobel_weight=0.5, var_weight=0.5, var_kernel=5):
    """
    Fast patch complexity using scipy sliding-window max pooling.

    Returns (N,) float32 in [0, 1].
    """
    H, W = rgb01.shape[:2]
    gray_u8 = cv2.cvtColor(
        np.clip(rgb01 * 255, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0

    sob = _sobel_magnitude(gray_u8)                  # (H, W)
    var = _local_variance(gray_u8, var_kernel)     # (H, W)

    r = patch_size // 2
    sob_max = maximum_filter(sob, size=2 * r + 1, mode='constant', cval=0.0)
    var_max = maximum_filter(var, size=2 * r + 1, mode='constant', cval=0.0)

    u = np.clip(uv[:, 0].astype(np.int32), 0, W - 1)
    v = np.clip(uv[:, 1].astype(np.int32), 0, H - 1)
    s_vals = sob_max[v, u]
    v_vals = var_max[v, u]

    eps = 0.01
    result = sobel_weight * (s_vals / (s_vals + eps)) + var_weight * (v_vals / (v_vals + eps))
    return result.astype(np.float32)


def sample_gaussians_from_2d_patches(
    mesh,
    train_cam_infos,
    *,
    n_seed=300000,
    min_coverage=3,
    patch_size=5,
    sobel_weight=0.5, var_weight=0.5, var_kernel=5,
    complexity_power=1.0,
    visual_area_power=0.5,
    stage1_extra_factor=2.0,
    seed=0,
):
    """
    2D-patch-aware Gaussian sampling — no hard cap on total count.

    Sampling strategy (3 stages):
      Stage 1: Draw F * min_coverage Gaussians via weighted sampling
               (weight = complexity^power * footprint^area_power * visibility).
               This is the PER-FACE MINIMUM GUARANTEE, NOT a total cap.
               Textured / high-resolution areas will naturally accumulate
               more samples than the minimum because they have higher weight.
      Stage 2: Optionally draw extra Gaussians for high-weight (seed, view) pairs
               to further over-sample complex regions.
      Stage 3: Uniform fallback for any still-under-covered faces.

    Key design: there is NO global hard cap.  The total M is determined by
    how many pairs the weighted sampler draws, which naturally reflects the
    complexity distribution in the image.

    Parameters
    ----------
    mesh            : trimesh.Trimesh
    train_cam_infos : list of CameraInfo
    n_seed          : number of surface seed points for projection
    min_coverage    : minimum Gaussians per mesh face (default 3)
    patch_size      : local patch size for texture complexity
    complexity_power: exponent on patch complexity
    visual_area_power: exponent on visual footprint
    stage1_extra_factor: multiply the guaranteed budget by this factor
                        for Stage-1 sampling (higher → more weighted samples
                        but still respects minimum per-face coverage)
    seed            : RNG seed

    Returns
    -------
    dict with xyz, normal, face_id, complexity, scale_init, rotation_init
    """
    rng = np.random.default_rng(seed)

    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mesh_faces    = np.asarray(mesh.faces,    dtype=np.int64)
    F = len(mesh_faces)
    min_gauss_per_face = max(1, min_coverage)
    target_M_stage1 = int(F * min_gauss_per_face * stage1_extra_factor)

    verts = mesh_vertices[mesh_faces]                               # (F, 3, 3)
    e1    = verts[:, 1] - verts[:, 0]
    e2    = verts[:, 2] - verts[:, 0]
    face_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)    # (F,)
    total_area = face_area.sum()

    # ---- 1. sample surface seeds ----
    print(f"[2D-Patch] Sampling {n_seed} mesh surface points ...")
    sampled = sample_mesh_surface(mesh, num_samples=n_seed, seed=seed)
    points_w = sampled["xyz"].numpy().astype(np.float32)             # (N, 3)
    normals_w = sampled["normal"].numpy().astype(np.float32)        # (N, 3)
    seed_face_id = sampled["face_id"].numpy().astype(np.int64)     # (N,)
    bary_seed = sampled["barycentric"].numpy().astype(np.float32)  # (N, 3)
    N = len(points_w)

    # ---- 2. per-view projection + patch complexity ----
    V = len(train_cam_infos)
    # Storage: (N, V) per-pair data
    uv_all   = np.full((N, V, 2), -1.0, dtype=np.float32)
    depth_all= np.zeros((N, V), dtype=np.float32)
    vis_all  = np.zeros((N, V), dtype=np.float32)
    comp_all = np.zeros((N, V), dtype=np.float32)
    color_all = np.zeros((N, V, 3), dtype=np.float32)  # for color init

    print(f"[2D-Patch] Projecting {N} points x {V} views + patch complexity ...")

    for vi, cam in enumerate(train_cam_infos):
        W, H = cam.width, cam.height
        K  = camera_info_intrinsic(cam)
        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx = float(K[0, 2]); cy = float(K[1, 2])
        cw = camera_info_center(cam)
        R  = np.asarray(cam.R, dtype=np.float32)
        T  = np.asarray(cam.T, dtype=np.float32)

        pts_cam = world_to_camera(points_w, R, T)
        uv, depth = project_to_pixel(pts_cam, K)

        uv_all[:, vi]    = uv
        depth_all[:, vi] = depth
        vis_all[:, vi]   = visibility_weight(normals_w, points_w, cw, depth,
                                               (uv[:, 0] >= 0) & (uv[:, 0] <= W-1) &
                                               (uv[:, 1] >= 0) & (uv[:, 1] <= H-1))

        # ---- patch complexity on REAL image ----
        img_rgb = np.asarray(cam.image.convert("RGB"), dtype=np.float32) / 255.0
        uv_valid = uv.copy()
        mask_valid = vis_all[:, vi] > 0
        uv_for_comp = uv_valid[mask_valid]  # (N_valid, 2)

        if len(uv_for_comp) > 0:
            comp_patch = _patch_complexity_at_uv_fast(
                img_rgb, uv_for_comp,
                patch_size=patch_size,
                sobel_weight=sobel_weight, var_weight=var_weight,
                var_kernel=var_kernel,
            )
            comp_all[mask_valid, vi] = comp_patch

            # ---- sample real image color at uv ----
            u_int = np.clip(uv_for_comp[:, 0].astype(np.int32), 0, W - 1)
            v_int = np.clip(uv_for_comp[:, 1].astype(np.int32), 0, H - 1)
            color_all[mask_valid, vi] = img_rgb[v_int, u_int]

        if (vi + 1) % 20 == 0 or vi == V - 1:
            print(f"  view {vi+1}/{V} done")

    # ---- 3. visual footprint area per (seed, view) pair ----
    # One image pixel subtends depth^2 / (fx * fy) world-space area.
    fp = np.zeros((N, V), dtype=np.float32)
    for vi, cam in enumerate(train_cam_infos):
        K = camera_info_intrinsic(cam)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        fp[:, vi] = (depth_all[:, vi] / max(fx, 1e-6)) * (depth_all[:, vi] / max(fy, 1e-6))
    fp = np.maximum(fp, 1e-10)

    # ---- 5. Sampling weights ----
    weight = (np.power(fp, visual_area_power)
              * np.power(comp_all, complexity_power)
              * vis_all).astype(np.float32)                             # (N, V)
    weight_flat = weight.flatten()                                       # (N*V,)
    weight_flat = np.maximum(weight_flat, 0.0).astype(np.float64)
    prob = weight_flat / (weight_flat.sum() + 1e-12)
    # Make sure prob sums to exactly 1 within float64 tolerance.
    drift = 1.0 - prob.sum()
    if drift != 0.0:
        first_nonzero = int(np.argmax(weight_flat > 0.0))
        prob[first_nonzero] += drift

    N_pairs = N * V

    # ---- 5. Stage 1: balanced weighted draw ----
    # Allocate a deterministic minimum quota to every face first. This avoids
    # letting a small set of highly textured faces consume the whole budget.
    face_seed_lists = [np.flatnonzero(seed_face_id == face_id) for face_id in range(F)]
    positive_faces = np.flatnonzero(np.array([len(indices) > 0 for indices in face_seed_lists]))
    quota = max(1, min_gauss_per_face - 3)
    balanced_seed_idx = []
    balanced_view_idx = []
    for face_id in positive_faces:
        seed_candidates = face_seed_lists[face_id]
        face_pair_weights = np.asarray(weight[seed_candidates], dtype=np.float32).reshape(-1).copy()
        face_pair_weights = np.maximum(face_pair_weights, 0.0)
        if face_pair_weights.sum() <= 0.0:
            face_pair_weights = np.ones_like(face_pair_weights, dtype=np.float32)
        face_pair_prob = face_pair_weights / face_pair_weights.sum()
        # Numerical clean-up: drop any tiny negative noise from subnormal fp arithmetic.
        face_pair_prob = np.where(face_pair_prob > 0.0, face_pair_prob, 0.0)
        face_pair_prob = face_pair_prob.astype(np.float64)
        face_pair_prob = face_pair_prob / face_pair_prob.sum()
        # Compensate float drift so numpy strict check passes.
        drift_face = 1.0 - face_pair_prob.sum()
        if drift_face != 0.0:
            first_nonzero_face = int(np.argmax(face_pair_prob > 0.0))
            face_pair_prob[first_nonzero_face] += drift_face
        local_indices = rng.choice(
            len(face_pair_weights), size=quota, p=face_pair_prob, replace=True
        )
        seed_local, view_local = np.unravel_index(local_indices, (len(seed_candidates), V))
        balanced_seed_idx.append(seed_candidates[seed_local])
        balanced_view_idx.append(view_local)

    if balanced_seed_idx:
        seed_idx_balanced = np.concatenate(balanced_seed_idx)
        view_idx_balanced = np.concatenate(balanced_view_idx)
    else:
        seed_idx_balanced = np.zeros(0, dtype=np.int64)
        view_idx_balanced = np.zeros(0, dtype=np.int64)

    n_balanced = len(seed_idx_balanced)
    remaining_budget = max(target_M_stage1 - n_balanced, 0)
    if remaining_budget > 0 and weight_flat.sum() > 0:
        flat_idx_extra = rng.choice(
            N_pairs, size=remaining_budget, p=prob, replace=True
        )
        seed_idx_extra, view_idx_extra = np.unravel_index(flat_idx_extra, (N, V))
        seed_idx_s1 = np.concatenate([seed_idx_balanced, seed_idx_extra])
        view_idx_s1 = np.concatenate([view_idx_balanced, view_idx_extra])
    else:
        seed_idx_s1 = seed_idx_balanced
        view_idx_s1 = view_idx_balanced
    n_stage1 = len(seed_idx_s1)

    # Accumulate per-face counts from Stage-1 draws
    cov_s1 = np.zeros(F, dtype=np.int32)
    np.add.at(cov_s1, seed_face_id[seed_idx_s1], 1)

    # ---- 5b. Stage 2: extra samples for under-represented faces ----
    # Re-weight: restrict to pairs from still-under-covered faces.
    shortfall2 = np.maximum(min_gauss_per_face - cov_s1, 0)
    shortfall2_total = int(shortfall2.sum())

    if shortfall2_total > 0:
        # Build mask: only pairs whose face is still below min_coverage
        under_covered_mask_f = shortfall2 > 0              # (F,)
        pair_seed_f = seed_face_id                          # (N,)
        pair_under_covered = under_covered_mask_f[pair_seed_f]  # (N,) bool
        # For each (seed, view) pair: is this seed's face still under-covered?
        pair_under_covered_2d = np.broadcast_to(pair_under_covered[:, None], (N, V))  # (N, V)

        mask_flat = pair_under_covered_2d.flatten()         # (N*V,)
        weight_under = (weight_flat * mask_flat.astype(np.float64))
        # Build a clean probability vector: drop exact zeros, renormalize, then
        # place zero mass back so the sum is exactly 1 within float tolerance.
        total_w = weight_under.sum()
        if total_w > 0.0 and np.any(mask_flat):
            prob_under = np.zeros_like(weight_under, dtype=np.float64)
            nonzero_mask = weight_under > 0.0
            prob_under[nonzero_mask] = (
                weight_under[nonzero_mask] / total_w
            )
            prob_under = np.maximum(prob_under, 0.0)
            prob_under = prob_under / prob_under.sum()
            # Compensate for any tiny numerical drift so the sum is exactly 1.
            drift = 1.0 - prob_under.sum()
            if drift != 0.0:
                first_nonzero = int(np.argmax(nonzero_mask))
                prob_under[first_nonzero] += drift

            n_stage2 = min(int(shortfall2_total), N_pairs)
            fb_flat = rng.choice(N_pairs, size=n_stage2, p=prob_under, replace=True)
            seed_idx_s2, view_idx_s2 = np.unravel_index(fb_flat, (N, V))
            np.add.at(cov_s1, seed_face_id[seed_idx_s2], 1)
        else:
            seed_idx_s2 = np.zeros(0, dtype=np.int64)
            view_idx_s2 = np.zeros(0, dtype=np.int64)
            n_stage2 = 0
    else:
        seed_idx_s2 = np.zeros(0, dtype=np.int64)
        view_idx_s2 = np.zeros(0, dtype=np.int64)
        n_stage2 = 0

    # ---- 5c. Stage 3: uniform fallback for still-uncovered faces ----
    shortfall3 = np.maximum(min_gauss_per_face - cov_s1, 0)
    shortfall3_total = int(shortfall3.sum())
    print(f"[2D-Patch] Stage 1: {n_stage1} drawn, Stage 2 extra: {n_stage2}, "
          f"Stage 3 fallback: {shortfall3_total}, faces below min_coverage: "
          f"{(cov_s1 < min_gauss_per_face).sum()}")

    # ---- 6. Assemble all Gaussian data ----
    # Combine Stage 1 + Stage 2 into weighted draws
    seed_idx_all_w = np.concatenate([seed_idx_s1, seed_idx_s2])
    view_idx_all_w = np.concatenate([view_idx_s1, view_idx_s2])
    n_weighted = len(seed_idx_all_w)

    xyz_w  = points_w[seed_idx_all_w].astype(np.float32)
    fn_w   = normals_w[seed_idx_all_w].astype(np.float32)
    fid_w  = seed_face_id[seed_idx_all_w].astype(np.int64)
    comp_w = comp_all[seed_idx_all_w, view_idx_all_w].astype(np.float32)

    # Aggregate real colors from all views (weighted by visibility)
    color_w = color_all[seed_idx_all_w, view_idx_all_w].astype(np.float32)  # (n_w, 3)

    if shortfall3_total > 0:
        fb_probs = shortfall3.astype(np.float64)
        fb_probs /= fb_probs.sum() + 1e-12
        fb_fids = rng.choice(F, size=shortfall3_total, p=fb_probs)

        u3 = rng.random(shortfall3_total)
        v3 = rng.random(shortfall3_total)
        bad = (u3 + v3) > 1.0
        u3[bad] = 1.0 - u3[bad]; v3[bad] = 1.0 - v3[bad]
        w3 = 1.0 - u3 - v3
        bary_fb = np.stack([u3, v3, w3], axis=1).astype(np.float32)
        fb_xyz  = (bary_fb[:, :, None] * verts[fb_fids]).sum(axis=1)
        fb_fn   = _face_interp_normal(fb_fids, mesh_vertices, mesh_faces)
        fb_comp = np.full(shortfall3_total, 0.05, dtype=np.float32)

        xyz_draw  = np.concatenate([xyz_w, fb_xyz],  axis=0)
        fid_draw  = np.concatenate([fid_w, fb_fids],  axis=0)
        comp_draw = np.concatenate([comp_w, fb_comp], axis=0)
        fn_draw   = np.concatenate([fn_w, fb_fn],    axis=0)
        # Fallback Gaussians: gray color
        fb_color = np.full((shortfall3_total, 3), 0.5, dtype=np.float32)
        color_draw = np.concatenate([color_w, fb_color], axis=0)
    else:
        xyz_draw  = xyz_w
        fid_draw  = fid_w
        comp_draw = comp_w
        fn_draw   = fn_w
        color_draw = color_w

    M = len(xyz_draw)
    print(f"[2D-Patch] Total Gaussians: {M} (weighted: {n_weighted}, fallback: {M - n_weighted})")
    print(f"[2D-Patch] Coverage: {(cov_s1 >= min_gauss_per_face).sum()}/{F} faces ≥ {min_gauss_per_face} Gaussians")

    # ---- 8. scale init (mesh-geometry-aligned, NOT pixel-footprint) ----
    # CRITICAL: The original GaMeS initializes scales from the mesh face geometry
    # (s1 ≈ half-edge length, s2 ≈ Gram-Schmidt component).  This gives
    # log(scaling) ≈ log(0.05) ≈ -3.  Using pixel footprint gives
    # log(scaling) ≈ log(5e-5) ≈ -10, which is ~1000x smaller and causes
    # poor coverage in early training.
    #
    # We fix this by:
    #   1. Compute per-face (s1_face, s2_face) from mesh geometry
    #      (identical to prepare_scaling_rot in gaussian_mesh_model.py)
    #   2. Use footprint_ratio = footprint_scale / face_baseline_scale
    #      to decide relative density, but keep absolute scale anchored to mesh.

    # 8a. Compute per-face tangent scales from mesh geometry
    normals_f = np.cross(
        verts[:, 1] - verts[:, 0],
        verts[:, 2] - verts[:, 0],
        axis=1
    )                                    # (F, 3)
    n_norm_f = np.linalg.norm(normals_f, axis=1, keepdims=True) + 1e-12
    v0_f = normals_f / n_norm_f          # face normals (F, 3)

    centroid_f = verts.mean(axis=1)       # (F, 3)
    v1_init_f = verts[:, 1] - centroid_f  # (F, 3)
    v1_norm_f = np.linalg.norm(v1_init_f, axis=1) + 1e-12
    v1_f = v1_init_f / v1_norm_f[:, None]

    v2_init_f = verts[:, 2] - centroid_f
    proj_v0 = (v2_init_f * v0_f).sum(axis=1, keepdims=True) * v0_f
    proj_v1 = (v2_init_f * v1_f).sum(axis=1, keepdims=True) * v1_f
    v2_init_f = v2_init_f - proj_v0 - proj_v1
    v2_norm_f = np.linalg.norm(v2_init_f, axis=1) + 1e-12
    v2_f = v2_init_f / v2_norm_f[:, None]

    s1_face = (v1_norm_f / 2.0).astype(np.float32)   # (F,)
    s2_face = ((v2_init_f * v2_f).sum(axis=1) / 2.0).astype(np.float32)  # (F,)

    # 8b. Build scale_init from face geometry for all Gaussians.
    # Weighted (Stage 1+2) Gaussians: modulate by footprint_ratio (high-res → smaller).
    # Stage 3 fallback Gaussians: use face geometry as-is.
    s1_all = s1_face[fid_draw].copy().astype(np.float32)
    s2_all = s2_face[fid_draw].copy().astype(np.float32)

    if n_weighted > 0:
        # Footprint-based modulation for weighted Gaussians.
        # One image pixel subtends depth^2 / (fx * fy) world-space area.
        # footprint_ratio = pixel-visible-area / face-area.
        # footprint_ratio << 1 → pixel "large" relative to face → use face scale
        # footprint_ratio >> 1 → pixel "small" relative to face → needs smaller s
        fp_pair = fp[seed_idx_all_w, view_idx_all_w]                # (n_weighted,)
        fp_pair = np.maximum(fp_pair, 1e-10)
        face_baseline = face_area[fid_draw[:n_weighted]].astype(np.float32)  # (n_weighted,)
        footprint_ratio = np.clip(
            fp_pair / (face_baseline + 1e-12),
            0.01, 10.0
        ).astype(np.float32)
        scale_factor = np.sqrt(footprint_ratio + 1e-12)
        s1_all[:n_weighted] = np.maximum(
            s1_face[fid_draw[:n_weighted]] * scale_factor, 1e-6
        ).astype(np.float32)
        s2_all[:n_weighted] = np.maximum(
            s2_face[fid_draw[:n_weighted]] * scale_factor, 1e-6
        ).astype(np.float32)

    s_normal = np.full(M, 1e-4, dtype=np.float32)
    scale_init = np.stack([s_normal, s1_all, s2_all], axis=1).astype(np.float32)
    scale_init = np.clip(scale_init, 1e-6, 0.5)

    # ---- 9. rotation init (align to face normal) ----
    v1_init = verts[fid_draw][:, 1, :] - verts[fid_draw][:, 0, :]
    v1_init = v1_init - (v1_init * fn_draw).sum(-1, keepdims=True) * fn_draw
    v1_init = v1_init / (np.linalg.norm(v1_init, axis=-1, keepdims=True) + 1e-12)
    v2_init = np.cross(fn_draw, v1_init)
    v2_init = v2_init / (np.linalg.norm(v2_init, axis=-1, keepdims=True) + 1e-12)
    R_mat   = np.stack([v1_init, v2_init, fn_draw], axis=-1).astype(np.float32)

    trace   = R_mat[:, 0, 0] + R_mat[:, 1, 1] + R_mat[:, 2, 2]
    sq      = np.sqrt(np.clip(1.0 + trace, 1e-12, None))
    rot_init= np.empty((M, 4), dtype=np.float32)
    rot_init[:, 0] = 0.5 * sq
    rot_init[:, 1] = (R_mat[:, 2, 1] - R_mat[:, 1, 2]) / (2.0 * sq)
    rot_init[:, 2] = (R_mat[:, 0, 2] - R_mat[:, 2, 0]) / (2.0 * sq)
    rot_init[:, 3] = (R_mat[:, 1, 0] - R_mat[:, 0, 1]) / (2.0 * sq)
    rn      = np.linalg.norm(rot_init, axis=-1, keepdims=True) + 1e-12
    rot_init = (rot_init / rn).astype(np.float32)

    # ---- 10. seed_idx ----
    first_seed = -np.ones(F, dtype=np.int64)
    seen = np.zeros(F, dtype=bool)
    for i, f in enumerate(fid_draw):
        if not seen[f]:
            first_seed[f] = i
            seen[f] = True
    seed_idx_out = first_seed[fid_draw]

    return {
        "xyz":           torch.from_numpy(xyz_draw),
        "normal":        torch.from_numpy(fn_draw),
        "face_id":       torch.from_numpy(fid_draw),
        "barycentric":   None,
        "seed_idx":      torch.from_numpy(seed_idx_out),
        "complexity":    torch.from_numpy(comp_draw),
        "color_draw":    torch.from_numpy(color_draw),
        "scale_init":    torch.from_numpy(scale_init),
        "rotation_init": torch.from_numpy(rot_init),
    }


def _face_interp_normal(face_ids, mesh_vertices, mesh_faces):
    """Compute per-face normals for a batch of face IDs."""
    verts = mesh_vertices[mesh_faces]
    e1 = verts[:, 1] - verts[:, 0]
    e2 = verts[:, 2] - verts[:, 0]
    fn = np.cross(e1, e2)
    fn = fn / (np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-8)
    return fn[face_ids].astype(np.float32)
