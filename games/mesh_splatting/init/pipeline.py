#
# Mesh-guided Gaussian Splatting - Initialization module
# End-to-end pipeline assembling all sub-modules
#

import numpy as np
import torch

from .mesh_sampler import sample_mesh_surface
from .complexity_extractor import compute_image_complexity
from .complexity_fusion import fuse_complexity_to_surface
from .adaptive_sampler import adaptive_gaussian_sampling, assign_initial_color
from .visual_sampler import visual_gaussian_sampling
from .patch2d_sampler import sample_gaussians_from_2d_patches
from .legacy_sampler import legacy_random_sampling


def build_mesh_guided_point_cloud(
    mesh,
    train_cam_infos,
    *,
    n_seed=100000,                  # candidate surface samples (for complexity)
    num_per_face=8,                 # Gaussians per face (M = faces * num_per_face)
    complexity_power=1.0,           # >1 emphasizes complex regions
    noise_scale_ratio=0.02,
    base_size=0.02,
    position_lr_scale=1.0,
    sobel_weight=0.5, var_weight=0.5, var_kernel=5,
    visibility_power=1.0,
    sampler="adaptive",             # 'adaptive', 'visual', 'patch2d' or 'legacy'
    visual_n_seed=300000,         # candidate seed count for the visual sampler
    patch2d_n_seed=300000,        # surface seed count for patch2d sampler
    patch2d_patch_size=5,         # local patch size for texture complexity
    patch2d_complexity_power=1.0,
    patch2d_visual_area_power=0.5,
    patch2d_min_coverage=3,
    patch2d_stage1_extra_factor=2.0,
    use_comp_scale_shrink=True,        # CCC affects scale? (default True)
    comp_shrink_alpha=0.5,             # how aggressively comp shrinks scale
    scale_jitter_lo=None,              # adaptive sampler scale randomisation
    scale_jitter_hi=None,
    weight_mode="power",               # 'power' or 'smooth' face weight
    weight_alpha=0.5,                  # alpha for 'smooth' mode
    rotation_mode="face_tangent",      # 'face_tangent' or 'identity'
    jitter_mode="tangent",             # 'tangent' or 'full'
    seed=0,
):
    """
    End-to-end mesh-guided initialization.

    Steps:
        1.  sample_mesh_surface     -> candidate points (n_seed surface seeds)
        2.  compute_image_complexity  -> per-image complexity maps (adaptive/visual only)
        3.  fuse_complexity_to_surface -> per-seed complexity (adaptive/visual only)
        4.  Gaussian sampling
              sampler == 'adaptive' : multinomial over faces weighted by
                                      area * face_complexity
              sampler == 'visual'   : multinomial over (seed, view) pairs
                                      weighted by visual_area * pixel_complexity
                                      * visibility
              sampler == 'patch2d'  : project seeds to each view, sample pixels
                                      by patch_complexity on real image + visual_area
                                      + visibility; back-project to mesh surface.
                                      Includes coverage-guarantee fallback.
        5.  assign_initial_color      -> tiny random init color

    Returns a dict with all tensors the GaMeS model needs:
        xyz, normal, face_id, barycentric, complexity,
        scale_init, rotation_init, colors, seed_idx,
        vertices, faces, triangles.
    """
    num_faces = np.asarray(mesh.faces).shape[0]
    if sampler != "patch2d":
        target_M  = num_faces * num_per_face
        print(f"[MeshInit] Mesh has {num_faces} faces -> target M = {target_M} Gaussians")

    # ---- 1. surface candidates ----
    if sampler == "patch2d":
        n_seed_actual = patch2d_n_seed
    elif sampler == "visual":
        n_seed_actual = max(n_seed, visual_n_seed)
    else:
        n_seed_actual = n_seed
    print(f"[MeshInit] Sampling {n_seed_actual} surface points (sampler={sampler}) ...")
    sampled = sample_mesh_surface(mesh, num_samples=n_seed_actual, seed=seed)

    points_world  = sampled["xyz"].numpy()
    normals_world = sampled["normal"].numpy()
    face_id       = sampled["face_id"].numpy()
    triangles     = sampled["triangles"]
    barycentric   = sampled["barycentric"].numpy()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces    = np.asarray(mesh.faces,    dtype=np.int64)

    # ---- 2. per-image complexity maps (for adaptive/visual; not needed for patch2d) ----
    if sampler != "patch2d":
        print(f"[MeshInit] Computing per-image complexity maps ...")
        cmap_dict = {}
        for cam in train_cam_infos:
            if cam.image_name in cmap_dict:
                continue
            img = np.asarray(cam.image.convert("RGB"), dtype=np.float32) / 255.0
            cmap = compute_image_complexity(
                img,
                sobel_weight=sobel_weight,
                var_weight=var_weight,
                var_kernel=var_kernel,
            )
            cmap_dict[cam.image_name] = cmap

        # ---- 3. multi-view complexity fusion onto surface seeds ----
        print(f"[MeshInit] Fusing complexity across {len(train_cam_infos)} views ...")
        complexity, _, _ = fuse_complexity_to_surface(
            points_world, normals_world,
            train_cam_infos, cmap_dict,
            power=visibility_power,
        )

    # ---- 4. Gaussian sampling ----
    if sampler == "patch2d":
        print(f"[MeshInit] Sampling Gaussians via 2D-patch method ...")
        gauss = sample_gaussians_from_2d_patches(
            mesh,
            train_cam_infos,
            n_seed=patch2d_n_seed,
            min_coverage=patch2d_min_coverage,
            patch_size=patch2d_patch_size,
            sobel_weight=sobel_weight,
            var_weight=var_weight,
            var_kernel=var_kernel,
            complexity_power=patch2d_complexity_power,
            visual_area_power=patch2d_visual_area_power,
            stage1_extra_factor=patch2d_stage1_extra_factor,
            seed=seed + 1,
        )
    elif sampler == "visual":
        print(f"[MeshInit] Sampling {target_M} Gaussians in 2D pixel space "
              f"(visual_area * pixel_complexity^{complexity_power} * visibility) ...")
        gauss = visual_gaussian_sampling(
            mesh,
            points_world, normals_world, face_id, complexity,
            train_cam_infos, cmap_dict,
            target_M=target_M,
            complexity_power=complexity_power,
            seed=seed + 1,
        )
    elif sampler == "legacy":
        # Ablation: pure-legacy GaMeS-style sampling.
        # - no complexity_extractor / complexity_fusion
        # - no adaptive face weighting (uniform-by-area only)
        # - no patch2d sampler
        # - no normal-aligned jitter
        print(f"[MeshInit] (ablation) complexity_extractor + complexity_fusion OFF")
        print(f"[MeshInit] Sampling {target_M} Gaussians via legacy_random_sampling "
              f"(uniform face area, random scale, no complexity, no jitter) ...")
        gauss = legacy_random_sampling(
            mesh,
            points_world, normals_world, face_id, complexity,
            num_per_face=num_per_face,
            base_size=base_size,
            noise_scale_ratio=noise_scale_ratio,
            seed=seed + 1,
            position_lr_scale=position_lr_scale,
        )
    else:
        print(f"[MeshInit] Sampling {target_M} Gaussians "
              f"({num_per_face} / face, weighting={weight_mode}"
              f"{', alpha=' + str(weight_alpha) if weight_mode == 'smooth' else ', power=' + str(complexity_power)}, "
              f"comp_scale_shrink={use_comp_scale_shrink}) ...")
        gauss = adaptive_gaussian_sampling(
            mesh,
            points_world, normals_world, face_id, complexity,
            num_per_face=num_per_face,
            complexity_power=complexity_power,
            noise_scale_ratio=noise_scale_ratio,
            base_size=base_size,
            seed=seed + 1,
            position_lr_scale=position_lr_scale,
            use_comp_scale_shrink=use_comp_scale_shrink,
            comp_shrink_alpha=comp_shrink_alpha,
            scale_jitter_lo=scale_jitter_lo,
            scale_jitter_hi=scale_jitter_hi,
            weight_mode=weight_mode,
            weight_alpha=weight_alpha,
            rotation_mode=rotation_mode,
            jitter_mode=jitter_mode,
        )

    M = gauss["xyz"].shape[0]
    print(f"[MeshInit] Produced {M} Gaussians.")

    # Use real color from 2D-patch sampling when available; else random
    if "color_draw" in gauss:
        colors = gauss["color_draw"].numpy().astype(np.float32)
        print(f"[MeshInit] Using real colors from 2D-patch sampling")
    else:
        colors = assign_initial_color(gauss, mode="random")

    return dict(
        xyz=gauss["xyz"],
        normal=gauss["normal"],
        face_id=gauss["face_id"],
        barycentric=barycentric,
        complexity=gauss["complexity"],
        scale_init=gauss["scale_init"],
        rotation_init=gauss["rotation_init"],
        colors=colors,
        seed_idx=gauss["seed_idx"],
        vertices=torch.from_numpy(vertices),
        faces=torch.from_numpy(faces),
        triangles=triangles,
    )
