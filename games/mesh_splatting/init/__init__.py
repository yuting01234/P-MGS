#
# Copyright (C) 2024, Gmum
# Mesh-guided Gaussian Splatting Initialization module
#

from .mesh_sampler import sample_mesh_surface
from .complexity_extractor import compute_image_complexity
from .projector import (
    fov_to_intrinsic,
    camera_center_world,
    world_to_camera,
    project_to_pixel,
    sample_complexity_at_uv,
    visibility_weight,
)
from .complexity_fusion import fuse_complexity_to_surface
from .adaptive_sampler import (
    compute_face_complexity,
    compute_face_area,
    compute_face_sampling_weights,
    adaptive_gaussian_sampling,
    assign_initial_color,
)
from .legacy_sampler import legacy_random_sampling
from .pipeline import build_mesh_guided_point_cloud
