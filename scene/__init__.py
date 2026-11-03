#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from games.scenes import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param args: full argparse Namespace (carries both ModelParams fields
                     like `source_path`/`sh_degree` AND the optional
                     `--mg_*` mesh-guided flags).  We deliberately pass the
                     *raw* Namespace (not `lp.extract(args)`) so the
                     mesh-guided branch can see `args.mg_num_per_face` etc.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.gs_type == "gs_multi_mesh":
                scene_info = sceneLoadTypeCallbacks["Colmap_Mesh"](
                    args.source_path, args.images, args.eval, args.num_splats, args.meshes
                )
            else:
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif (os.path.exists(os.path.join(args.source_path, "transforms_train.json"))
              or os.path.exists(os.path.join(args.source_path, "sfm_scene.json"))):
            if os.path.exists(os.path.join(args.source_path, "sfm_scene.json")) and args.gs_type != "gs_mesh_guided":
                raise ValueError("Calibrated DTU input requires gs_mesh_guided")
            if args.gs_type == "gs_mesh":
                print("Found transforms_train.json file, assuming Blender_Mesh data set!")
                scene_info = sceneLoadTypeCallbacks["Blender_Mesh"](
                    args.source_path, args.white_background, args.eval, args.num_splats[0]
                )
            elif args.gs_type == "gs_mesh_guided":
                print("Reading mesh-guided scene (Blender transforms or calibrated DTU)")
                scene_info = sceneLoadTypeCallbacks["Blender_Mesh_Guided"](
                    args.source_path, args.white_background, args.eval,
                    n_seed=getattr(args, "mg_n_seed", 100000),
                    num_per_face=getattr(args, "mg_num_per_face", 8),
                    complexity_power=getattr(args, "mg_complexity_power", 1.0),
                    noise_scale_ratio=getattr(args, "mg_noise_scale_ratio", 0.02),
                    base_size=getattr(args, "mg_base_size", 0.001),
                    sobel_weight=getattr(args, "mg_sobel_weight", 0.5),
                    var_weight=getattr(args, "mg_var_weight", 0.5),
                    var_kernel=getattr(args, "mg_var_kernel", 5),
                    visibility_power=getattr(args, "mg_visibility_power", 1.0),
                    seed=getattr(args, "mg_seed", 0),
                    sampler=getattr(args, "mg_sampler", "adaptive"),
                    visual_n_seed=getattr(args, "mg_visual_n_seed", 300000),
                    patch2d_n_seed=getattr(args, "mg_patch2d_n_seed", 300000),
                    patch2d_patch_size=getattr(args, "mg_patch2d_patch_size", 5),
                    patch2d_complexity_power=getattr(args, "mg_patch2d_complexity_power", 1.0),
                    patch2d_visual_area_power=getattr(args, "mg_patch2d_visual_area_power", 0.5),
                    patch2d_min_coverage=getattr(args, "mg_patch2d_min_coverage", 3),
                    patch2d_stage1_extra_factor=getattr(args, "mg_patch2d_stage1_extra_factor", 2.0),
                    use_comp_scale_shrink=getattr(args, "mg_use_comp_scale_shrink", True),
                    comp_shrink_alpha=getattr(args, "mg_comp_shrink_alpha", 0.5),
                    scale_jitter_lo=getattr(args, "mg_scale_jitter_lo", None),
                    scale_jitter_hi=getattr(args, "mg_scale_jitter_hi", None),
                    weight_mode=getattr(args, "mg_weight_mode", "power"),
                    weight_alpha=getattr(args, "mg_weight_alpha", 0.5),
                    rotation_mode=getattr(args, "mg_rotation_mode", "face_tangent"),
                    jitter_mode=getattr(args, "mg_jitter_mode", "tangent"),
                )
            else:
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            if args.gs_type == "gs_multi_mesh":
                for i, ply_path in enumerate(scene_info.ply_path):
                    with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, f"input_{i}.ply") , 'wb') as dest_file:
                        dest_file.write(src_file.read())
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # Stash scene_info so callers (e.g. the SuGaR-style freeze-geometry
        # path in train.py) can re-bind mesh attributes after `load_ply`
        # wipes them out.
        self.scene_info = scene_info

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                          "point_cloud",
                                                          "iteration_" + str(self.loaded_iter),
                                                          "point_cloud.ply"))
            self.gaussians.point_cloud = scene_info.point_cloud
            if args.gs_type == "gs_mesh":
                self.gaussians.triangles = scene_info.point_cloud.triangles
        else:
            if getattr(scene_info, "is_mesh_guided", False):
                # pick up the init dict from the module-level registry
                from games.mesh_splatting.scene.dataset_readers import take_mesh_guided_init
                init_dict = take_mesh_guided_init(scene_info.ply_path)
                if init_dict is None:
                    raise RuntimeError(
                        "Mesh-guided init dict not found for ply_path "
                        f"{scene_info.ply_path}. Did readNerfSyntheticMeshGuidedInfo run?"
                    )
                self.gaussians.create_from_mesh_guided_init(
                    init_dict, self.cameras_extent
                )
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
