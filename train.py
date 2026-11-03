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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, progressive_frequency_loss, coarse_to_fine_blur_loss
from utils.loss_utils import (
    meshgs_normal_consistency_loss,
    meshgs_scale_regularization,
    meshgs_projection_loss,
)
from utils.system_utils import searchForMaxIteration
from renderer.gaussian_renderer import render, network_gui
import sys
from scene import Scene
from games import (
    optimizationParamTypeCallbacks,
    gaussianModel
)

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams


def _apply_opt_overrides(args):
    """Apply command-line optimization param overrides to args namespace."""
    overrides = {
        'feature_lr': getattr(args, 'opt_feature_lr', None),
        'lambda_dssim': getattr(args, 'opt_lambda_dssim', None),
        'densify_grad_threshold': getattr(args, 'opt_densify_grad_threshold', None),
        'densify_from_iter': getattr(args, 'opt_densify_from_iter', None),
        'densify_until_iter': getattr(args, 'opt_densify_until_iter', None),
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(args, key, value)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(gs_type, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, save_xyz):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = gaussianModel[gs_type](dataset.sh_degree)

    # SuGaR-style refinement: load a trained checkpoint into the model
    # instead of re-initializing from the dataset.
    if getattr(args, "freeze_geometry", False) and getattr(args, "freeze_geometry_dir", None):
        scene = Scene(dataset, gaussians)
        # Allow an explicit PLY path to override auto-detection (useful for
        # old-format dirs where searchForMaxIteration picks a .pt instead).
        if getattr(args, "freeze_geometry_ply", None):
            ckpt_ply = args.freeze_geometry_ply
        else:
            ckpt_ply = os.path.join(
                args.freeze_geometry_dir, "point_cloud",
                "iteration_" + str(searchForMaxIteration(
                    os.path.join(args.freeze_geometry_dir, "point_cloud"))),
                "point_cloud.ply",
            )
        print(f"[Freeze] Loading checkpoint PLY: {ckpt_ply}")
        gaussians.load_ply(ckpt_ply)
        gaussians.training_setup(opt)  # rebuild optimizer for loaded params
        if hasattr(gaussians, "rebind_for_freeze"):
            gaussians.rebind_for_freeze(scene.scene_info)
        gaussians.freeze_geometry()
        print("[Freeze] Geometry params frozen (xyz/scaling/rotation). "
              "Training SH + opacity only.")
    else:
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss + progressive_frequency_loss(
            image, gt_image, iteration,
            args.frequency_loss_start, args.frequency_loss_end,
            args.frequency_loss_weight, args.frequency_lowpass_ratio,
        )
        # Opti3DGS-style coarse-to-fine: train on blurred GT first
        loss = loss + coarse_to_fine_blur_loss(
            image, gt_image, iteration,
            args.blur_loss_start, args.blur_loss_fade_iter,
            args.blur_kernel_init, args.blur_loss_weight,
        )

        # === MeshGS-style surface alignment losses (only for mesh-guided) ===
        # All three losses are gated by feature flags so they default to off
        # and can be tuned from the command line.
        if getattr(gaussians, "_is_mesh_guided", False) and gaussians.face_id is not None:
            # Re-sync face_id defensively (in case any legacy code-path grew the
            # Gaussian count without going through our override).
            if gaussians.face_id.shape[0] != gaussians.get_xyz.shape[0]:
                gaussians._reconcile_face_id()

            if args.meshgs_proj_weight > 0.0 and iteration >= args.meshgs_proj_start:
                loss = loss + meshgs_projection_loss(
                    gaussians, gaussians.face_id, gaussians.face_verts,
                    weight=args.meshgs_proj_weight,
                    max_dist=args.meshgs_proj_max_dist,
                )
            if args.meshgs_nc_weight > 0.0 and iteration >= args.meshgs_nc_start:
                loss = loss + meshgs_normal_consistency_loss(
                    gaussians, gaussians.face_id, gaussians.face_normals,
                    weight=args.meshgs_nc_weight,
                )
            if (args.meshgs_scale_lam_min > 0.0 or args.meshgs_scale_lam_max > 0.0) \
                    and iteration >= args.meshgs_scale_start:
                loss = loss + meshgs_scale_regularization(
                    gaussians,
                    rho=args.meshgs_scale_rho if args.meshgs_scale_rho > 0
                        else gaussians.face_avg_radius,
                    lam_min=args.meshgs_scale_lam_min,
                    lam_max=args.meshgs_scale_lam_max,
                )

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification enabled for gs_mesh_guided (GaussianMeshModel correctly implements all required methods)
            do_densify = (args.gs_type == "gs" or args.gs_type == "gs_flat" or args.gs_type == "gs_mesh_guided")
            # SuGaR-style refinement: geometry is frozen, no densification.
            if getattr(args, "freeze_geometry", False):
                do_densify = False
            if do_densify and iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--gs_type', type=str, default="gs_mesh")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--meshes", nargs="+", type=str, default=[])
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--frequency_loss_weight', type=float, default=0.01)  # Enabled by default
    parser.add_argument('--frequency_loss_start', type=int, default=1000)
    parser.add_argument('--frequency_loss_end', type=int, default=25000)  # Extended to 25k
    parser.add_argument('--frequency_lowpass_ratio', type=float, default=0.25)
    # Opti3DGS-style coarse-to-fine blur loss
    parser.add_argument('--blur_loss_weight', type=float, default=1.0,
                        help='weight for blur-to-GT L1 loss (1.0=full weight, 0=off)')
    parser.add_argument('--blur_loss_start', type=int, default=1,
                        help='iteration to start blur loss')
    parser.add_argument('--blur_loss_fade_iter', type=int, default=3000,
                        help='iteration to fade out blur (step-function, 0=always blur)')
    parser.add_argument('--blur_kernel_init', type=int, default=31,
                        help='initial Gaussian blur kernel size (31=very blurry)')

    # -------- MeshGS surface-alignment loss (only effective for gs_mesh_guided) --------
    # Paper: Choi et al., "MeshGS: Adaptive Mesh-Aligned Gaussian Splatting
    # for High-Quality Rendering", ACCV 2024
    # L_proj: pin Gaussian center to its face plane
    parser.add_argument('--meshgs_proj_weight', type=float, default=0.0,
                        help='weight for projection loss (0=off)')
    parser.add_argument('--meshgs_proj_start', type=int, default=1000,
                        help='iteration to start projection loss')
    parser.add_argument('--meshgs_proj_max_dist', type=float, default=-1.0,
                        help='only penalize Gaussians farther than this from face '
                             '(negative = apply to all)')
    # L_nc: align Gaussian normal to face normal
    parser.add_argument('--meshgs_nc_weight', type=float, default=0.0,
                        help='weight for normal-consistency loss (0=off)')
    parser.add_argument('--meshgs_nc_start', type=int, default=1000,
                        help='iteration to start normal-consistency loss')
    # L_scale: keep min scale small, max scale <= rho
    parser.add_argument('--meshgs_scale_lam_min', type=float, default=0.0,
                        help='lambda_min for scale regularization (0=off)')
    parser.add_argument('--meshgs_scale_lam_max', type=float, default=0.0,
                        help='lambda_max for scale regularization (0=off)')
    parser.add_argument('--meshgs_scale_start', type=int, default=1000,
                        help='iteration to start scale regularization')
    parser.add_argument('--meshgs_scale_rho', type=float, default=-1.0,
                        help='target max scale; <=0 means use face average radius')
    # Advanced optimization params - use custom prefix to avoid conflicts
    parser.add_argument('--opt_feature_lr', type=float, default=None, help='Override feature learning rate')
    parser.add_argument('--opt_lambda_dssim', type=float, default=None, help='Override DSSIM loss weight')
    parser.add_argument('--opt_densify_grad_threshold', type=float, default=None, help='Override densification gradient threshold')
    parser.add_argument('--opt_densify_from_iter', type=int, default=None, help='Override densification start iteration')
    parser.add_argument('--opt_densify_until_iter', type=int, default=None, help='Override densification end iteration')
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--save_xyz", action='store_true')

    # -------- SuGaR-style "Mesh Binding Refinement" mode --------
    # Load a trained model from a checkpoint directory and freeze all
    # geometry-related parameters (xyz, scaling, rotation). Only SH features
    # and opacity remain trainable. This implements the SuGaR
    # refinement stage: re-fit appearance to a fixed mesh+Gaussian scaffold.
    parser.add_argument("--freeze_geometry", action='store_true',
                        help='Load a trained checkpoint and freeze xyz/scaling/rotation. '
                             'Only SH (features_dc, features_rest) and opacity remain trainable.')
    parser.add_argument("--freeze_geometry_dir", type=str, default=None,
                        help='Path to a trained model_path containing point_cloud/iteration_*/point_cloud.ply. '
                             'When set (with --freeze_geometry), the model is loaded from this directory instead '
                             'of being re-initialized.')
    parser.add_argument("--freeze_geometry_ply", type=str, default=None,
                        help='Absolute path to the checkpoint PLY file to load. '
                             'If not set, auto-detected from freeze_geometry_dir. '
                             'Useful when the dir has .pt files that confuse auto-detection.')

    # -------- mesh-guided init params (only used when --gs_type gs_mesh_guided) --------
    # Target Gaussian count = num_faces * num_per_face.  For chair
    # (258k faces) at the default num_per_face=8 this gives ~2.07M Gaussians,
    # which is slightly more than the original GaMeS 5/triangle = 1.29M.
    parser.add_argument("--mg_n_seed", type=int, default=100000)
    parser.add_argument("--mg_num_per_face", type=int, default=8)
    parser.add_argument("--mg_complexity_power", type=float, default=1.0)
    parser.add_argument("--mg_noise_scale_ratio", type=float, default=0.02)
    parser.add_argument("--mg_base_size", type=float, default=0.001)
    parser.add_argument("--mg_sobel_weight", type=float, default=0.5)
    parser.add_argument("--mg_var_weight", type=float, default=0.5)
    parser.add_argument("--mg_var_kernel", type=int, default=5)
    parser.add_argument("--mg_visibility_power", type=float, default=1.0)
    parser.add_argument("--mg_seed", type=int, default=0)
    # Sampler: 'visual' (2D pixel-space, view-aware) or 'adaptive' (face-area)
    parser.add_argument("--mg_sampler", type=str, default="adaptive",
                        choices=["adaptive", "visual", "patch2d", "legacy"])
    parser.add_argument("--mg_visual_n_seed", type=int, default=300000)
    parser.add_argument("--mg_patch2d_n_seed", type=int, default=300000)
    parser.add_argument("--mg_patch2d_patch_size", type=int, default=5)
    parser.add_argument("--mg_patch2d_complexity_power", type=float, default=1.0)
    parser.add_argument("--mg_patch2d_visual_area_power", type=float, default=0.5)
    parser.add_argument("--mg_patch2d_min_coverage", type=int, default=3,
                        help="minimum Gaussians per mesh face")
    parser.add_argument("--mg_patch2d_stage1_extra_factor", type=float, default=2.0,
                        help="multiply min-coverage budget by this factor for Stage-1 draw")
    parser.add_argument("--mg_use_comp_scale_shrink", action="store_true", default=True,
                        help="CCC shrinks scale by (1 - 0.5*complexity); turn OFF for ablation")
    parser.add_argument("--mg_no_comp_scale_shrink", action="store_false",
                        dest="mg_use_comp_scale_shrink",
                        help="disable comp_factor scale shrink")
    parser.add_argument("--mg_scale_jitter_lo", type=float, default=None,
                        help="adaptive sampler: per-Gaussian scale multiplier "
                             "uniform(lo, hi). default=None (no jitter).")
    parser.add_argument("--mg_scale_jitter_hi", type=float, default=None)
    parser.add_argument("--mg_weight_mode", type=str, default="power",
                        choices=["power", "smooth"],
                        help="face weighting mode: "
                             "'power' = area*comp^power (default), "
                             "'smooth' = area*(1+alpha*comp)")
    parser.add_argument("--mg_weight_alpha", type=float, default=0.5,
                        help="alpha for --mg_weight_mode smooth")
    parser.add_argument("--mg_comp_shrink_alpha", type=float, default=0.5,
                        help="how aggressively comp shrinks scale "
                             "(0=no shrink, 0.5=default)")
    parser.add_argument("--mg_rotation_mode", type=str, default="face_tangent",
                        choices=["face_tangent", "identity"],
                        help="rotation init: 'face_tangent' (default) or "
                             "'identity' (legacy-style, free to rotate)")
    parser.add_argument("--mg_jitter_mode", type=str, default="tangent",
                        choices=["tangent", "full"],
                        help="position jitter: 'tangent' (along surface only) "
                             "or 'full' (3D noise, legacy-like)")


    # Parse twice: first to get gs_type, then for full args
    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.num_splats = args.num_splats
    lp.meshes = args.meshes
    lp.gs_type = args.gs_type

    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    # Apply command-line overrides for optimization params
    # These are applied after op.extract() to override the extracted opt
    _apply_opt_overrides(args)
    opt = op.extract(args)

    args.save_iterations.append(args.iterations)

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        args.gs_type,
        args, opt, pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, args.save_xyz
    )

    # All done
    print("\nTraining complete.")
