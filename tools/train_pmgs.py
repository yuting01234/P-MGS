#!/usr/bin/env python3
"""Execute the recorded PMGS E4 trainer with a portable initialization audit."""
import argparse
import os
from pathlib import Path
import runpy
import sys
import time

import _bootstrap
from pmgs.allocation import camera_signature, initial_fingerprints, load_allocation, matching_allocation, patch_sampler
from pmgs.workflow import ROOT, infer_scene, require_dataset, save_status, set_memory_limit, trainer_args


def install_loss_telemetry(output, status):
    """Observe original losses without consuming RNG or replacing gradients."""
    import json
    from utils import loss_utils

    handle = (output / "loss_components.jsonl").open("x", buffering=1)
    names = ("l1_loss", "ssim", "coarse_to_fine_blur_loss", "meshgs_projection_loss",
             "meshgs_normal_consistency_loss", "meshgs_scale_regularization")
    for name in names:
        original = getattr(loss_utils, name)

        def wrapped(*args, _original=original, _name=name, **kwargs):
            value = _original(*args, **kwargs)
            local = sys._getframe(1).f_locals
            if "gaussians" in local and "iteration" in local:
                iteration, model = int(local["iteration"]), local["gaussians"]
                status["peak_GS"] = max(status.get("peak_GS", 0), len(model.get_xyz))
                if iteration % 500 == 0 or iteration in (1, 999, 1000, 2999, 3000, 4999, 5000, 9999, 10000, 29999):
                    weight = float(local["opt"].lambda_dssim)
                    weighted = ((1 - weight) * value if _name == "l1_loss" else
                                weight * (1 - value) if _name == "ssim" else value)
                    handle.write(json.dumps(dict(iteration=iteration, component=_name,
                                                 weighted_value=float(weighted.detach()), GS=len(model.get_xyz))) + "\n")
            return value

        setattr(loss_utils, name, wrapped)
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--scene", help="Optional scene label; otherwise infer from --source-path")
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocation", choices=("auto", "qgain", "adaptive"), default="auto")
    parser.add_argument("--allow-initialization-drift", action="store_true")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    dataset, output = args.source_path.expanduser().resolve(), args.output.expanduser().resolve()
    args.scene = infer_scene(dataset, args.scene)
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError("Choose an output directory separate from the source dataset")
    if output == ROOT or output in ROOT.parents:
        raise ValueError("Choose a dedicated run directory, rather than the source project root")
    require_dataset(dataset)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("qgain_full_status.json", "run_status.json", "dataset_private", "point_cloud")):
        raise FileExistsError("Use a fresh run directory: " + str(output))
    set_memory_limit()
    import numpy as np
    import torch
    from games.mesh_splatting.init import adaptive_sampler, pipeline
    import games.mesh_splatting.scene.dataset_readers as readers

    # Dataset readers generate points3d.ply; isolate those writes from inputs.
    private = output / "dataset_private"
    private.mkdir()
    for item in dataset.iterdir():
        if not item.name.startswith(".") and not item.name.startswith("points3d"):
            (private / item.name).symlink_to(item, target_is_directory=item.is_dir())
    status = dict(status="preparing", scene=args.scene, method="PMGS", historical_configuration="E4_retained",
                  original_dataset=str(dataset), source_path=str(dataset), source_project=str(ROOT),
                  gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "default"),
                  memory_fraction=os.environ.get("PMGS_MEMORY_FRACTION"), allocation_requested=args.allocation,
                  allocation="pending_signature_check",
                  exact_face_counts_verified=False, initialization_drift_allowed=args.allow_initialization_drift,
                  extra_loss_weights=dict(blur_loss_weight=0.0, frequency_loss_weight=0.0, meshgs_proj_weight=0.01,
                                          meshgs_nc_weight=0.005, meshgs_scale_lam_min=0.005, meshgs_scale_lam_max=0.005),
                  lambda_dssim=0.3, start_epoch=time.time())
    original_build = readers.build_mesh_guided_point_cloud

    def build(mesh, cameras, **kwargs):
        measured = camera_signature(mesh, cameras)
        counts = entry = None
        matched_scene = matching_allocation(measured) if args.allocation != "adaptive" else None
        if args.allocation == "qgain" and matched_scene is None:
            raise ValueError("No bundled Qgain allocation matches these mesh/camera/image signatures. Use the corresponding original inputs, or explicitly choose --allocation auto or adaptive.")
        if matched_scene is not None:
            counts, entry = load_allocation(matched_scene)
            pipeline.adaptive_gaussian_sampling = patch_sampler(adaptive_sampler, counts)
            status.update(allocation="qgain", allocation_reference_scene=matched_scene,
                          allocation_selection_reason="Exact mesh/camera/image signature match to a bundled plan")
            print("QGAIN_SIGNATURE_MATCHED", matched_scene, flush=True)
        else:
            status.update(allocation="adaptive", allocation_reference_scene=None,
                          allocation_selection_reason=("Explicit adaptive selection" if args.allocation == "adaptive" else
                                                       "No exact bundled mesh/camera/image signature match"))
            print("ADAPTIVE_INITIALIZATION_SELECTED", status["allocation_selection_reason"], flush=True)
        save_status(output, status)
        if entry is not None:
            if kwargs["seed"] != 0 or kwargs["num_per_face"] != 3 or kwargs["sampler"] != "adaptive":
                raise ValueError("The published Qgain plan requires adaptive initialization, seed 0, and budget 3F")
        result = original_build(mesh, cameras, **kwargs)
        fingerprints = initial_fingerprints(result)
        status.update(GS_initial=len(result["xyz"]), train_views=len(cameras), input_signature=measured,
                      initial_fingerprints=fingerprints)
        if counts is not None:
            actual = np.bincount(result["face_id"].cpu().numpy(), minlength=len(counts))
            if not np.array_equal(actual, counts):
                raise RuntimeError("Initializer changed the precomputed Qgain face counts")
            matches = fingerprints == entry["initial_fingerprints"]
            status.update(exact_face_counts_verified=True, max_GS_per_face=int(counts.max()),
                          allocation_sha256=entry["sha256"], initial_tensors_match_reference=matches)
            if not matches and not args.allow_initialization_drift:
                raise RuntimeError("Initial tensor hashes differ from the published E4 reference. Check dependency versions, or explicitly use --allow-initialization-drift; input signature and face counts have already passed.")
            if not matches:
                print("INITIALIZATION_DRIFT_EXPLICITLY_ALLOWED", flush=True)
        status["status"] = "initialized"
        save_status(output, status)
        print("PMGS_INITIALIZATION_VERIFIED", args.scene, status["GS_initial"], flush=True)
        return result

    readers.build_mesh_guided_point_cloud = build
    save_status(output, status)
    telemetry = install_loss_telemetry(output, status)
    sys.argv = [str(ROOT / "train.py")] + trainer_args(private, output, args.port)
    try:
        runpy.run_path(str(ROOT / "train.py"), run_name="__main__")
        checkpoint = output / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        if not checkpoint.is_file() or "GS_initial" not in status:
            raise RuntimeError("Training returned without the verified initialization and final checkpoint")
        status.update(status="complete", finished_epoch=time.time(),
                      peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated() / 2 ** 20)
        save_status(output, status)
    except BaseException as error:
        status.update(status="failed", error=repr(error), finished_epoch=time.time())
        save_status(output, status)
        raise
    finally:
        telemetry.close()


if __name__ == "__main__":
    main()
