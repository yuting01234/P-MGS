#!/usr/bin/env python3
"""Render a PMGS checkpoint and evaluate float RGB with the original E4 metrics."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

import _bootstrap
from pmgs.workflow import atomic_json, checkpoint_paths, infer_scene, load_status, read_test_cameras, set_memory_limit


def main(argv=None, render_only_default=False):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-path", "--output", dest="model_path", type=Path, required=True,
                        help="Run directory, iteration directory, or point_cloud.ply")
    parser.add_argument("--source-path", type=Path, help="Dataset scene directory; otherwise read training status")
    parser.add_argument("--destination", type=Path, help="Write results here instead of inside the training run")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--gpu", help="CUDA device index/UUID; otherwise respect CUDA_VISIBLE_DEVICES")
    parser.add_argument("--memory-fraction", type=float)
    parser.add_argument("--max-views", type=int, default=0, help="0 evaluates all views; positive values mark a partial smoke check")
    parser.add_argument("--render-only", action="store_true", default=render_only_default,
                        help="Save rendered images without computing metrics or loading LPIPS")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace evaluation reports and rendered PNGs")
    args = parser.parse_args(argv)
    if args.max_views < 0:
        parser.error("--max-views must be nonnegative")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run, weights = checkpoint_paths(args.model_path, args.iteration)
    try:
        status = load_status(run)
    except FileNotFoundError:
        status = {}
    source = args.source_path or status.get("original_dataset") or status.get("source_path")
    if not source:
        parser.error("Provide --source-path when the checkpoint has no dataset path in its run status")
    source = Path(source).expanduser().resolve()
    if not any((source / name).is_file() for name in ("transforms_test.json", "sfm_scene.json")):
        raise FileNotFoundError("No supported test cameras (transforms_test.json or sfm_scene.json) in " + str(source))
    destination = (args.destination or run).expanduser().resolve()
    image_root = destination / "test" / ("ours_" + str(args.iteration))
    report_path = destination / ("render_status.json" if args.render_only else "evaluation.json")
    if not args.overwrite and (report_path.exists() or image_root.exists()):
        raise FileExistsError("Evaluation output already exists; use --destination NEW_DIRECTORY or explicitly --overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    renders, ground_truth = image_root / "renders", image_root / "gt"
    folders = (renders,) if args.render_only else (renders, ground_truth)
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for image in folder.glob("*.png"):
                image.unlink()
    import numpy as np
    import torch
    from PIL import Image
    from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
    from renderer.gaussian_renderer import render
    from utils.camera_utils import cameraList_from_camInfos

    set_memory_limit(args.memory_fraction)
    start = time.time()
    info = dict(status="rendering" if args.render_only else "evaluating", scene=status.get("scene", infer_scene(source)), method="PMGS",
                render_only=args.render_only,
                iteration=args.iteration, model_path=str(weights), source_path=str(source), views_finished=0)
    atomic_json(report_path, info)
    try:
        with torch.no_grad():
            model = GaussianMeshModel(3)
            model.load_ply(str(weights / "point_cloud.ply"))
            for name in ("_xyz", "_scaling", "_rotation", "_opacity"):
                if not bool(torch.isfinite(getattr(model, name)).all()):
                    raise ValueError("Nonfinite model parameter: " + name)
            if not args.render_only:
                from lpipsPyTorch.modules.lpips import LPIPS
                from utils.image_utils import psnr
                from utils.loss_utils import ssim
                perceptual = LPIPS("vgg", "0.1").cuda().eval()
            camera_infos = read_test_cameras(source)
            available = len(camera_infos)
            if not available:
                raise ValueError("The dataset has no test cameras")
            if args.max_views:
                camera_infos = camera_infos[:args.max_views]
            partial = len(camera_infos) < available
            cameras = cameraList_from_camInfos(camera_infos, 1., SimpleNamespace(resolution=1, data_device="cpu"))
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            background = torch.zeros(3, device="cuda")
            rows = []
            for index, camera in enumerate(cameras):
                image = render(camera, model, pipe, background)["render"].clamp(0, 1)
                row = dict(frame_index=index, image_name=camera.image_name, filename="%05d.png" % index)
                if not args.render_only:
                    target = camera.original_image.cuda().clamp(0, 1)
                    row.update(PSNR=float(psnr(image, target).mean().double()),
                               SSIM=float(ssim(image, target).mean().double()),
                               LPIPS_VGG=float(perceptual(image[None] * 2 - 1, target[None] * 2 - 1).mean().double()))
                if not args.render_only and not all(np.isfinite(row[key]) for key in ("PSNR", "SSIM", "LPIPS_VGG")):
                    raise ValueError("Nonfinite metric at view " + str(index))
                rows.append(row)
                saved_images = ((renders, image),) if args.render_only else ((renders, image), (ground_truth, target))
                for folder, tensor in saved_images:
                    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255 + .5).astype(np.uint8)
                    Image.fromarray(array).save(folder / row["filename"])
                if (index + 1) % 20 == 0 or index + 1 == len(cameras):
                    info["views_finished"] = index + 1
                    atomic_json(report_path, info)
                    print("RENDERING" if args.render_only else "EVALUATING", info["scene"], index + 1, "/", len(cameras), flush=True)
            means = {} if args.render_only else {
                key: float(np.mean([row[key] for row in rows])) for key in ("PSNR", "SSIM", "LPIPS_VGG")}
            match = None
            log_path = run / "train.log"
            if not args.render_only and not partial and log_path.is_file():
                expression = r"\[ITER " + str(args.iteration) + r"\] Evaluating test:.*?PSNR ([0-9.eE+\-]+)"
                expected = re.findall(expression, log_path.read_text(errors="replace"))
                if expected:
                    match = abs(means["PSNR"] - float(expected[-1])) < 1e-5
            with (image_root / "per_view.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            info.update(status="complete", partial=partial, views_finished=len(rows), test_views=available,
                        GS_init=status.get("GS_initial"), GS_final=len(model.get_xyz),
                        background="black", resolution="native",
                        train_log_PSNR_match=match, elapsed_seconds=time.time() - start)
            if not args.render_only:
                info.update(PSNR=means["PSNR"], SSIM=means["SSIM"], LPIPS=means["LPIPS_VGG"],
                            metric_images="float RGB before PNG quantization",
                            psnr_definition="Original project: average per-channel R/G/B PSNR, then average views",
                            lpips_definition="v0.1 VGG RGB [-1,1]")
            atomic_json(report_path, info)
            print("RENDER_COMPLETE" if args.render_only else "EVALUATION_COMPLETE", json.dumps(info), flush=True)
    except BaseException as error:
        info.update(status="failed", error=repr(error))
        atomic_json(report_path, info)
        raise


if __name__ == "__main__":
    main()
