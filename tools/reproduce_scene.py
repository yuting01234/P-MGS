#!/usr/bin/env python3
"""Run the portable PMGS E4 pipeline: train, evaluate, diagnose, and bind."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import _bootstrap
from pmgs.workflow import ROOT, atomic_json, data_root, infer_scene, require_dataset, trainer_args


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--scene", help="Optional scene label; otherwise infer it from --source-path")
    parser.add_argument("--data-root", "--dataset-root", dest="data_root", type=Path, default=data_root(),
                        help="Directory containing scene directories (or set PMGS_DATA_ROOT)")
    parser.add_argument("--source-path", type=Path, help="Use this scene directory directly")
    parser.add_argument("--output", type=Path, help="A new run directory; default: output/<scene>/attempt_01")
    parser.add_argument("--gpu", help="CUDA device index/UUID; otherwise respect CUDA_VISIBLE_DEVICES")
    parser.add_argument("--memory-fraction", type=float, help="Optional per-process CUDA allocation limit in (0,1]")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, help="Override the training viewer port when running jobs concurrently")
    parser.add_argument("--allocation", choices=("auto", "qgain", "adaptive"), default="auto",
                        help="auto reuses a bundled Qgain plan only on an exact input signature match, otherwise records adaptive initialization")
    parser.add_argument("--allow-initialization-drift", action="store_true",
                        help="Explicitly allow initial tensor hash drift across dependency versions; data signatures and face counts remain strict")
    parser.add_argument("--binding-output-root", "--affine-output-root", dest="binding_output_root", type=Path,
                        help="Independent new-format binding root; default: bindings (custom runs get a unique subdirectory)")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete plan without creating files or using CUDA")
    args = parser.parse_args()
    if args.source_path is None and args.scene is None:
        parser.error("Provide --source-path, or --scene with --data-root")
    if args.threads < 1 or (args.memory_fraction is not None and not 0 < args.memory_fraction <= 1):
        parser.error("--threads must be positive and --memory-fraction must lie in (0,1]")
    return args


def main():
    args = arguments()
    dataset = (args.source_path or args.data_root / args.scene).expanduser().resolve()
    args.scene = infer_scene(dataset, args.scene)
    output = (args.output or ROOT / "output" / args.scene / "attempt_01").expanduser().resolve()
    if args.binding_output_root is not None:
        affine_root = args.binding_output_root.expanduser().resolve()
    elif output == (ROOT / "output" / args.scene / "attempt_01").resolve():
        affine_root = ROOT / "bindings"
    else:
        suffix = hashlib.sha256(str(output).encode()).hexdigest()[:8]
        affine_root = ROOT / "bindings" / (args.scene + "_" + output.name + "_" + suffix)
    if output == dataset or dataset in output.parents:
        raise ValueError("Choose an output directory outside the source dataset")
    if output.exists():
        raise FileExistsError("Choose a new output directory; this run already exists: " + str(output))
    command = [sys.executable, "-u", str(ROOT / "tools" / "train_pmgs.py"),
               "--scene", args.scene, "--source-path", str(dataset), "--output", str(output),
               "--allocation", args.allocation]
    if args.port is not None:
        command.extend(["--port", str(args.port)])
    if args.allow_initialization_drift:
        command.append("--allow-initialization-drift")
    steps = [("train", command)]
    if not args.train_only:
        steps.extend([
            ("evaluation", [sys.executable, "-u", str(ROOT / "tools" / "evaluate.py"), "--output", str(output)]),
            ("geometry", [sys.executable, "-u", str(ROOT / "tools" / "geometry_diagnostics.py"), "--output", str(output)]),
        ])
        steps.append(("binding", [sys.executable, "-u", str(ROOT / "scripts" / "post_bind_affine.py"),
                                   "--model-path", str(output), "--scenes", args.scene, "--output-root", str(affine_root),
                                   "--iteration", "30000", "--bind-only"]))
    plan = dict(method="PMGS", configuration="E4_retained", scene=args.scene, source_path=str(dataset),
                output=str(output), cwd=str(ROOT), gpu=args.gpu or os.environ.get("CUDA_VISIBLE_DEVICES", "default"),
                memory_fraction=args.memory_fraction, allocation_requested=args.allocation,
                binding_format="signed_normal_local_affine_v1", binding_output_root=str(affine_root),
                trainer_args=trainer_args(dataset, output, args.port),
                stages={name: shlex.join(cmd) for name, cmd in steps})
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    require_dataset(dataset)
    environment = os.environ.copy()
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.memory_fraction is not None:
        environment["PMGS_MEMORY_FRACTION"] = str(args.memory_fraction)
    environment.update(OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
                       MKL_NUM_THREADS=str(args.threads), PYTHONUNBUFFERED="1")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "REPLAY_PROTOCOL.json", plan)
    print("OUTPUT", output, flush=True)
    for name, cmd in steps:
        atomic_json(output / (name + ".command.json"), dict(command=cmd, cwd=str(ROOT), gpu=plan["gpu"]))
        log_path = output / (name + ".log")
        print("START", name, "LOG", log_path, flush=True)
        with log_path.open("x") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit("{} failed ({}); inspect {}".format(name, result.returncode, log_path))
        print("COMPLETE", name, flush=True)
    print("DONE", output, flush=True)


if __name__ == "__main__":
    main()
