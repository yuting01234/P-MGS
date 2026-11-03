"""Paths, recorded E4 options, and run metadata shared by PMGS tools.

This module deliberately imports only the Python standard library so command
previews and configuration inspection work without CUDA or PyTorch installed.
"""
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENES = ("chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def data_root():
    return Path(os.environ.get("PMGS_DATA_ROOT", ROOT / "data" / "nerf_synthetic"))


def e4_config():
    return json.loads((ROOT / "configs" / "e4.json").read_text())


def allocation_manifest():
    return json.loads((ROOT / "configs" / "allocations" / "manifest.json").read_text())


def trainer_args(dataset, output, port=None):
    args = e4_config()["trainer_args"].copy()
    if port is not None:
        args[args.index("--port") + 1] = str(port)
    return args + ["--source_path", str(dataset), "--model_path", str(output)]


def require_dataset(dataset):
    dataset = Path(dataset)
    if (dataset / "sfm_scene.json").is_file():
        names = ("sfm_scene.json", "val_names.txt", "mesh_world.ply")
        for name in ("images", "pmasks"):
            if not (dataset / name).is_dir():
                raise FileNotFoundError("Missing calibrated DTU directory: " + str(dataset / name))
    else:
        names = ("mesh.obj", "transforms_train.json", "transforms_test.json")
    for name in names:
        if not (dataset / name).is_file():
            raise FileNotFoundError("Missing dataset input: " + str(dataset / name))


def infer_scene(dataset, explicit=None):
    """Infer a generic scene label, including the standard DTU inputs layout."""
    dataset = Path(dataset).expanduser().resolve()
    name = explicit or (dataset.parent.name if dataset.name == "inputs" else dataset.name)
    if not name or name in (".", "..") or Path(name).name != name:
        raise ValueError("The scene label must be a directory name without path separators")
    return name


def read_test_cameras(dataset):
    """Use the same native test split and camera conventions as training."""
    dataset = Path(dataset)
    if (dataset / "sfm_scene.json").is_file():
        from scene.dtu_dataset import readDTUCameras
        return readDTUCameras(dataset, False)[1]
    from scene.dataset_readers import readCamerasFromTransforms
    return readCamerasFromTransforms(str(dataset), "transforms_test.json", False)


def save_status(output, value):
    # Keep the historical filename readable by existing binding/editing tools.
    atomic_json(Path(output) / "qgain_full_status.json", value)
    atomic_json(Path(output) / "run_status.json", value)


def load_status(output):
    output = Path(output)
    for name in ("run_status.json", "qgain_full_status.json"):
        path = output / name
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError("No PMGS training status in " + str(output))


def checkpoint_paths(model_path, iteration=30000):
    """Accept a run directory, iteration directory, or point_cloud.ply."""
    path = Path(model_path).expanduser().resolve()
    if path.name == "point_cloud.ply":
        weights = path.parent
    elif (path / "point_cloud.ply").is_file():
        weights = path
    else:
        weights = path / "point_cloud" / ("iteration_" + str(iteration))
    for name in ("point_cloud.ply", "model_params.pt"):
        if not (weights / name).is_file():
            raise FileNotFoundError("Missing checkpoint input: " + str(weights / name))
    run = weights.parent.parent if weights.parent.name == "point_cloud" else weights.parent
    return run, weights


def set_memory_limit(fraction=None):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This PMGS stage requires a CUDA-enabled PyTorch environment")
    value = fraction if fraction is not None else os.environ.get("PMGS_MEMORY_FRACTION")
    if value is not None:
        value = float(value)
        if not 0 < value <= 1:
            raise ValueError("Memory fraction must lie in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(value, 0)

