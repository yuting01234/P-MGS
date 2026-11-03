"""Check PMGS imports; initialize CUDA only when --cuda is requested."""
import argparse
import importlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Also allocate a CUDA tensor")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    checks = {}
    modules = ["numpy", "scipy", "cv2", "trimesh", "open3d", "plyfile",
               "torch", "torchvision", "simple_knn._C", "diff_gaussian_rasterization",
               "games.mesh_splatting.scene.gaussian_mesh_model", "lpipsPyTorch"]
    for name in modules:
        try:
            module = importlib.import_module(name)
            checks[name] = {"status": "ok", "version": getattr(module, "__version__", None)}
        except Exception as exc:
            checks[name] = {"status": "failed", "error": str(exc)}
    if checks["torch"]["status"] == "ok":
        import torch
        checks["torch"]["build_cuda"] = torch.version.cuda
        if args.cuda:
            try:
                value = torch.ones(1, device="cuda")
                checks["cuda"] = {"status": "ok", "device": torch.cuda.get_device_name(value.device)}
            except Exception as exc:
                checks["cuda"] = {"status": "failed", "error": str(exc)}
    print(json.dumps({"python": sys.version, "checks": checks}, indent=2))
    return 1 if any(item["status"] == "failed" for item in checks.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
