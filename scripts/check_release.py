"""Check the portable source tree without importing PyTorch or accessing data."""
import argparse
import ast
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    errors, count, total = [], 0, 0
    generated = {"__pycache__", ".git", ".pytest_cache"}
    readme_media = {"assets/butterflyanimation.gif", "assets/leopard_animation_100.gif",
                    "assets/video.mp4", "assets/video_preview.gif"}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in generated for part in rel.parts):
            continue
        if path.is_symlink():
            errors.append(str(rel) + ": symlinks are not portable release assets")
            continue
        if not path.is_file():
            continue
        count += 1
        size = path.stat().st_size
        total += size
        if size > 50 * 1024 ** 2:
            errors.append(str(rel) + ": file exceeds 50 MiB")
        if (path.suffix in {".so", ".o", ".pyc", ".pt", ".pth", ".ckpt", ".ply", ".mp4", ".gif"}
                and rel.as_posix() not in readme_media):
            errors.append(str(rel) + ": generated or training artifact in source release")
        if path.suffix == ".py":
            try:
                text = path.read_text(encoding="utf-8")
                ast.parse(text, filename=str(rel))
                if "/" + "home/" in text or "/" + "Users/" in text:
                    errors.append(str(rel) + ": hard-coded user home path")
            except (SyntaxError, UnicodeError) as exc:
                errors.append(str(rel) + ": " + str(exc))
        elif path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as exc:
                errors.append(str(rel) + ": " + str(exc))
    required = ["README.md", "LICENSE.md", "requirements.txt", "environment.yml",
                "tools/reproduce_scene.py", "tools/evaluate.py", "tools/render.py",
                "scripts/post_bind_affine.py", "tools/animate.py", "tools/assemble_animation.py",
                "submodules/simple-knn/setup.py", "submodules/diff-gaussian-rasterization/setup.py"]
    for name in required:
        if not (root / name).is_file():
            errors.append(name + ": missing release entry point")
    print(json.dumps({"status": "passed" if not errors else "failed", "files": count,
                      "size_MiB": round(total / 1024 ** 2, 3), "errors": errors}, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
