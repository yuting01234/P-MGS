# Validation

Portable entry points can be inspected without training or loading a dataset:

```bash
python scripts/check_release.py
python scripts/check_environment.py
python tools/reproduce_scene.py --source-path data/example --dry-run
python tools/render.py --help
python tools/evaluate.py --help
python scripts/post_bind_affine.py --help
python tools/animate.py --help
python tools/assemble_animation.py --help
```

After separately providing allocation caches in `configs/allocations/`, run
`python tools/check_allocations.py` to validate them. The caches and animation
motion NPZ are excluded from Git.

Binding uses the independent affine-normal format. `--bind-only` creates,
saves, reloads and validates that format on CPU, including reference-position
and covariance reconstruction. Reusing an existing binding validates it
against the supplied checkpoint.

The animation asset retains the original reference-face inputs and motion
trajectory. CPU checks cover topology-preserving rig application and reference
reconstruction. Frame assembly has been checked with synthetic images.

The source package does not include training data, meshes or checkpoints.
Full training and GPU image rendering are outside the source packaging checks.
CUDA extensions were imported from the existing environment rather than
rebuilt in a fresh environment. Recorded reference metrics remain separate
from these packaging checks.
