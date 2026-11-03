# Training, rendering, and evaluation

The portable entry points use the active Python environment, repository-relative
paths, and explicit input/output arguments. None requires a fixed machine path,
GPU number, or scene name.

## Dataset layouts

For NeRF Synthetic or another scene in its transform format, the source directory
contains `transforms_train.json`, `transforms_test.json`, `mesh.obj`, and the
referenced images. The historical mesh transform in the original reader remains
in use for this format.

For calibrated DTU input, use the existing Relightable3DGaussian layout:

```text
<scene>/inputs/
  sfm_scene.json
  images/
  pmasks/
  val_names.txt
  mesh_world.ply
```

`val_names.txt` contains the held-out test image filenames, one per line. The
reader applies the foreground masks and uses the calibrated camera intrinsics,
including the principal point. `mesh_world.ply` must already be in the same
IDR-normalized world coordinates as the calibrated cameras. The existing DTU
reader expects the supplied bbox transform `diag(2,2,2,1)`; it does not perform
another coordinate conversion. Dataset download links are in the main README.

## Train

```bash
python tools/reproduce_scene.py \
  --source-path /path/to/source \
  --output /path/to/new/run \
  --train-only --gpu 0
```

The scene label is inferred from the source directory; when that directory is
named `inputs`, its parent name is used. `--scene` can override the label. The
output directory must be new, and must be separate from the input data.

Omit `--train-only` to run training, full rendering/evaluation, geometry
measurements, and the signed-normal/local-affine binding in sequence. Set
`--binding-output-root /path/to/bindings` for an explicit independent binding
location. Its generated file is
`<binding-output-root>/<scene>/binding_affine_normal_v1.pt`.

If a source directory is not specified, the shorthand `--scene LABEL
--data-root /path/to/dataset/root` is available. `PMGS_DATA_ROOT` sets the default
root; otherwise it is `data/nerf_synthetic` inside the repository. A source path
is recommended when mixing dataset formats.

`--dry-run` prints all commands and low-level trainer flags without creating
files or initializing CUDA. `--gpu` selects a visible CUDA device; omitting it
respects `CUDA_VISIBLE_DEVICES`. Optional controls include `--memory-fraction
0.9`, `--threads 4`, and `--port 6010` for concurrent viewer ports. Subprocesses
use the active `sys.executable`.

## Initialization and retained training parameters

`--allocation auto` is the default. Before initialization, the trainer hashes
mesh vertices/faces, camera parameters, and training RGB images. A complete match
to a saved allocation signature selects that exact Qgain per-face allocation;
otherwise it selects the original complexity-weighted adaptive sampler. The
choice and reason are printed and saved in `run_status.json`. Renaming a scene
or moving its directory does not prevent an otherwise exact signature match.

`--allocation qgain` requires a saved signature match and stops on mismatch.
`--allocation adaptive` explicitly selects the adaptive sampler. The repository
contains signatures for the original eight Qgain plans; their NPZ caches are
external assets and are not tracked by Git. Put the corresponding cache in
`configs/allocations/` to replay a matched plan. A missing matched cache stops
the run rather than silently changing the initialization method. This is not
a general Qgain recomputation pipeline. The integer count arrays do not include the corresponding meshes,
images, or trained Gaussian parameters. For a new mesh or DTU input, auto mode
normally selects adaptive initialization.

For a matched plan, the exact face counts and five initial tensor fingerprints
are checked. Floating-point changes across dependency versions can cause a
fingerprint mismatch. That stops the run by default; explicitly use
`--allow-initialization-drift` only when that change is intended. The flag is
recorded and never bypasses signature or per-face count checks.

All trainer arguments are in `configs/e4.json`. The retained configuration uses
30,000 iterations, SH degree 3, an initial budget of `3F`, black backgrounds, and
DSSIM weight 0.3. Blur and progressive frequency weights are both zero. The
soft mesh regularizers are:

| Term | Weight | Starts at iteration |
| --- | --- | --- |
| Distance to assigned face plane | 0.01 | 10,000 |
| Normal consistency | 0.005 | 5,000 |
| Minimum/maximum scale regularization | 0.005 each | 1,000 |

Distance is clamped at 0.01. Densification uses interval 100, gradient threshold
0.0005, and iterations 1,000 through 25,000. Gaussian positions, rotations, and
scales are optimized freely; mesh regularizers are soft penalties. Editing uses
the post-training signed-normal/local-affine binding.

Training saves `point_cloud/iteration_30000/point_cloud.ply`, `model_params.pt`,
`cfg_args`, `train.log`, loss telemetry, `REPLAY_PROTOCOL.json`, and run status.
Generated `dataset_private` links isolate the readers' `points3d.ply` writes from
the original input directory. These runtime links and outputs are not part of
the source release.

## Render without metrics

```bash
python tools/render.py \
  --model-path /path/to/trained/run \
  --source-path /path/to/source \
  --destination /path/to/render/output --gpu 0
```

This saves images under `test/ours_30000/renders` and a `render_status.json`.
It does not load LPIPS or compute image quality metrics. `tools/evaluate.py
--render-only` runs the same branch without a second model initialization.
Both transform-format scenes and calibrated DTU test cameras are supported.

## Evaluate

```bash
python tools/evaluate.py \
  --model-path /path/to/trained/run \
  --source-path /path/to/source \
  --destination /path/to/evaluation/output --gpu 0
```

Evaluation renders the full test split, saves RGB and ground truth PNGs,
`per_view.csv`, and `evaluation.json`. PSNR and SSIM are computed from float RGB
before PNG quantization; VGG LPIPS v0.1 receives RGB in `[-1,1]`. The first LPIPS
use may download its pretrained weights. To match the original project, PSNR
averages per-channel R/G/B PSNR, then averages views; it does not take a single
logarithm of the joint RGB MSE. The definition is recorded in the report.

Both tools accept a run directory, an iteration directory, or `point_cloud.ply`
as `--model-path`. `--iteration` defaults to 30000. `--source-path` can be omitted
when the saved run status still refers to valid input data. Existing result
files are rejected; use a fresh destination or explicitly `--overwrite`.
`--max-views 3` provides a short check and marks the report partial when fewer
than all test views are used.

Geometry diagnostics and allocation integrity checks remain available:

```bash
python tools/geometry_diagnostics.py \
  --model-path /path/to/trained/run --destination /path/to/geometry.json
python tools/check_allocations.py --scenes all
```

Geometry measurements do not establish deformation quality. New binding and
editing commands are documented in the main README and the binding/editor docs.
