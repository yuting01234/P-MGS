# Edit animation

The public commands render an edit animation from a trained checkpoint and its
saved affine-normal binding, then assemble the frames into video and GIF:

```bash
RUN=/path/to/training_run
BINDING=/path/to/binding_affine_normal_v1.pt
OUT=animations/edit
GPU=0

python tools/animate.py \
  --model-path "$RUN" --binding-path "$BINDING" \
  --output-root "$OUT" --gpu "$GPU"

python tools/assemble_animation.py --output-root "$OUT"
```

These entry points forward to the bundled rig example. The checkpoint must be
aligned with that rig's reference geometry. They do not import arbitrary rigs
or automatically define an editing motion for another asset. The saved protocol
and motion assets determine the geometry correspondence, trajectory and camera.

## Checkpoint and binding

`--model-path` accepts a training run directory, a saved iteration directory or
its `point_cloud.ply`. The directory containing the PLY must also contain
`model_params.pt`. `--iteration` defaults to 30000 when a run directory is used.
The binding file must use `signed_normal_local_affine_v1` and match the selected
checkpoint's Gaussian centers, mesh vertices and face topology.

Generate a separate binding before animation if needed:

```bash
python scripts/post_bind_affine.py \
  --model-path "$RUN" --output-root bindings --bind-only
```

Pass the emitted `binding_affine_normal_v1.pt` as `--binding-path`. If that option
is omitted, the bundled example first searches
`bindings/lego/binding_affine_normal_v1.pt`, then beside the selected PLY. An
explicit path takes precedence.

The renderer calls `deform_gaussians()` with unconstrained barycentric
coordinates, signed normal offsets and local-affine covariance transport.
It checks the reconstructed reference-pose rendering against the original
checkpoint before generating animation frames. Colors, trained SH and opacity
stay unchanged; this example does not estimate new illumination.

## Preview and reuse

For the four original keyframes, add `--preview` to **both** commands:

```bash
python tools/animate.py \
  --model-path "$RUN" --binding-path "$BINDING" \
  --output-root "$OUT" --gpu "$GPU" --preview
python tools/assemble_animation.py --output-root "$OUT" --preview
```

The preview contains frames 0, 15, 29 and 59 and a contact sheet. A full render
contains 120 frames at 24 fps, 800×800 resolution, and a black background.
The fixed camera and original evaluated motion are recorded in the packaged
`examples/lego_bucket_lift/protocol.json`.

Completed output is reused only when the checkpoint and binding hashes,
protocol, motion assets and expected frames match. Choose a new `--output-root`
or add `--force` to rerender. `--dry-run` prints the render command without
opening checkpoints, creating output or using CUDA. For direct rendering, the
underlying `examples/lego_bucket_lift/render_rig.py` accepts the same checkpoint,
binding and output options and overwrites its frame filenames.

## Outputs and dependencies

```text
<output-root>/
  frames/frame_0000.png ... frame_0119.png
  rest.png
  control_rest.png
  metadata.json
  render.log
  pmgs_bucket_lift.mp4
  pmgs_bucket_lift.gif
  keyframes.jpg
  assembly_report.json
```

Rendering requires CUDA and the project rasterizer. Omitting `--gpu` preserves
`CUDA_VISIBLE_DEVICES`. Assembly needs Pillow plus `ffmpeg` and `ffprobe` on PATH.
The animation does not read training images or modify checkpoint/binding files.

## Motion assets and provenance

The external `assets/motion_assets.npz` contains the original valid reference-face
centroids, their control IDs and the original float32 per-frame transforms.
Repeated frames are losslessly deduplicated. `assets/manifest.json` records
source and archive hashes. The archive preserves the original nearest
reference-face-centroid association and contains no trained weights, image
dataset, rendered frames or Blender scene. All assets are ordinary files with
no external symbolic links. The archive is excluded from Git and must be placed
at `examples/lego_bucket_lift/assets/motion_assets.npz` before rendering.

Each bound face is split into independent corners in memory before applying the
rig. Ordered reference triangles and Gaussian-to-face assignments are preserved,
preventing mechanical part seams from stretching shared vertices. Only an
in-memory binding copy receives the split vertex indices; the saved reference
binding stays unchanged. The evaluated deformable components retain their
original triangle transforms.

Motion provenance and source attribution are recorded in the example's
`protocol.json` and `assets/manifest.json`. The example demonstrates geometric
editing with fixed learned appearance; it provides no dynamic ground-truth
quality score.
