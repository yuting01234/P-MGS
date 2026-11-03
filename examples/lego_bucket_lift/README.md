# Edit animation example

See [the animation guide](../../docs/animation.md) for compatible checkpoint and
binding setup, preview, output reuse and video/GIF assembly.

From the repository root:

```bash
python tools/animate.py \
  --model-path /path/to/training_run \
  --binding-path /path/to/binding_affine_normal_v1.pt \
  --output-root animations/edit --gpu 0
python tools/assemble_animation.py --output-root animations/edit
```

The entry points use the bundled rig protocol and motion assets. They require a
checkpoint aligned with that reference geometry; they do not automatically
create an editing motion or rig for arbitrary assets. Only the affine-normal
binding format is supported.
