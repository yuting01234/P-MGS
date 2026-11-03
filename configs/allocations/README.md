# Qgain initialization plans

Each NPZ contains integer arrays `counts`, `lower`, and `upper`, with one
element per original mesh face. The `counts` sum is exactly three times the
original number of faces. Bounds encode the saved coverage and training-pixel
support constraints. `manifest.json` records checksums, data signatures, and
initial tensor fingerprints extracted from the completed new E4 runs.

`pmgs/allocation.py` verifies and replays these plans without the original
experiment directory or its dense feature caches. The original adaptive
sampler's within-face position, scale, rotation, and color rules remain in
use. The plans alone do not contain mesh geometry, training images, or learned
Gaussian weights.

Qgain construction used a `Q_f/n_f` allocation objective with coverage/support
bounds and separate training-view work prices. The later E4 trainer uses its
own scales and adaptive densification, so the allocation preflight's renderer
work bounds are not guarantees for the complete E4 training run.

The bundle supports exact replay for the original eight input scenes. It is
not a general Qgain recomputation tool for an arbitrary replacement mesh.
For the explicit original adaptive-sampler alternative, see
[`docs/training.md`](../../docs/training.md).
