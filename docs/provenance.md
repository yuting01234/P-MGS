# Source provenance

The training loop, Gaussian representation, initialization samplers and renderer
originate from the frozen source used for the completed E4 experiment. Original
copyright and license notices are retained. `source_manifest.json` records the
initial import hashes for retained files, before the portability changes.

The public binding path uses `affine_normal_binding.py`: unconstrained
barycentric coordinates, signed normal offsets and local-affine covariance
transport. Binding outputs are independent of training weights. The editing
example consumes this same format.

DTU data loading and calibrated camera projection were ported from the sibling
project's existing DTU adapter. NeRF Synthetic inputs retain their coordinate
convention. Runtime tools use configurable data paths, output directories, GPUs
and the active Python interpreter.

Allocation archives and the motion asset are local external files excluded
from Git. Their small metadata manifests remain in the source tree. Datasets,
meshes, trained weights, generated images/videos and compiled extensions are
external to this source release. The optional face-model and depth-estimation
components are outside its scope.

The reference result tables are existing evaluation records, rather than new
training results produced during packaging. Current validation coverage is
described in `validation.md`.
