# Third-party notices

PMGS retains the copyright notices in imported source files. The original
[GaMeS / Gaussian Splatting license](LICENSE.md) is included without changes.
It describes research and non-commercial use; this repository does not replace
it with an MIT or Apache license.

| Component | Source / credit | Included terms |
|---|---|---|
| GaMeS framework | [gaussian-mesh-splatting](https://github.com/waczjoan/gaussian-mesh-splatting), GMUM | [LICENSE.md](LICENSE.md) and source headers |
| Gaussian Splatting framework | [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), Inria | [LICENSE.md](LICENSE.md) and source headers |
| CUDA rasterizer | [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) | [bundled license](submodules/diff-gaussian-rasterization/LICENSE.md) |
| CUDA nearest-neighbor initialization | simple-knn, Inria | [bundled license](submodules/simple-knn/LICENSE.md) |
| GLM headers | Bundled in the rasterizer's `third_party/glm` directory | Original copyright and permission notices retained in headers |
| LPIPS implementation / weights | Snapshot's `lpipsPyTorch`; [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) weights | Existing source retained; pretrained model terms remain applicable |

The two CUDA dependencies are vendored as source to preserve the local
implementation used by the original project. No extension binaries are shipped.
Optional FLAME face-model code and its model assets are outside this release's
scope and are omitted. No dataset or pretrained model license is granted by
this source package.
