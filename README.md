<div align="center">

# P-MGS

### Perception Over Geometry: Rethinking Gaussian Allocation in Mesh-based Splatting

Official implementation of the paper.

[Paper](#) | [Project Page](#) | [Video](https://www.youtube.com/watch?v=7Gqp5pUjiRo)


**English** · [简体中文](README_zh-CN.md)

</div>

<p align="center">
  <a href="#news">News</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#data-preparation">Data</a> ·
  <a href="#training">Training</a> ·
  <a href="#binding-and-evaluation">Binding &amp; Evaluation</a> ·
  <a href="#mesh-editing-lego-example">Mesh Editing</a> ·
  <a href="#acknowledgments">Acknowledgments</a>
</p>

---

## News


## Installation

Run the following commands from the project root. A CUDA toolkit and compatible C++ compiler are required to build the CUDA extensions.

```bash
conda env create -f environment.yml
conda activate pmgs

python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
conda install -c conda-forge ffmpeg

python scripts/check_environment.py --cuda
```

## Data Preparation

### NeRF Synthetic

Download [nerf_synthetic](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4). The scene directories are organized as follows:

```text
nerf_synthetic/
└── <scene>/
    ├── train/
    ├── val/
    ├── test/
    ├── transforms_train.json
    ├── transforms_val.json
    ├── transforms_test.json
    └── mesh.obj
```

If `mesh.obj` is not already available, download the corresponding Blender scene from [blend_files](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Open the `.blend` file in [Blender](https://www.blender.org/), select **File → Export → Wavefront (.obj)**, and save it as `mesh.obj` in the scene directory.

Use an absolute path for `NERF_ROOT` to link an existing dataset into PMGS:

```bash
NERF_ROOT="/path/to/nerf_synthetic"
mkdir -p data
ln -s "$NERF_ROOT" data/nerf_synthetic
```

### DTU

For real-world DTU scenes, we use the data preprocessed by [Relightable3DGaussian](https://github.com/NJU-3DV/Relightable3DGaussian), available [here](https://box.nju.edu.cn/f/d9858b670ab9480fb526/?dl=1). The training meshes are available on [Google Drive](https://drive.google.com/file/d/1nWOQcIqVn7CZNJEZikVDB_06Sg12Ww3k/view?usp=drive_link).

<details>
<summary><strong>Dataset directory structure and file reference</strong></summary>

The existing dataset collection is organized as follows:

```text
<dataset-root>/
├── dtu_idr/
│   └── scan<id>/
│       ├── image/
│       ├── mask/
│       └── cameras.npz
├── data_dtu/
│   └── DTU_scan<id>/inputs/
│       ├── images/
│       ├── pmasks/
│       ├── sfm_scene.json
│       └── model/
│           └── mesh_aligned.ply
└── data_dtu_valnames/
    └── dtu_scan<id>/
        └── val_names.txt
```

| Directory / file | Content |
|---|---|
| `dtu_idr/scan<id>/` | Original IDR images, masks, and cameras |
| `data_dtu/DTU_scan<id>/inputs/images/` | Training RGB images |
| `data_dtu/DTU_scan<id>/inputs/pmasks/` | Foreground masks |
| `data_dtu/DTU_scan<id>/inputs/sfm_scene.json` | Calibrated camera parameters |
| `data_dtu_valnames/dtu_scan<id>/val_names.txt` | Official held-out test image filenames |
| `data_dtu/DTU_scan<id>/inputs/model/mesh_aligned.ply` | Mesh aligned with the calibrated cameras |

</details>

PMGS reads the preprocessed inputs and the official test split. To prepare one scan, set `DTU_ROOT` to the absolute path of the dataset collection and replace `<id>` with the scan number:

```bash
DTU_ROOT="/path/to/Mani-GS-main/datasets"
SCAN_ID="<id>"
SOURCE="$DTU_ROOT/data_dtu/DTU_scan${SCAN_ID}/inputs"
TARGET="data/dtu/scan${SCAN_ID}"

mkdir -p "$TARGET"
ln -s "$SOURCE/images" "$TARGET/images"
ln -s "$SOURCE/pmasks" "$TARGET/pmasks"
ln -s "$SOURCE/sfm_scene.json" "$TARGET/sfm_scene.json"
ln -s "$SOURCE/model/mesh_aligned.ply" "$TARGET/mesh_world.ply"
ln -s "$DTU_ROOT/data_dtu_valnames/dtu_scan${SCAN_ID}/val_names.txt" "$TARGET/val_names.txt"
```

This creates the input layout expected by PMGS without moving the original data:

```text
PMGS/data/dtu/scan<id>/
├── images/
├── pmasks/
├── sfm_scene.json
├── mesh_world.ply
└── val_names.txt
```

## Training

Set the input and output directories. For NeRF Synthetic:

```bash
SCENE="<scene>"
DATA_PATH="data/nerf_synthetic/$SCENE"
MODEL_PATH="output/$SCENE"
GPU=0
```

For DTU, use the prepared scan directory instead:

```bash
SCENE="scan${SCAN_ID}"
DATA_PATH="data/dtu/$SCENE"
MODEL_PATH="output/$SCENE"
GPU=0
```

Train with the selected input:

```bash
python tools/reproduce_scene.py \
  --source-path "$DATA_PATH" --output "$MODEL_PATH" \
  --gpu "$GPU" --train-only
```

## Binding and Evaluation

Create the binding, render the test views, and evaluate the trained model:

```bash
BINDING_ROOT="bindings"

# Binding
python scripts/post_bind_affine.py \
  --model-path "$MODEL_PATH" --scene "$SCENE" \
  --output-root "$BINDING_ROOT" --bind-only

# Rendering
python tools/render.py \
  --model-path "$MODEL_PATH" --source-path "$DATA_PATH" \
  --destination "renders/$SCENE" --gpu "$GPU"

# Evaluation
python tools/evaluate.py \
  --model-path "$MODEL_PATH" --source-path "$DATA_PATH" \
  --destination "evaluation/$SCENE" --gpu "$GPU"
```

## Mesh Editing (Lego Example)

Use the trained Lego model and its binding to render the supplied bucket-lift motion and assemble the frames into video and GIF:

```bash
MODEL_PATH="output/lego"
BINDING_PATH="bindings/lego/binding_affine_normal_v1.pt"
ANIMATION_PATH="animations/lego"
GPU=0

python tools/animate.py \
  --model-path "$MODEL_PATH" --binding-path "$BINDING_PATH" \
  --output-root "$ANIMATION_PATH" --gpu "$GPU"

python tools/assemble_animation.py --output-root "$ANIMATION_PATH"
```

For a four-keyframe preview, add `--preview` to both commands. The example uses the bundled Lego rig and motion assets.

## Acknowledgments

Our code is built upon [gaussian-mesh-splatting](https://github.com/waczjoan/gaussian-mesh-splatting). We sincerely thank the authors for their efforts.
