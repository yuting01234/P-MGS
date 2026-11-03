<div align="center">

# P-MGS

### Perception Over Geometry: Rethinking Gaussian Allocation in Mesh-based Splatting

论文官方实现。

[Paper](#) | [Project Page](#) | [Video](https://www.youtube.com/watch?v=7Gqp5pUjiRo)


[English](README.md) · **简体中文**

</div>

<p align="center">
  <a href="#news">News</a> ·
  <a href="#installation">安装</a> ·
  <a href="#数据准备">数据准备</a> ·
  <a href="#training">训练</a> ·
  <a href="#binding-and-evaluation">绑定与评估</a> ·
  <a href="#编辑网格lego-为例">编辑网格</a> ·
  <a href="#致谢">致谢</a>
</p>

---

## News


## Installation

在项目根目录执行以下命令。编译 CUDA 扩展需要 CUDA toolkit 和兼容的 C++ 编译器。

```bash
conda env create -f environment.yml
conda activate pmgs

python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
conda install -c conda-forge ffmpeg

python scripts/check_environment.py --cuda
```

## 数据准备

### NeRF Synthetic

下载 [nerf_synthetic](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4) 数据集。各场景目录的组织格式如下：

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

如果目录中尚无 `mesh.obj`，从 [blend_files](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 下载对应的 Blender 文件，使用 [Blender](https://www.blender.org/) 打开 `.blend` 文件，通过 **文件 → 导出 → Wavefront (.obj)** 导出网格，并保存为对应场景目录下的 `mesh.obj`。

将 `NERF_ROOT` 设置为数据集的绝对路径，即可将已有数据链接到 PMGS：

```bash
NERF_ROOT="/path/to/nerf_synthetic"
mkdir -p data
ln -s "$NERF_ROOT" data/nerf_synthetic
```

### DTU

对于真实世界的 DTU 数据，我们采用 [Relightable3DGaussian](https://github.com/NJU-3DV/Relightable3DGaussian) 预处理的 DTU 数据，可在此[下载](https://box.nju.edu.cn/f/d9858b670ab9480fb526/?dl=1)。训练所用的网格可从 [Google Drive](https://drive.google.com/file/d/1nWOQcIqVn7CZNJEZikVDB_06Sg12Ww3k/view?usp=drive_link) 下载。

<details>
<summary><strong>查看数据目录结构与文件说明</strong></summary>

现有数据的组织格式如下：

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

| 目录或文件 | 内容 |
|---|---|
| `dtu_idr/scan<id>/` | 原始 IDR 图像、mask 和相机参数 |
| `data_dtu/DTU_scan<id>/inputs/images/` | 训练用 RGB 图像 |
| `data_dtu/DTU_scan<id>/inputs/pmasks/` | 前景 mask |
| `data_dtu/DTU_scan<id>/inputs/sfm_scene.json` | 标定相机参数 |
| `data_dtu_valnames/dtu_scan<id>/val_names.txt` | 官方测试图像文件名列表 |
| `data_dtu/DTU_scan<id>/inputs/model/mesh_aligned.ply` | 与标定相机对齐的训练网格 |

</details>

PMGS 读取预处理数据和官方测试划分。将 `DTU_ROOT` 设置为上述数据根目录的绝对路径，并将 `<id>` 替换为场景编号，即可整理单个场景：

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

整理后，PMGS 的输入目录如下；原数据无需搬动：

```text
PMGS/data/dtu/scan<id>/
├── images/
├── pmasks/
├── sfm_scene.json
├── mesh_world.ply
└── val_names.txt
```

## Training

设置数据与输出路径。NeRF Synthetic 使用：

```bash
SCENE="<scene>"
DATA_PATH="data/nerf_synthetic/$SCENE"
MODEL_PATH="output/$SCENE"
GPU=0
```

DTU 则使用整理后的场景目录：

```bash
SCENE="scan${SCAN_ID}"
DATA_PATH="data/dtu/$SCENE"
MODEL_PATH="output/$SCENE"
GPU=0
```

使用选定的数据进行训练：

```bash
python tools/reproduce_scene.py \
  --source-path "$DATA_PATH" --output "$MODEL_PATH" \
  --gpu "$GPU" --train-only
```

## Binding and Evaluation

生成绑定、渲染测试视角并评估训练结果：

```bash
BINDING_ROOT="bindings"

# 绑定
python scripts/post_bind_affine.py \
  --model-path "$MODEL_PATH" --scene "$SCENE" \
  --output-root "$BINDING_ROOT" --bind-only

# 渲染
python tools/render.py \
  --model-path "$MODEL_PATH" --source-path "$DATA_PATH" \
  --destination "renders/$SCENE" --gpu "$GPU"

# 评估
python tools/evaluate.py \
  --model-path "$MODEL_PATH" --source-path "$DATA_PATH" \
  --destination "evaluation/$SCENE" --gpu "$GPU"
```

## 编辑网格（Lego 为例）

使用训练好的 Lego 模型及其绑定文件，渲染随项目提供的铲斗抬升动作，再合成视频和 GIF：

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

如需四个关键帧预览，给两条命令都加上 `--preview`。此示例使用随项目提供的 Lego rig 和运动资产。

## 致谢

我们的代码基于 [gaussian-mesh-splatting](https://github.com/waczjoan/gaussian-mesh-splatting) 构建，衷心感谢他们的努力。
