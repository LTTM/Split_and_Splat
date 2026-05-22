# Split&Splat: Zero-Shot Instance Segmentation via Explicit Object Modeling and 3D Gaussian Splatting

| [Leonardo Monchieri](https://leonardomonchieri.github.io/) | [Elena Camuffo](https://elenacamuffo.me/) | [Francesco Barbato](https://medialab.dei.unipd.it/members/francesco-barbato/) | [Pietro Zanuttigh](https://medialab.dei.unipd.it/members/pietro-zanuttigh/) | [Simone Milani](https://medialab.dei.unipd.it/members/simone-milani/) | 

[![License: GPL3](https://img.shields.io/badge/License-GPL3-yellow.svg)](https://opensource.org/license/gpl-3-0)

![Split & Splat Overview](./split_&_splat_GA.png)

## About
3D Gaussian Splatting (GS) enables fast and high-quality scene reconstruction, but it lacks an object-consistent and semantically aware structure. We propose **Split&Splat**, a framework for instance scene reconstruction using 3DGS that explicitly models object instances.

The pipeline works in four stages:
1. **Split**(Segment): instance masks are propagated across views using depth information, producing view-consistent 2D masks.
2. **Splat**(Reconstruct): each object is reconstructed independently as a separate Gaussian model, then merged back into the scene with refined boundaries.
3. **Compose**: per-instance Gaussians are progressively merged into a full scene model using a composition pipeline with increasing mask loss weights.
4. **Evaluate**: instance-level semantic descriptors are embedded into the reconstructed objects and evaluated against ground-truth annotations.

Unlike existing methods, **Split&Splat** segments the scene first and reconstructs each object individually. This design naturally supports downstream tasks and allows Split&Splat to achieve state-of-the-art performance on the ScanNetv2 segmentation benchmark.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/LTTM/Split_and_Splat.git
cd Split_and_Splat
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate split_and_splat
```

### 3. Install SAM 2

**Split&Splat relies on SAM 2 (Segment Anything Model 2) by Meta. You must download and install it separately.**

Clone the official SAM 2 repository into the `sam2_repo/` folder (or replace the existing placeholder):

```bash
git clone https://github.com/facebookresearch/sam2.git sam2_repo
```

Then install it:

```bash
cd sam2_repo
pip install -e .
cd ..
```

Download the SAM 2 model checkpoints and place them in the `checkpoints/` directory:

```bash
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cd ..
```

> Available checkpoints: `sam2.1_hiera_tiny.pt`, `sam2.1_hiera_small.pt`, `sam2.1_hiera_base_plus.pt`, `sam2.1_hiera_large.pt`. The large model is recommended for best segmentation quality.

### 4. Install Gaussian Splatting CUDA extensions

```bash
cd submodules/diff-gaussian-rasterization
pip install -e . --no-build-isolation
cd ../simple-knn
pip install -e . --no-build-isolation
cd ../fused-ssim
pip install -e . --no-build-isolation
cd ../..
```

> The `--no-build-isolation` flag is required so the build process can access the already-installed PyTorch headers.

### 5. Build the CUDA point projection extension

```bash
cd point_projection
pip install -e . --no-build-isolation
cd ..
```

### 6. Set the PyTorch library path

Add the PyTorch shared libraries to `LD_LIBRARY_PATH` so CUDA extension `.so` files can load at runtime:

```bash
export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
```

To make this permanent across Conda sessions, add it to your environment's activation hook:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), '"'"'lib'"'"'))"):$LD_LIBRARY_PATH' \
    > $CONDA_PREFIX/etc/conda/activate.d/torch_libs.sh
```

---

## Datasets

We evaluated Split&Splat on the [ScanNetv2](http://www.scan-net.org/) and [LERF](https://www.lerf.io/) datasets. For ScanNet, we used scenes: _scene0000_, _scene0062_, _scene0070_, _scene0097_, _scene0140_, _scene0200_, _scene0347_, _scene0400_, _scene0590_, and _scene0645_ (following the selection by Yanmin Wu et al. in [OpenGaussian](https://3d-aigc.github.io/OpenGaussian/)).

### Dataset Preparation

For manual preparation:

1. **Download and extract** the raw ScanNet scene into `data/<scene_name>/`.
2. **Run COLMAP** to generate camera poses and a sparse point cloud:
   ```bash
   colmap automatic_reconstructor \
       --image_path data/<scene_name>/images \
       --sparse_model_path data/<scene_name>/sparse
   ```
3. **Convert COLMAP output** to the required format:
   ```bash
   python convert.py --dataset_path data/<scene_name>
   ```

### Data Directory Structure

Before running the reconstruction preparation, each scene should have the following layout:

```
data/
└── <scene_name>/
    ├── depth/                        # Raw depth frames (aligned with color)
    │   └── <NNNN>.png
    ├── pose/                         # Per-frame camera-to-world 4×4 matrices
    │   └── <NNNN>.txt
    ├── intrinsic/                    # Camera calibration
    │   ├── intrinsic_color.txt
    │   ├── intrinsic_depth.txt
    │   ├── extrinsic_color.txt
    │   └── extrinsic_depth.txt
    ├── images/                       # Subsampled frames used for training
    │   └── <NNNN>.JPEG
    ├── transforms_train.json         # NeRF-style camera transforms
    ├── point_cloud.ply               # Scene point cloud
    │
    ├── masks/                        # Per-instance training folders (pipeline-generated)
    │   ├── <id>/                     # Single instance folder
    │   │   ├── images/               # Training frames for this instance
    │   │   ├── masks/                # Binary mask per frame
    │   │   ├── points3d.ply
    │   │   └── transforms_train.json
    │   ├── <id1>_<id2>/              # Composition: two instances merged
    │   │   └── ...
    │   └── <id1>_<id2>_..._<idN>/   # Composition: all instances merged
    │       └── ...
    │
    ├── discard/                      # Instances pruned during refinement
    │   └── <id>/
    │       └── ...                   # Same structure as masks/<id>/
    │
    ├── test/                         # Evaluation inputs
    │   ├── pred.ply
    │   ├── <scene_name>_vh_clean_2.ply
    │   ├── <scene_name>_vh_clean_2.labels.ply
    │   ├── <scene_name>_vh_clean_2.*.segs.json
    │   ├── <scene_name>_vh_clean.aggregation.json
    │   └── transforms_train.json
    │
    ├── <scene_name>.sens             # Raw ScanNet sensor stream
    ├── <scene_name>.txt              # Scene metadata
    ├── <scene_name>.aggregation.json
    ├── <scene_name>_vh_clean_2.ply
    ├── <scene_name>_vh_clean_2.labels.ply
    ├── <scene_name>_vh_clean_2.*.segs.json
    ├── <scene_name>_vh_clean.aggregation.json
    ├── <scene_name>_vh_clean.segs.json
    ├── <scene_name>_2d-instance.zip
    ├── <scene_name>_2d-instance-filt.zip
    ├── <scene_name>_2d-label.zip
    └── <scene_name>_2d-label-filt.zip
```

> **Note on `masks/`:** this folder is created and populated by the pipeline. Single-instance folders (`<id>/`) are produced in Stage 1. Composition folders (`<id1>_<id2>_..._<idN>/`) are produced incrementally in Stage 3 as instances are merged one by one.

---

## Pipeline

The full pipeline consists of four stages: mask generation, instance reconstruction, composition, and evaluation. All steps use `scene0347_00` as the example scene — replace it with your target scene name.

### Stage 1 — Mask Generation and Propagation

```bash
# 1. Generate automatic segmentation masks with SAM 2
python ./sam2/auto_seg.py --scene scene0347_00

# 2. Propagate masks across frames using depth information
python ./sam2/mask_propagation_scanet.py --scene scene0347_00 --verbose

# 3. Move the generated masks to the data directory
mv ./output/scene0347_00_masks ./data/scene0347_00/masks

# 4. Prepare the per-instance folder structure for training
./bash_dir_utils/prepare_folder.sh scene0347_00
```

### Stage 2 — Instance Reconstruction

```bash
# 5. Initial per-instance training
./run_all.sh scene0347_00

# 6. First refinement pass
./run_ref.sh scene0347_00

# 7. Move extracted PLY files
./bash_dir_utils/move_extra.sh ./data/scene0347_00

# 8. Refinement training pass
./run_all_ref.sh scene0347_00
```

### Stage 3 — Composition

```bash
# 9. Set mask_loss weight to 0.05 in train.py for the first composition pass
sed -i 's/Ll1_mask \* [0-9.]*/Ll1_mask * 0.05/' train.py

# 10–11. Copy refined per-instance PLY files into two staging folders:
#   PLY_ref  — reference copy preserved throughout the composition loop
#   tmp      — working copy consumed and updated each iteration
# (--init reads from output/<scene>/ref, i.e. the refined per-instance training output)
python move_PLY.py --scene scene0347_00 --output=PLY_ref --init
python move_PLY.py --scene scene0347_00 --output=tmp --init

# 12. Copy images and camera data into the masks folder
cp ./data/scene0347_00/images/* ./data/scene0347_00/masks/
cp ./data/scene0347_00/transforms_train.json ./data/scene0347_00/masks/

# 13. Combine masks from different segmentation methods
#     This creates ./data/scene0347_00/masks/combined/ with merged per-instance mask folders
python ./utils_mask/mask_combination.py --scene scene0347_00

# 14. Train on the combined masks
./combo.sh ./data/scene0347_00/masks/combined ./output/scene0347_00/comb

# 15. Run the automated composition pipeline (mask_loss = 0.05)
./run_composition_pipeline.sh scene0347_00

# 16. Increase mask_loss to 0.1, then run again
sed -i 's/Ll1_mask \* [0-9.]*/Ll1_mask * 0.1/' train.py
./run_composition_pipeline.sh scene0347_00

# 17. Increase mask_loss to 0.25 for the final refinement pass
sed -i 's/Ll1_mask \* [0-9.]*/Ll1_mask * 0.25/' train.py
./run_composition_pipeline.sh scene0347_00
```

### Stage 4 — Evaluation

```bash
# 18. Move the final PLY to the data directory
mv ./output/scene0347_00/final.ply ./data/scene0347_00/pred.ply

# 19. Run instance clustering and semantic analysis
python ./evaluation/instance_cluster.py --scene scene0347_00 --verbose
```

**Compute image quality metrics** (PSNR, SSIM, LPIPS):

```bash
python metrics.py --output_path output/scene0347_00
```

---

## Output Structure

```
output/<scene_name>/
├── raw/                    # Per-instance training output
│   └── <instance_id>/
│       ├── point_cloud/
│       ├── iterations/
│       └── renders/
├── comb/                   # Combined segmentation training output
│   └── tmp/                # Previous recontruction
└── PLY_ref/                # Refined per instance PLY training output
```

---

## Project Structure

| Directory | Description |
|-----------|-------------|
| `sam2/` | Project scripts for mask generation and propagation (auto_seg, mask_propagation) |
| `sam2_repo/` | Cloned SAM 2 library from Meta (installed as a dependency) |
| `point_projection/` | CUDA-accelerated 2D/3D point projection module |
| `gaussian_renderer/` | 3D Gaussian splatting renderer |
| `scene/` | Scene and camera data management |
| `utils_mask/` | Mask processing and combination utilities |
| `evaluation/` | Metrics, instance clustering, and semantic analysis |
| `submodules/` | Gaussian rasterization and optimization libraries |
| `arguments/` | Command-line parameter definitions |
| `bash_dir_utils/` | Shell scripts for data preparation |
| `checkpoints/` | SAM 2 model checkpoints |
| `data/` | Input scenes and processed datasets |
| `output/` | Training outputs and rendered results |

---

## Citation

If you use Split & Splat in your research, please cite:

```bibtex
@misc{monchieri2026splitsplatzeroshotpanopticsegmentation,
      title={Split&Splat: Zero-Shot Panoptic Segmentation via Explicit Instance Modeling and 3D Gaussian Splatting}, 
      author={Leonardo Monchieri and Elena Camuffo and Francesco Barbato and Pietro Zanuttigh and Simone Milani},
      year={2026},
      eprint={2602.03809},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2602.03809}, 
}
```

---

## License

This project is licensed under the GPL-3.0 License. See [LICENSE.md](LICENSE.md) for details.

---

## Acknowledgments

This work builds upon:
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Original 3DGS framework by the GRAPHDECO group at Inria
- [SAM 2](https://github.com/facebookresearch/segment-anything-2) — Segment Anything Model 2 by Meta
- [COLMAP](https://colmap.github.io/) — Structure from Motion and Multi-View Stereo

---

## Contact

For questions and inquiries, please reach out to:
- [Leonardo Monchieri](https://leonardomonchieri.github.io/)
- [Simone Milani](https://medialab.dei.unipd.it/members/simone-milani/)
