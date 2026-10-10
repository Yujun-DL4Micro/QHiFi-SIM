# QHiFi-SIM Main

This directory contains the training, testing, full-image inference, and quantitative analysis code for **qHiFi-SIM**. It is the main computational entry point for the method described in the associated paper.

For an overview of the whole project and the GUI, see the repository root [`README.md`](../README.md).

## Directory structure

```text
QHiFi-SIM main/
├── main.py                         # Training / patch-wise test entry point
├── train.py                        # Training loop and validation
├── test_model.py                   # Patch-wise deep-learning test
├── test_full_image.py              # Full-image inference and intensity scaling
├── analyze_quantitative.py         # Analysis of saved full-image results
├── lipid_droplet_quantification.py # Lipid droplet ROI quantification
├── augmented_dataset.py            # Dataset class for training / testing
├── models/                         # Network architectures (e.g., QHiFi.py)
├── loss/                           # Loss functions (e.g., SSIM_loss.py)
├── util/                           # Utility functions (e.g., img_comp)
└── simulation_and_analysis/        # SIM simulation and supporting analysis scripts
```

## Requirements

Install the training and analysis dependencies with either pip or conda.

### Option A: pip

```bash
pip install -r ../requirements-training.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate qifi-sim-main
```

PyTorch must be installed separately according to your CUDA version if the default package does not match your GPU environment. See the root README for details.

## Data preparation

Training, patch-wise testing, and full-image inference use TIFF images and plain-text list files.

### Dataset layout

A typical dataset directory looks like:

```text
dataset_QHiFi/
└── MT/
    ├── train_list.txt
    ├── eval_list.txt
    ├── test_list.txt
    ├── WF/            # Widefield images
    ├── SIM/           # SIM reconstructions (or raw SIM data)
    └── PSF/           # Point-spread function images
```

Each list file contains one sample per line. Three columns are supported:

```text
WF_filename.tif SIM_filename.tif PSF_filename.tif
```

If only one column is provided, the same filename is used for all three modalities.

### Example list file

```text
cell01_wf.tif cell01_sim.tif psf.tif
cell02_wf.tif cell02_sim.tif psf.tif
```

## Training

Run training from this directory:

```bash
python main.py --train_or_test train \
    --datapath ../dataset_QHiFi/MT/ \
    --datalist ../dataset_QHiFi/MT/train_list.txt \
    --evallist ../dataset_QHiFi/MT/eval_list.txt \
    --nEpochs 150 \
    --batch_size 16 \
    --lr 1e-4 \
    --patch_size 256 \
    --psf_size 31 \
    --experiment_name my_experiment
```

Key arguments:

| Argument | Description |
|---|---|
| `--datapath` | Root directory of the dataset. |
| `--datalist` | Training list file. |
| `--evallist` | Validation list file. |
| `--nEpochs` | Number of training epochs. |
| `--batch_size` | Training batch size. |
| `--lr` | Initial learning rate. |
| `--patch_size` | Spatial size of training patches. |
| `--psf_size` | PSF kernel size (odd, ≤ 31). |
| `--experiment_name` | Name of the experiment; checkpoints and logs are saved under this name. |
| `--resume` | Path to a checkpoint to resume training. |

Paper training settings (already set as defaults in `main.py`):

| Parameter | Value |
|---|---:|
| `weight` | 0.5 |
| `loss_corrected_l1` | 0.1 |
| `loss_corrected_ssim` | 0.1 |
| `seed` | 42 |

Checkpoints are saved in `./checkpoint/<experiment_name>/`. Training logs and TensorBoard runs are saved under `runs/` and `train_*.log`.

## Patch-wise testing

Patch-wise testing evaluates the trained network on small patches and compares QWF with WF, and DWF with WF.

```bash
python main.py --train_or_test test \
    --datapath ../dataset_QHiFi/MT/ \
    --testlist ../dataset_QHiFi/MT/test_list.txt \
    --checkpoint_for_test ./checkpoint/my_experiment/best_checkpoint.pth.tar \
    --psf_size 31 \
    --results_path ./test_outs/my_experiment/
```

Outputs:

- `DL_test/DWF/` – PSF-blurred SIM patches.
- `DL_test/QWF/` – Network output patches.
- `DL_test/patch_metrics.csv` – Per-patch PSNR, SSIM, MSE, NRMSE.
- `DL_test/metrics_summary.txt` – Mean and standard deviation of metrics.

This test does **not** perform full-image intensity scaling. Use `test_full_image.py` for that.

## Full-image inference and intensity scaling

`test_full_image.py` processes full images and saves four outputs: `QWF`, `QSIM`, `DWF`, and intensity-scaled `SIM`.

Before running, open `test_full_image.py` and configure the paths in the `USER CONFIGURATION` section:

```python
CHECKPOINT_PATH = Path("./checkpoint/best_checkpoint.pth.tar")
WF_DIR = Path("./data/WF")
SIM_DIR = Path("./data/SIM")
PSF_PATH = Path("./data/PSF/psf.tif")
OUTPUT_DIR = Path("./full_image_results/PSF31")
PSF_SIZE = 31
DEVICE = "auto"
```

Then run:

```bash
python test_full_image.py
```

The script reads WF and SIM folders, pairs files by natural sort order, and writes results to `OUTPUT_DIR`. A manifest file `inference_manifest.csv` records the input files and scaling factors.

## Quantitative analysis

After full-image inference, run:

```bash
python analyze_quantitative.py
```

Configure the input and output paths at the top of `analyze_quantitative.py`:

```python
WF_DIR = Path("./data/WF")
SIM_INPUT_DIR = Path("./data/SIM")
RESULT_DIR = Path("./full_image_results/PSF31")
GT_DIR = None  # Set to a folder to enable GT-based analysis
ROI_SIZES = (6, 12)
NUM_ROIS = 500
```

Outputs are saved under `RESULT_DIR/Quant_Analysis/` and include:

- `R2_summary.xlsx` – Summary of R² and image metrics.
- `per_sample_metrics.csv` – Per-sample metrics.
- `ROI_sum_table/` – ROI sum tables.
- `difference_images/` – Absolute difference images.
- `ROI_marked_images/` – Images with ROI boxes.
- `analysis_summary.txt` – Human-readable summary.

## Lipid droplet quantification

`lipid_droplet_quantification.py` segments lipid droplets from WF images and measures mean intensities in WF, SIM, and QSIM images.

Example:

```bash
python lipid_droplet_quantification.py \
    --wf-folder /path/to/WF \
    --sim-folders /path/to/SIM_1 /path/to/SIM_2 \
    --qsim-folders /path/to/QSIM_1 /path/to/QSIM_2 \
    --out-root /path/to/output
```

The number of SIM and QSIM folders must match. Outputs include Excel tables, marked TIFF images, and linear regression plots.

## SIM simulation and supporting analysis

Scripts for SIM simulation and additional analyses are in `simulation_and_analysis/`. See the individual scripts for usage.

## Citation

If you use this code, please cite the associated work:

Tang, Y. et al. *Physics-driven self-supervised learning for quantitative high-fidelity structured illumination microscopy*. bioRxiv (2026). https://doi.org/10.64898/2026.06.12.731778

## License

This project is licensed under the MIT License. See the root [`LICENSE`](../LICENSE) for details.

Nothing in this repository grants a patent license.