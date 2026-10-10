# Standalone quantitative analysis

`analyze_quantitative.py` reads existing full-image inference results. It does **not** load a neural network, run inference, apply the DWF scale fit, or rewrite QWF/QSIM/SIM/DWF results.

## Configuration

Open `QHiFi-SIM main/analyze_quantitative.py` and edit the settings in the `USER CONFIGURATION` section:

```python
WF_DIR = Path("./data/WF")
SIM_INPUT_DIR = Path("./data/SIM")
RESULT_DIR = Path("./full_image_results/PSF25")
GT_DIR = None
ROI_SIZES = (6, 12)
NUM_ROIS = 500
ROI_BORDER = 30
BG_THRESHOLD = 0.05
MIN_FOREGROUND_RATIO = 0.8
RANDOM_SEED = None
```

`ROI_SIZES = (6, 12)` controls the two square ROI windows used for ROI-sum R-squared calculations and ROI tables. Change the values here if other window sizes are needed. `RANDOM_SEED = None` retains unseeded ROI sampling, as in the previous analysis code; set an integer if repeatable ROI coordinates are required. Note that changing the ROI seed changes the selected ROIs and may slightly change ROI-based metrics.

Set `GT_DIR` to a directory containing ground-truth/reference TIFFs to enable the optional SIM-vs-GT and QSIM-vs-GT analyses. Leave it as `None` when no GT data are available.

## Run

Run the full-image inference script first. Then, from `QHiFi-SIM main/`, run:

```bash
python analyze_quantitative.py
```

The WF and SIM input directories must contain corresponding images in the same numeric order used by `test_full_image.py`. The result directory must contain `QWF/`, `QSIM/`, `DWF/`, and `SIM/` subfolders with matching output file stems.

## Metrics and calculations

The analysis keeps the definitions used in the previous combined script:

- PSNR, SSIM, MSE, and NRMSE: `util.util.img_comp` is used, so metric definitions follow that shared implementation.
- Pixel-level R-squared: squared Pearson correlation between the two images.
- ROI-sum R-squared: squared Pearson correlation between sums measured in corresponding square ROIs.
- ROI intensity ratios: DWF/WF and QWF/WF; when GT is available, SIM/GT and QSIM/GT are also reported.
- Optional pixel-level values: paired image pixels excluding a 30-pixel border, written to Excel when GT is configured and `SAVE_PIXEL_VALUES` is enabled.

WF is used as the intensity reference for QWF/DWF comparisons. It is not an independent GT image. GT-based metrics are produced only when GT files are available and the GT file count matches the WF file count.

## Outputs

Results are written to `RESULT_DIR/Quant_Analysis/`:

```text
Quant_Analysis/
├── analysis_log.txt
├── analysis_summary.txt
├── per_sample_metrics.csv
├── R2_summary.xlsx
├── ROI_sum_tables/
├── difference_images/          # when SAVE_DIFFERENCE_IMAGES=True
├── ROI_marked_images/          # when SAVE_ROI_MARKED_IMAGES=True
└── Pixel_values/               # when GT is available and pixel export is enabled
    └── pixel_values.xlsx
```

ROI tables are written per sample and per ROI size. The pixel-level workbook can become large; disable `SAVE_PIXEL_VALUES` if it is not required for a particular analysis run.
