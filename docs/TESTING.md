# Testing and inference

QHiFi-SIM provides two distinct test workflows in the `QHiFi-SIM main/` directory. Choose the workflow that matches the question being asked.

## 1. Patch-wise deep-learning test: `test_model.py`

This workflow is launched through `main.py` and uses the existing `AugmentedDataset(mode='test')` pipeline. Images are processed as training-style patches. It saves QWF and DWF patch outputs and calculates PSNR, SSIM, MSE, and NRMSE for each of them against the paired WF patch.

From `QHiFi-SIM main/`, configure the test list and checkpoint in `main.py`, then run:

```bash
python main.py --train_or_test test --seed 42
```

This workflow does not fit an intensity scale to full-size images, does not export QSIM/SIM comparisons, and does not perform ROI or pixel-level analysis. The comparison with WF is an input-consistency assessment, not an independent ground-truth reconstruction score.

## 2. Full-image inference: `test_full_image.py`

This script directly reads full-size WF and SIM TIFF files from separate folders and saves four outputs:

- `QWF/`: QHiFi's WF-domain output, rescaled using the maximum of the paired original WF image.
- `QSIM/`: QHiFi's SIM-domain output, rescaled using the maximum of the paired original WF image.
- `DWF/`: the SIM image convolved with the cropped PSF and scaled by the least-squares factor used in the previous combined workflow.
- `SIM/`: the original SIM input multiplied by the same scale factor used for DWF.

At the top of `test_full_image.py`, set `CHECKPOINT_PATH`, `WF_DIR`, `SIM_DIR`, `PSF_PATH`, `OUTPUT_DIR`, and `PSF_SIZE`. The default PSF crop size is 31 pixels, matching the previous script's default behavior. Paths may be absolute or relative to the script directory.

Run from the main source directory:

```bash
python test_full_image.py
```

The script pairs WF and SIM files by numeric sort order, matching the previous workflow. Confirm that the two folders contain corresponding files in that order. If names differ, the script emits a warning but preserves the positional pairing. Output files with matching names are overwritten on rerun, so back up results you need to keep.

## Checkpoint architecture

A standard QHiFi checkpoint should be loaded with all model ablation switches disabled. When evaluating an ablation checkpoint, change the settings in `MODEL_ABLATION` in `test_full_image.py` to match the training configuration.

## Output scale and interpretation

The full-image script restores the model outputs to an intensity scale using the paired WF image's maximum value and creates the scaled SIM/DWF references following the original workflow. These outputs are not guaranteed to retain the exact original integer bit depth; they are saved as float32 TIFFs to preserve numeric values.

Quantitative metrics and ROI outputs are generated separately by `analyze_quantitative.py`. See [Quantitative analysis](QUANTITATIVE_ANALYSIS.md).
