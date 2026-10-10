# QHiFi-SIM

**Quantitative high-fidelity structured-illumination microscopy reconstruction and intensity correction**

qHiFi-SIM is a research software framework for quantitative structured illumination microscopy (SIM) reconstruction and intensity correction. Conventional SIM algorithms can produce high-fidelity super-resolution (SR) structures, but their reconstructed intensities are often distorted by background subtraction, OTF attenuation, spectrum optimization, or regularized deconvolution. qHiFi-SIM addresses this long-standing trade-off by using the experimental widefield (WF) image as a physical intensity reference and a conventional SIM reconstruction, such as HiFi-SIM, as a structural prior. A physics-driven self-supervised neural network then learns an intensity correction map without requiring ground-truth SR intensity images.

The framework preserves the structural fidelity and resolution gain of the input SIM reconstruction while restoring intensity linearity at the SIM point-spread-function (PSF) scale. It is compatible with widely used SIM algorithms such as HiFi-SIM, fairSIM, BF-SIM, FlexSIM, and selected denoising or deconvolution pipelines. qHiFi-SIM also supports live-cell dynamic intensity quantification, photobleaching correction, and quantitative SIM-FRET imaging.

This repository provides a Qt-based GUI, PyTorch training and patch-wise testing, full-image inference and intensity scaling, standalone quantitative analysis, lipid-droplet ROI quantification, and SIM simulation utilities.

If you use qHiFi-SIM, please cite the associated work:

Tang, Y. et al. *Physics-driven self-supervised learning for quantitative high-fidelity structured illumination microscopy*. bioRxiv (2026). https://doi.org/10.64898/2026.06.12.731778

## Repository structure

```text
QHiFi-SIM/
├── README.md
├── CITATION.cff
├── .gitignore
├── docs/
│   ├── INSTALLATION.md
│   ├── TRAINING.md
│   ├── TESTING.md
│   ├── QUANTITATIVE_ANALYSIS.md
│   ├── GUI_GUIDE.md
│   └── DATA_AND_CHECKPOINTS.md
├── Images_for_test/                 # Only small, authorized public examples
├── QHiFi-SIM GUI/
│   ├── QHiFiGUI_main.py
│   ├── requirements.txt
│   ├── util/
│   └── QHiFi-SIM Software User Manual.docx
└── QHiFi-SIM main/
    ├── main.py                      # Training / patch-wise test entry point
    ├── train.py
    ├── test_model.py                # Patch-wise deep-learning test
    ├── test_full_image.py           # Full-image inference and intensity scaling
    ├── analyze_quantitative.py      # Analysis of saved full-image results
    ├── augmented_dataset.py
    ├── models/
    ├── loss/
    ├── util/
    ├── lipid_droplet_quantification.py  # Lipid droplet ROI quantification for WF/SIM/QSIM images
    └── simulation_and_analysis/
```

Local datasets, trained checkpoints, generated results, caches, and IDE settings should not be committed by default. See [Data and checkpoints](docs/DATA_AND_CHECKPOINTS.md).

## Workflows

- **GUI reconstruction:** run the GUI from source and choose raw SIM reconstruction or correction of existing SIM images. See [GUI guide](docs/GUI_GUIDE.md).
- **Training:** train QHiFi from the WF/SIM/PSF triplets listed in the training and validation text files. See [Training guide](docs/TRAINING.md).
- **Patch-wise deep-learning test:** run `main.py --train_or_test test`. `test_model.py` compares QWF with WF and DWF with WF on the test patches. It does not run the full-image intensity-scaling workflow.
- **Full-image inference:** run `test_full_image.py`. It reads WF and SIM folders directly and saves QWF, QSIM, DWF, and intensity-scaled SIM images.
- **Standalone quantitative analysis:** run `analyze_quantitative.py` after full-image inference. It reads the saved output images and reproduces the ROI, R-squared, image-metric, and optional GT analyses. ROI window sizes can be configured near the top of that file.
- **SIM simulation and supporting analysis:** see the scripts in `QHiFi-SIM main/simulation_and_analysis/`.

See [Testing](docs/TESTING.md) and [Quantitative analysis](docs/QUANTITATIVE_ANALYSIS.md) for input layouts and output definitions.

## Paper configuration

The author-confirmed paper training settings are:

| Parameter | Value |
|---|---:|
| `weight` | `0.5` |
| `loss_corrected_l1` | `0.1` |
| `loss_corrected_ssim` | `0.1` |
| `seed` | `42` |

Ablation switches used in the paper and Supplementary Information are retained. Checkpoint architecture and ablation settings must match when loading a trained model.

## Citation

If you use QHiFi-SIM, please cite the associated work. Citation metadata is supplied in [`CITATION.cff`](CITATION.cff). The currently verified public record is the 2026 bioRxiv preprint; update the citation metadata if a journal version of record is the version you wish to cite.

Tang, Y. et al. *Physics-driven self-supervised learning for quantitative high-fidelity structured illumination microscopy*. bioRxiv (2026). https://doi.org/10.64898/2026.06.12.731778

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

Nothing in this repository grants a patent license.