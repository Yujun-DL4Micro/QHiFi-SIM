# GUI user guide

The GUI is a Qt/Python application launched from source. No executable is included.

## Start the application

Use the GUI environment described in [Installation](INSTALLATION.md), then run from the `QHiFi-SIM GUI/` directory:

```bash
python QHiFiGUI_main.py
```

## Main workflows

### Raw SIM input

Use this workflow when the input consists of raw SIM acquisitions. Set the optical parameters required by the GUI, select the raw acquisitions and output location, and configure a QHiFi checkpoint if the neural-network correction stage is needed. Confirm the expected raw frame organization in the application and user manual before processing a dataset.

### Existing SIM reconstructions

Use this workflow when SIM reconstructions already exist. Select the WF and SIM input folders and the checkpoint, and confirm that corresponding files are paired correctly. Keep the original input data unchanged.

### Quantitative analysis

The GUI has its own analysis implementation in `util/quant_analysis.py`. Its outputs and settings are separate from the command-line `analyze_quantitative.py` script under `QHiFi-SIM main/`. Avoid assuming that the two interfaces use identical defaults without checking them against the version of the GUI being released.

## Model checkpoint

Select a checkpoint matching the GUI's QHiFi model implementation. A checkpoint trained with an architectural ablation may not load into the standard model implementation. The checkpoint path and the selected model must be compatible.

## Troubleshooting

- **Checkpoint loading fails:** verify the path, checkpoint format, and compatibility between the saved architecture and GUI model.
- **Input file pairing is wrong:** confirm matching file counts, file ordering, and filenames in the WF and SIM folders.
- **CUDA/CuPy import fails:** use a compatible Python, PyTorch, CUDA, and CuPy combination as described by the dependency file; do not assume CPU fallback without testing.
- **Unexpected quantitative outputs:** preserve original inputs and inspect the analysis settings, intensity scaling, ROI definitions, and output directory before rerunning.

Refer to `QHiFi-SIM Software User Manual.docx` for interface-specific controls. That manual should be checked against the exact GUI revision included in a release.
