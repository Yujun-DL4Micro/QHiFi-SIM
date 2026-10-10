# QHiFi-SIM GUI

This directory contains the Qt/Python graphical interface for QHiFi-SIM. The application is launched from source; a packaged executable is not provided.

## Start

Install the GUI dependencies with either pip or conda.

### Option A: pip

```bash
pip install -r requirements.txt
python QHiFiGUI_main.py
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate qifi-sim-gui
python QHiFiGUI_main.py
```

Use a compatible Python, PyTorch, CUDA, and CuPy combination. CPU fallback is supported but significantly slower. GPU execution is recommended.

## Documentation

See the repository-level [README](../README.md), [installation guide](../docs/INSTALLATION.md), and [GUI guide](../docs/GUI_GUIDE.md). The detailed interface walkthrough is provided in `QHiFi-SIM Software User Manual.docx`.

The GUI keeps its reconstruction and analysis implementation under `util/`. Its QHiFi model implementation is separate from the training-side model under `QHiFi-SIM main/models/`; ensure that the checkpoint architecture is compatible with the GUI model before distributing a checkpoint.

## Dependencies

`requirements.txt` and `environment.yml` contain the GUI dependency versions currently recorded by the project. Validate them in a clean environment before release, particularly the CUDA/CuPy combination.

## License

This GUI is part of QHiFi-SIM and is licensed under the MIT License.
See the repository-level [LICENSE](../LICENSE) for details.

Third-party components and embedded or derived HiFi-SIM code remain under their own licenses. Nothing in this repository grants a patent license.