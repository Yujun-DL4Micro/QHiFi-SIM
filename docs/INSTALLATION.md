# Installation

QHiFi-SIM contains two related but distinct Python applications: the training/analysis code and a Qt GUI. Installing them in separate environments is recommended because their dependency and accelerator requirements differ.

## 1. Get the repository

```bash
git clone https://github.com/Yujun-DL4Micro/QHiFi-SIM.git
cd QHiFi-SIM
```

Run Python commands from the relevant source directory unless a command says otherwise. The current repository layout contains spaces in the directory names, so quote those paths in shell commands when needed.

## 2. Training and analysis environment

Create and activate a Python environment using a Python version compatible with the PyTorch release you intend to use. Install a matching PyTorch build from the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) first, selecting the appropriate operating system, package manager, and CUDA option for the target machine.

Then install the supporting packages listed in `requirements-training.txt`:

```bash
python -m pip install -r requirements-training.txt
```

The dependency list is intentionally not version-pinned in this draft because the exact training environment used for the paper has not yet been exported and verified. Before making a tagged reproducibility release, record the actual Python, PyTorch, CUDA, NumPy, scikit-image, SciPy, and other relevant versions from the validated environment.

## 3. GUI environment

Change to the GUI source directory and install its existing requirements:

```bash
cd "QHiFi-SIM GUI"
python -m pip install -r requirements.txt
python QHiFiGUI_main.py
```

The GUI requirements currently pin specific versions of PyTorch, CuPy, NumPy, PyQt6, and other libraries. Use a compatible Python/CUDA setup and verify the environment on the target computer. Do not assume that a GPU-specific CuPy installation or CPU-only execution works without testing the actual GUI code paths.

## 4. Model checkpoint

The software requires a QHiFi checkpoint compatible with the selected model architecture. The checkpoint is not bundled automatically by this documentation package. Configure the checkpoint path in the relevant script or GUI. If loading an ablation checkpoint, use the same architectural ablation switches used to create it.

## 5. Quick checks

From `QHiFi-SIM main/`, run:

```bash
python -m py_compile main.py train.py test_model.py test_full_image.py analyze_quantitative.py
python main.py --help
```

These commands check Python syntax and entry-point imports; they do not prove that a complete training or inference run succeeds. Test with a small authorized dataset and a compatible checkpoint before processing a large dataset.
