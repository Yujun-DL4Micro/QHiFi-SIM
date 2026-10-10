"""Full-image QHiFi-SIM inference and intensity scaling.

This script replaces the inference portion of the previous combined full-image script.
It processes full images (not training-style patches) and saves four outputs:
QWF, QSIM, DWF, and intensity-scaled SIM. Quantitative metrics, ROI analysis,
and Excel reporting are handled separately by ``analyze_quantitative.py``.

Configure the paths and PSF size in the USER CONFIGURATION section before use.
Run this file from the project environment, for example:
    python test_full_image.py
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from scipy.signal import convolve2d

from models.QHiFi import QHiFi


# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Paths are relative to this script's directory unless you provide absolute paths.
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = Path("./checkpoint/best_checkpoint.pth.tar")
WF_DIR = Path("./data/WF")
SIM_DIR = Path("./data/SIM")
PSF_PATH = Path("./data/PSF/psf.tif")
OUTPUT_DIR = Path("./full_image_results/PSF25")

# A centered crop of this size is used from the supplied square PSF.
PSF_SIZE = 31

# Use "auto", "cuda", or "cpu".
DEVICE = "auto"

# Set these to the same values used when training an ablation checkpoint.
# For the standard full QHiFi model, leave all values False.
MODEL_ABLATION = {
    "disable_ssfe": False,
    "disable_wife": False,
    "disable_channel_attn": False,
    "disable_spatial_attn": False,
    "disable_identity": False,
}

# Existing output files with the same names are overwritten, matching the
# original full-image workflow. Back up important results before rerunning.
# =============================================================================


OUTPUT_SUBDIRS = ("QWF", "QSIM", "DWF", "SIM")


def resolve_path(path: Path) -> Path:
    """Resolve relative configuration paths against this script directory."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("QHiFiSIMFullImageTest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(output_dir / "full_image_test.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def natural_sort_key(path: Path) -> tuple[int, str]:
    """Sort image names by the first integer in the file name, as in the old script."""
    import re

    matches = re.findall(r"\d+", path.stem)
    number = int(matches[0]) if matches else -1
    return number, path.name.lower()


def list_tiffs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
    return sorted(files, key=natural_sort_key)


def read_single_channel_tiff(path: Path) -> np.ndarray:
    image = np.asarray(tifffile.imread(path), dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a single-channel 2D TIFF: {path} (shape={image.shape})")
    if not np.isfinite(image).all():
        raise ValueError(f"Image contains NaN or infinite values: {path}")
    return image


def normalize_by_max(image: np.ndarray) -> np.ndarray:
    """Use the same maximum-value normalization as the previous full-image implementation."""
    maximum = float(np.max(image))
    return image / (maximum + 1e-8)


def load_psf(path: Path, size: int) -> np.ndarray:
    psf = read_single_channel_tiff(path)
    if psf.shape[0] != psf.shape[1]:
        raise ValueError(f"PSF must be square, got {psf.shape}: {path}")
    if size <= 0 or size % 2 == 0:
        raise ValueError(f"PSF_SIZE must be a positive odd integer, got {size}")
    if size > psf.shape[0]:
        raise ValueError(f"PSF_SIZE={size} is larger than the PSF image size {psf.shape[0]}")
    start = (psf.shape[0] - size) // 2
    psf = psf[start:start + size, start:start + size].astype(np.float32, copy=False)
    total = float(psf.sum())
    if not np.isfinite(total) or abs(total) < 1e-12:
        raise ValueError(f"PSF sum is zero or invalid: {path}")
    return psf / total


def choose_device(device_name: str) -> torch.device:
    name = device_name.strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE is set to 'cuda', but CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("DEVICE must be 'auto', 'cuda', or 'cpu'.")


def torch_load_checkpoint(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with PyTorch versions without weights_only.
        return torch.load(path, map_location=device)


def load_model(path: Path, device: torch.device) -> QHiFi:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    checkpoint = torch_load_checkpoint(path, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format: expected a state dict or a checkpoint containing 'model_state_dict'.")

    model = QHiFi(**MODEL_ABLATION).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_float_tiff(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D output image, got shape {image.shape}: {path}")
    if not np.isfinite(image).all():
        raise ValueError(f"Model output contains NaN or infinite values: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, image)


def main() -> None:
    checkpoint_path = resolve_path(CHECKPOINT_PATH)
    wf_dir = resolve_path(WF_DIR)
    sim_dir = resolve_path(SIM_DIR)
    psf_path = resolve_path(PSF_PATH)
    output_dir = resolve_path(OUTPUT_DIR)

    logger = setup_logger(output_dir)
    device = choose_device(DEVICE)
    logger.info("Device: %s", device)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("WF directory: %s", wf_dir)
    logger.info("SIM directory: %s", sim_dir)
    logger.info("PSF: %s (crop size=%d)", psf_path, PSF_SIZE)

    wf_files = list_tiffs(wf_dir)
    sim_files = list_tiffs(sim_dir)
    if not wf_files:
        raise RuntimeError(f"No TIFF images found in WF directory: {wf_dir}")
    if len(wf_files) != len(sim_files):
        raise ValueError(f"File count mismatch: {len(wf_files)} WF files vs {len(sim_files)} SIM files")

    # Preserve the original script's numeric sorting/pair-by-position behavior.
    # A warning is emitted when corresponding file stems differ.
    for index, (wf_path, sim_path) in enumerate(zip(wf_files, sim_files)):
        if wf_path.stem != sim_path.stem:
            logger.warning(
                "Pair %d uses different file stems: WF=%s, SIM=%s. Confirm that numeric sorting gives the intended pairing.",
                index, wf_path.name, sim_path.name,
            )

    psf = load_psf(psf_path, PSF_SIZE)
    psf_tensor = torch.from_numpy(psf).unsqueeze(0).unsqueeze(0).to(device)
    model = load_model(checkpoint_path, device)

    output_dirs = {name: output_dir / name for name in OUTPUT_SUBDIRS}
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "inference_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "index", "wf_input", "sim_input", "output_stem", "wf_max_scale",
            "dwf_scale_factor", "psf_size", "checkpoint",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for index, (wf_path, sim_path) in enumerate(zip(wf_files, sim_files)):
            logger.info("Processing %d/%d: WF=%s | SIM=%s", index + 1, len(wf_files), wf_path.name, sim_path.name)
            wf_original = read_single_channel_tiff(wf_path)
            sim_raw = read_single_channel_tiff(sim_path)
            if wf_original.shape != sim_raw.shape:
                raise ValueError(
                    f"WF/SIM shape mismatch for pair {wf_path.name} / {sim_path.name}: "
                    f"{wf_original.shape} vs {sim_raw.shape}"
                )

            wf_normalized = normalize_by_max(wf_original)
            sim_normalized = normalize_by_max(sim_raw)
            wf_tensor = torch.from_numpy(wf_normalized).unsqueeze(0).unsqueeze(0).to(device)
            sim_tensor = torch.from_numpy(sim_normalized).unsqueeze(0).unsqueeze(0).to(device)

            with torch.inference_mode():
                qsim_tensor, qwf_tensor, _ = model(sim_tensor, wf_tensor, psf_tensor)

            wf_max_scale = float(np.percentile(wf_original, 100))
            qsim = qsim_tensor.detach().float().cpu().numpy()[0, 0] * wf_max_scale
            qwf = qwf_tensor.detach().float().cpu().numpy()[0, 0] * wf_max_scale

            # Preserve the previous full-image implementation baseline calculation and scaling.
            dwf_raw = convolve2d(sim_raw, psf, mode="same", boundary="wrap")
            denominator = float(np.dot(dwf_raw.ravel(), dwf_raw.ravel()))
            numerator = float(np.dot(dwf_raw.ravel(), wf_original.ravel()))
            c_opt = numerator / (denominator + 1e-12)
            if c_opt <= 0:
                logger.warning("Sample %s: fitted DWF scale <= 0 (%.6g); using 1e-3 as in the original workflow.", wf_path.stem, c_opt)
                c_opt = 1e-3

            dwf = dwf_raw * c_opt
            sim_scaled = sim_raw * c_opt
            output_stem = wf_path.stem

            save_float_tiff(output_dirs["QWF"] / f"{output_stem}.tif", qwf)
            save_float_tiff(output_dirs["QSIM"] / f"{output_stem}.tif", qsim)
            save_float_tiff(output_dirs["DWF"] / f"{output_stem}.tif", dwf)
            save_float_tiff(output_dirs["SIM"] / f"{output_stem}.tif", sim_scaled)

            writer.writerow({
                "index": index,
                "wf_input": str(wf_path),
                "sim_input": str(sim_path),
                "output_stem": output_stem,
                "wf_max_scale": wf_max_scale,
                "dwf_scale_factor": c_opt,
                "psf_size": PSF_SIZE,
                "checkpoint": str(checkpoint_path),
            })

    logger.info("Finished full-image inference. Outputs: %s", output_dir)
    logger.info("Quantitative metrics and ROI analyses are handled by analyze_quantitative.py.")


if __name__ == "__main__":
    main()
