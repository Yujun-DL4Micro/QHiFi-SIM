"""Patch-wise deep-learning evaluation for QHiFi-SIM.

This module is called by ``main.py`` through ``test(args, test_loader, device)``.
It evaluates the trained QHiFi model on the patch items prepared by
``AugmentedDataset(mode='test')`` and saves the network outputs directly.

Scope
-----
- Loads WF, SIM, and PSF samples from the existing test DataLoader.
- Runs QHiFi inference without the intensity rescaling / correction workflow
  used by TEST-NEW.
- Saves DWF (the model's PSF-blurred SIM input) and QWF (the network's
  WF-domain output) as float32 TIFF files in the normalized dataset range.
- Reports patch-wise PSNR, SSIM, MSE, and NRMSE for QWF vs WF and DWF vs WF.
- Does not save/evaluate QSIM, fit intensity scale factors, or perform ROI
  analysis and other post-processing. WF is used as the comparison reference,
  not as independent ground truth.

Expected model interface
------------------------
``qsim, qwf, sim_blurred = model(sim_image, wf_image, psf_tensor)``

The first output (QSIM) is intentionally unused here. The third output is the
PSF-blurred SIM image (DWF) before the intensity fitting performed by TEST-NEW.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import tifffile
import torch

from models.QHiFi import QHiFi
from util.util import img_comp


METRIC_NAMES = ("PSNR", "SSIM", "MSE", "NRMSE")


def _make_logger(output_dir: Path) -> logging.Logger:
    """Create a logger that writes to both the console and a log file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("QHiFiSIM_PatchTest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "test_model.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _load_checkpoint_file(path: Path, device: torch.device) -> Any:
    """Load a checkpoint while remaining compatible with older PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # ``weights_only`` is not accepted by some older PyTorch releases.
        return torch.load(path, map_location=device)


def _resolve_checkpoint(args, logger: logging.Logger) -> Path:
    """Resolve the test checkpoint, falling back to the current checkpoint folder."""
    explicit_path = str(getattr(args, "checkpoint_for_test", "") or "").strip()
    candidates = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    checkpoint_dir = Path(getattr(args, "checkpoint_path", "./checkpoint/")).expanduser()
    candidates.extend(
        [
            checkpoint_dir / "best_checkpoint.pth.tar",
            checkpoint_dir / "checkpoint_qHiFi.pth.tar",
        ]
    )

    # If there is no best checkpoint, use the most recent numbered epoch checkpoint.
    numbered = list(checkpoint_dir.glob("checkpoint_epoch*.pth.tar"))
    def epoch_number(path: Path) -> int:
        import re
        match = re.search(r"checkpoint_epoch(\d+)\.pth\.tar$", path.name)
        return int(match.group(1)) if match else -1
    numbered.sort(key=epoch_number, reverse=True)
    candidates.extend(numbered)

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            logger.info("Using checkpoint: %s", resolved)
            return resolved

    tried = "\n  - ".join(str(p) for p in candidates) if candidates else "(no path supplied)"
    raise FileNotFoundError(
        "Could not locate a QHiFi checkpoint. Paths checked:\n  - " + tried +
        "\nSet --checkpoint_for_test to the checkpoint file you want to evaluate."
    )


def _build_model(args, device: torch.device) -> QHiFi:
    """Build QHiFi with the requested ablation switches, then load its weights."""
    if getattr(args, "model_name", "QHiFi") != "QHiFi":
        raise ValueError(
            f"Unsupported model_name={getattr(args, 'model_name', None)!r}; this test script supports QHiFi only."
        )

    model = QHiFi(
        disable_ssfe=bool(getattr(args, "disable_ssfe", False)),
        disable_wife=bool(getattr(args, "disable_wife", False)),
        disable_channel_attn=bool(getattr(args, "disable_channel_attn", False)),
        disable_spatial_attn=bool(getattr(args, "disable_spatial_attn", False)),
        disable_identity=bool(getattr(args, "disable_identity", False)),
    ).to(device)
    return model


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Accept both training checkpoints and plain model state dictionaries."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format: expected a state dictionary or a checkpoint dictionary.")

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("The checkpoint does not contain a valid model state dictionary.")
    return state_dict


def _to_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Convert a BCHW tensor with batch size 1 to an HWC NumPy array."""
    array = tensor.detach().float().cpu().numpy()
    if array.ndim != 4 or array.shape[0] != 1:
        raise ValueError(f"Expected a BCHW tensor with batch size 1; received shape {array.shape}.")
    return np.transpose(array, (0, 2, 3, 1))[0]


def _save_tiff(path: Path, image_hwc: np.ndarray) -> None:
    """Save a model output as a single-channel float32 TIFF."""
    image = np.asarray(image_hwc, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected a single-channel 2D image, got shape {image.shape}.")
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite values encountered in output image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), image.astype(np.float32, copy=False))


def _compute_metrics(reference: np.ndarray, prediction: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute the same four image metrics used by the training validation code."""
    reference = np.asarray(reference, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Metric image shape mismatch: reference {reference.shape}, prediction {prediction.shape}."
        )
    # Match train.py's channel-last metric input convention and normalized data range.
    return tuple(float(v) for v in img_comp(reference, prediction, data_range=1.0))


def _get_dataset_item(dataset, index: int) -> Optional[dict]:
    """Return AugmentedDataset metadata when it is available."""
    items = getattr(dataset, "items", None)
    if isinstance(items, Sequence) and index < len(items):
        item = items[index]
        if isinstance(item, dict):
            return item
    return None


def _sample_stem(item: Optional[dict], index: int) -> str:
    """Build a stable, collision-resistant output stem from dataset metadata."""
    if not item:
        return f"patch_{index:06d}"

    sim_path = item.get("sim") or item.get("wf")
    source_stem = Path(str(sim_path)).stem if sim_path else "sample"
    x = item.get("x")
    y = item.get("y")

    if x is None or y is None:
        # Whole-frame sentinel items are intentionally not processed by this
        # patch-wise evaluator; this suffix is only a defensive fallback.
        suffix = "full"
    else:
        suffix = f"y{int(y):05d}_x{int(x):05d}"
    return f"patch_{index:06d}_{source_stem}_{suffix}"


def _write_metrics_csv(rows: list, output_path: Path) -> None:
    """Write per-patch metrics and source coordinates to CSV."""
    columns = [
        "dataset_index", "source_wf", "source_sim", "x_start", "y_start",
        "height", "width",
        "QWF_PSNR_vs_WF", "QWF_SSIM_vs_WF", "QWF_MSE_vs_WF", "QWF_NRMSE_vs_WF",
        "DWF_PSNR_vs_WF", "DWF_SSIM_vs_WF", "DWF_MSE_vs_WF", "DWF_NRMSE_vs_WF",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(rows: list, output_path: Path) -> None:
    """Write mean and standard deviation for QWF-vs-WF and DWF-vs-WF metrics."""
    keys = [
        "QWF_PSNR_vs_WF", "QWF_SSIM_vs_WF", "QWF_MSE_vs_WF", "QWF_NRMSE_vs_WF",
        "DWF_PSNR_vs_WF", "DWF_SSIM_vs_WF", "DWF_MSE_vs_WF", "DWF_NRMSE_vs_WF",
    ]
    lines = [
        "QHiFi-SIM patch-wise deep-learning test summary",
        f"Number of evaluated patches: {len(rows)}",
        "Note: QWF and DWF are each compared with the paired WF patch. WF is a comparison reference, not independent GT.",
        "",
    ]
    for key in keys:
        values = []
        for row in rows:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if values:
            arr = np.asarray(values, dtype=np.float64)
            lines.append(f"{key}: mean={arr.mean():.6f}, std={arr.std():.6f}, n={arr.size}")
        else:
            lines.append(f"{key}: unavailable")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test(args, test_loader, device):
    """Run patch-wise QHiFi evaluation using the test DataLoader built in main.py.

    The existing AugmentedDataset in ``mode='test'`` also appends a full-frame
    sentinel item (``x is None``) for direct whole-image inference. This function
    deliberately skips those sentinel items so that it only evaluates the
    training-style patches. A separate whole-image reconstruction script can
    handle full-frame outputs, independently of TEST-NEW's intensity analysis.
    """
    device = torch.device(device)
    output_root = Path(getattr(args, "results_path", "./test_outs/")) / "DL_test"
    dwf_dir = output_root / "DWF"
    qwf_dir = output_root / "QWF"
    dwf_dir.mkdir(parents=True, exist_ok=True)
    qwf_dir.mkdir(parents=True, exist_ok=True)
    logger = _make_logger(output_root)

    try:
        checkpoint_path = _resolve_checkpoint(args, logger)
        checkpoint = _load_checkpoint_file(checkpoint_path, device)
        model = _build_model(args, device)
        model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
        model.eval()

        dataset = getattr(test_loader, "dataset", None)
        dataset_items = getattr(dataset, "items", None)
        if dataset_items is None:
            logger.warning(
                "The test dataset does not expose AugmentedDataset.items; whole-frame sentinel filtering "
                "and source-coordinate naming may not be available."
            )

        rows = []
        skipped_full_frames = 0
        evaluated = 0
        logger.info("Device: %s", device)
        logger.info("Output directory: %s", output_root.resolve())
        logger.info("Starting patch-wise QHiFi evaluation.")

        with torch.inference_mode():
            for dataset_index, (wf_images, sim_images, psf_tensors) in enumerate(test_loader):
                item = _get_dataset_item(dataset, dataset_index) if dataset is not None else None

                # AugmentedDataset(mode='test') appends one x=None item per input image.
                # It belongs to full-frame inference, not to this patch-wise test.
                if item is not None and item.get("x") is None:
                    skipped_full_frames += 1
                    continue

                wf_image = wf_images.to(device=device, dtype=torch.float32)
                sim_image = sim_images.to(device=device, dtype=torch.float32)
                psf_tensor = psf_tensors.to(device=device, dtype=torch.float32)

                _qsim, qwf, dwf = model(sim_image, wf_image, psf_tensor)
                wf_hwc = _to_hwc(wf_image)
                qwf_hwc = _to_hwc(qwf)
                dwf_hwc = _to_hwc(dwf)

                stem = _sample_stem(item, dataset_index)
                _save_tiff(dwf_dir / f"{stem}.tif", dwf_hwc)
                _save_tiff(qwf_dir / f"{stem}.tif", qwf_hwc)

                row = {
                    "dataset_index": dataset_index,
                    "source_wf": str(item.get("wf", "")) if item else "",
                    "source_sim": str(item.get("sim", "")) if item else "",
                    "x_start": item.get("x", "") if item else "",
                    "y_start": item.get("y", "") if item else "",
                    "height": int(qwf_hwc.shape[0]),
                    "width": int(qwf_hwc.shape[1]),
                }

                # These are consistency metrics against the paired inputs, not GT-based metrics.
                try:
                    for prefix, reference, prediction in (
                        ("QWF", wf_hwc, qwf_hwc),
                        ("DWF", wf_hwc, dwf_hwc),
                    ):
                        metric_values = _compute_metrics(reference, prediction)
                        for metric_name, value in zip(METRIC_NAMES, metric_values):
                            row[f"{prefix}_{metric_name}_vs_WF"] = value
                except Exception as exc:
                    logger.warning(
                        "Metric calculation failed for dataset item %d (%s): %s",
                        dataset_index, stem, exc,
                    )
                    for key in (
                        "QWF_PSNR_vs_WF", "QWF_SSIM_vs_WF", "QWF_MSE_vs_WF", "QWF_NRMSE_vs_WF",
                        "DWF_PSNR_vs_WF", "DWF_SSIM_vs_WF", "DWF_MSE_vs_WF", "DWF_NRMSE_vs_WF",
                    ):
                        row[key] = float("nan")

                rows.append(row)
                evaluated += 1
                logger.info(
                    "[%d] %s | shape=%dx%d",
                    evaluated, stem, qwf_hwc.shape[0], qwf_hwc.shape[1],
                )

        if evaluated == 0:
            raise RuntimeError(
                "No patch samples were evaluated. Check the test list and AugmentedDataset item generation."
            )

        _write_metrics_csv(rows, output_root / "patch_metrics.csv")
        _write_summary(rows, output_root / "metrics_summary.txt")

        logger.info("Evaluated patches: %d", evaluated)
        logger.info("Skipped full-frame sentinel items: %d", skipped_full_frames)
        logger.info("Saved DWF outputs to: %s", dwf_dir.resolve())
        logger.info("Saved QWF outputs to: %s", qwf_dir.resolve())
        logger.info("Saved per-patch metrics to: %s", (output_root / 'patch_metrics.csv').resolve())
        logger.info("Finished patch-wise deep-learning test.")
        return True

    except Exception as exc:
        logger.exception("QHiFi patch-wise testing failed: %s", exc)
        raise
