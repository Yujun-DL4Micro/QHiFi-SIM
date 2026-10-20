"""Standalone quantitative analysis for full-image QHiFi-SIM outputs.

This script does not run the neural network. It reads the original WF/SIM
images plus outputs saved by ``test_full_image.py`` and reproduces the main
quantitative analyses from the previous combined testing script.

Configure paths and ROI sizes in the USER CONFIGURATION section, then run:
    python analyze_quantitative.py
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile

from util.util import img_comp


# =============================================================================
# USER CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# These input folders must correspond to the same numeric ordering used for
# full-image inference. The result folder is created by test_full_image.py.
WF_DIR = Path("./data/WF")
SIM_INPUT_DIR = Path("./data/SIM")
RESULT_DIR = Path("./full_image_results")

# Set to a GT folder to enable SIM/QSIM-vs-GT analysis; use None when GT is absent.
GT_DIR: Optional[Path] = None

# ROI window sizes are user-configurable here. (9, 15) reproduces the original
# 9x9 and 15x15 ROI-sum analyses. Change these integers as needed.
ROI_SIZES = (9, 15)
NUM_ROIS = 500

ROI_BORDER = 30
BG_THRESHOLD = 0.05
MIN_FOREGROUND_RATIO = 0.8

# Set an integer to make the random ROI coordinates repeatable. None preserves
# the unseeded random sampling behavior of the previous script.
RANDOM_SEED: Optional[int] = None

SAVE_PIXEL_VALUES = True       # A potentially large Excel workbook; requires GT_DIR.
SAVE_DIFFERENCE_IMAGES = True
SAVE_ROI_MARKED_IMAGES = True
# =============================================================================


METRICS = ("PSNR", "SSIM", "MSE", "NRMSE")
METHODS = ("SIM", "QSIM", "DWF", "QWF")


def resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("QHiFiSIMQuantitativeAnalysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "analysis_log.txt", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def natural_sort_key(path: Path) -> tuple[int, str]:
    matches = re.findall(r"\d+", path.stem)
    number = int(matches[0]) if matches else -1
    return number, path.name.lower()


def list_tiffs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
    return sorted(files, key=natural_sort_key)


def read_image(path: Path) -> np.ndarray:
    image = np.asarray(tifffile.imread(path), dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D single-channel TIFF: {path} (shape={image.shape})")
    if not np.isfinite(image).all():
        raise ValueError(f"Image contains NaN or infinite values: {path}")
    return image


def ensure_same_shape(reference: np.ndarray, candidate: np.ndarray, label: str) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(f"Image shape mismatch for {label}: {reference.shape} vs {candidate.shape}")


def compute_r2(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Squared Pearson correlation, matching the previous full-image analysis calculation."""
    x = np.asarray(image_a, dtype=np.float64).ravel()
    y = np.asarray(image_b, dtype=np.float64).ravel()
    if x.size != y.size or x.size == 0:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def generate_rois(
    shape: tuple[int, int],
    patch_size: int,
    num_rois: int = 500,
    border: int = 30,
    image: Optional[np.ndarray] = None,
    bg_threshold: float = 0.05,
    min_foreground_ratio: float = 0.8,
) -> list[tuple[int, int]]:
    """Generate grid-spread ROI top-left coordinates using the previous logic."""
    if patch_size <= 0:
        raise ValueError(f"ROI size must be positive, got {patch_size}")
    h, w = shape
    max_y = h - patch_size - border
    max_x = w - patch_size - border
    if max_y < border or max_x < border:
        raise ValueError(
            f"Image too small for ROI size={patch_size} and border={border}: shape={shape}"
        )

    foreground_mask = None
    if image is not None:
        max_value = float(np.max(image))
        foreground_mask = image > (bg_threshold * max_value)

    def is_valid(y: int, x: int) -> bool:
        if foreground_mask is None:
            return True
        region = foreground_mask[y:y + patch_size, x:x + patch_size]
        return bool(region.mean() >= min_foreground_ratio)

    def random_coordinate() -> tuple[int, int]:
        y = int(np.random.randint(border, max_y + 1))
        x = int(np.random.randint(border, max_x + 1))
        return y, x

    rois: list[tuple[int, int]] = []
    grid_size = max(patch_size + 1, int(patch_size * 1.5))
    grid_h = (h - 2 * border) // grid_size
    grid_w = (w - 2 * border) // grid_size

    for grid_y in range(grid_h):
        for grid_x in range(grid_w):
            y_start = border + grid_y * grid_size
            x_start = border + grid_x * grid_size
            y_end = min(y_start + grid_size - patch_size, max_y)
            x_end = min(x_start + grid_size - patch_size, max_x)
            if y_end < y_start or x_end < x_start:
                continue
            for _ in range(10):
                y = int(np.random.randint(y_start, y_end + 1))
                x = int(np.random.randint(x_start, x_end + 1))
                if is_valid(y, x):
                    rois.append((y, x))
                    break

    if len(rois) < num_rois:
        attempt = 0
        max_attempts = 2000 * num_rois
        while len(rois) < num_rois and attempt < max_attempts:
            y, x = random_coordinate()
            if is_valid(y, x):
                rois.append((y, x))
            attempt += 1
    else:
        np.random.shuffle(rois)
        rois = rois[:num_rois]

    return rois


def roi_sums(image: np.ndarray, rois: list[tuple[int, int]], patch_size: int) -> np.ndarray:
    return np.asarray(
        [image[y:y + patch_size, x:x + patch_size].sum() for y, x in rois],
        dtype=np.float64,
    )


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    result[~np.isfinite(result)] = np.nan
    return result


def draw_rois_on_image(image: np.ndarray, rois: list[tuple[int, int]], patch_size: int) -> np.ndarray:
    scaled = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    rgb = np.stack([scaled, scaled, scaled], axis=-1).astype(np.float32)
    for y, x in rois:
        y2 = min(y + patch_size - 1, rgb.shape[0] - 1)
        x2 = min(x + patch_size - 1, rgb.shape[1] - 1)
        rgb[y, x:x2 + 1] = (1.0, 1.0, 0.0)
        rgb[y2, x:x2 + 1] = (1.0, 1.0, 0.0)
        rgb[y:y2 + 1, x] = (1.0, 1.0, 0.0)
        rgb[y:y2 + 1, x2] = (1.0, 1.0, 0.0)
    return rgb


def save_rgb_tiff(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(image, dtype=np.float32))


def mean_ignore_nan(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def std_ignore_nan(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.std(array)) if array.size else float("nan")


def result_file(result_dir: Path, subfolder: str, stem: str) -> Path:
    folder = result_dir / subfolder
    for suffix in (".tif", ".tiff", ".TIF", ".TIFF"):
        candidate = folder / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {subfolder} result for '{stem}' in {folder}")


def main() -> None:
    wf_dir = resolve_path(WF_DIR)
    sim_input_dir = resolve_path(SIM_INPUT_DIR)
    result_dir = resolve_path(RESULT_DIR)
    gt_dir = resolve_path(GT_DIR) if GT_DIR is not None else None
    output_dir = result_dir / "Quant_Analysis"
    logger = setup_logger(output_dir)

    if not ROI_SIZES or any(int(size) <= 0 for size in ROI_SIZES):
        raise ValueError(f"ROI_SIZES must contain positive integers, got {ROI_SIZES}")
    roi_sizes = tuple(dict.fromkeys(int(size) for size in ROI_SIZES))
    if NUM_ROIS <= 0 or ROI_BORDER < 0:
        raise ValueError("NUM_ROIS must be positive and ROI_BORDER must be non-negative.")
    if RANDOM_SEED is not None:
        np.random.seed(int(RANDOM_SEED))

    wf_files = list_tiffs(wf_dir)
    sim_input_files = list_tiffs(sim_input_dir)
    if not wf_files:
        raise RuntimeError(f"No WF TIFF files found in {wf_dir}")
    if len(wf_files) != len(sim_input_files):
        raise ValueError(
            f"WF/SIM input count mismatch: {len(wf_files)} WF files vs {len(sim_input_files)} SIM files"
        )

    gt_files: Optional[list[Path]] = None
    if gt_dir is not None:
        if gt_dir.is_dir():
            possible_gt = list_tiffs(gt_dir)
            if len(possible_gt) == len(wf_files):
                gt_files = possible_gt
            else:
                logger.warning(
                    "GT file count (%d) does not match WF count (%d); GT-based analysis is disabled.",
                    len(possible_gt), len(wf_files),
                )
        else:
            logger.warning("GT_DIR was configured but does not exist; GT-based analysis is disabled: %s", gt_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    roi_tables_dir = output_dir / "ROI_sum_tables"
    roi_tables_dir.mkdir(parents=True, exist_ok=True)
    difference_dir = output_dir / "difference_images"
    marked_dir = output_dir / "ROI_marked_images"
    if SAVE_DIFFERENCE_IMAGES:
        difference_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_ROI_MARKED_IMAGES:
        marked_dir.mkdir(parents=True, exist_ok=True)

    pixel_writer = None
    pixel_excel_path = output_dir / "Pixel_values" / "pixel_values.xlsx"
    if SAVE_PIXEL_VALUES and gt_files is not None:
        pixel_excel_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("WF samples: %d", len(wf_files))
    logger.info("ROI sizes: %s", ", ".join(f"{size}x{size}" for size in roi_sizes))
    logger.info("ROIs requested per size per image: %d", NUM_ROIS)
    logger.info("GT analysis: %s", "enabled" if gt_files is not None else "disabled")

    summary_rows: list[dict] = []
    before_metrics = {metric.lower(): [] for metric in METRICS}
    after_metrics = {metric.lower(): [] for metric in METRICS}
    r2_all = {"DWF_vs_WF_pixel": [], "QWF_vs_WF_pixel": []}
    for size in roi_sizes:
        r2_all[f"DWF_vs_WF_{size}x{size}"] = []
        r2_all[f"QWF_vs_WF_{size}x{size}"] = []
        r2_all[f"SIM_vs_GT_{size}x{size}"] = []
        r2_all[f"QSIM_vs_GT_{size}x{size}"] = []
    r2_all.update({
        "SIM_vs_GT_pixel": [], "QSIM_vs_GT_pixel": [],
        "SIM_vs_GT_PSNR": [], "SIM_vs_GT_SSIM": [], "SIM_vs_GT_MSE": [], "SIM_vs_GT_NRMSE": [],
        "QSIM_vs_GT_PSNR": [], "QSIM_vs_GT_SSIM": [], "QSIM_vs_GT_MSE": [], "QSIM_vs_GT_NRMSE": [],
    })

    try:
        for index, wf_path in enumerate(wf_files):
            sim_input_path = sim_input_files[index]
            stem = wf_path.stem
            logger.info("Analyzing %d/%d: %s", index + 1, len(wf_files), stem)

            wf = read_image(wf_path)
            # Use the rescaled SIM output from inference for the quantitative tables,
            # matching the original combined testing workflow.
            images = {name: read_image(result_file(result_dir, name, stem)) for name in METHODS}
            for name, image in images.items():
                ensure_same_shape(wf, image, f"{name} / WF for {stem}")

            gt = read_image(gt_files[index]) if gt_files is not None else None
            if gt is not None:
                ensure_same_shape(wf, gt, f"GT / WF for {stem}")

            qwf_scores = img_comp(wf, images["QWF"])
            dwf_scores = img_comp(wf, images["DWF"])
            for metric, value in zip(("psnr", "ssim", "mse", "nrmse"), dwf_scores):
                before_metrics[metric].append(float(value))
            for metric, value in zip(("psnr", "ssim", "mse", "nrmse"), qwf_scores):
                after_metrics[metric].append(float(value))

            row: dict = {
                "sample_index": index,
                "sample": stem,
                "wf_input": wf_path.name,
                "sim_input": sim_input_path.name,
                "DWF_vs_WF_PSNR": float(dwf_scores[0]),
                "DWF_vs_WF_SSIM": float(dwf_scores[1]),
                "DWF_vs_WF_MSE": float(dwf_scores[2]),
                "DWF_vs_WF_NRMSE": float(dwf_scores[3]),
                "QWF_vs_WF_PSNR": float(qwf_scores[0]),
                "QWF_vs_WF_SSIM": float(qwf_scores[1]),
                "QWF_vs_WF_MSE": float(qwf_scores[2]),
                "QWF_vs_WF_NRMSE": float(qwf_scores[3]),
                "DWF_vs_WF_pixel_R2": compute_r2(images["DWF"], wf),
                "QWF_vs_WF_pixel_R2": compute_r2(images["QWF"], wf),
            }
            r2_all["DWF_vs_WF_pixel"].append(row["DWF_vs_WF_pixel_R2"])
            r2_all["QWF_vs_WF_pixel"].append(row["QWF_vs_WF_pixel_R2"])

            roi_by_size: dict[int, list[tuple[int, int]]] = {}
            for size in roi_sizes:
                rois = generate_rois(
                    wf.shape,
                    size,
                    num_rois=NUM_ROIS,
                    border=ROI_BORDER,
                    image=wf,
                    bg_threshold=BG_THRESHOLD,
                    min_foreground_ratio=MIN_FOREGROUND_RATIO,
                )
                roi_by_size[size] = rois
                if len(rois) < NUM_ROIS:
                    logger.warning("%s: only %d/%d valid ROIs for size %dx%d", stem, len(rois), NUM_ROIS, size, size)

                summed = {name: roi_sums(image, rois, size) for name, image in {"WF": wf, **images}.items()}
                roi_r2_dwf = compute_r2(summed["DWF"], summed["WF"])
                roi_r2_qwf = compute_r2(summed["QWF"], summed["WF"])
                row[f"DWF_vs_WF_{size}x{size}_R2"] = roi_r2_dwf
                row[f"QWF_vs_WF_{size}x{size}_R2"] = roi_r2_qwf
                r2_all[f"DWF_vs_WF_{size}x{size}"].append(roi_r2_dwf)
                r2_all[f"QWF_vs_WF_{size}x{size}"].append(roi_r2_qwf)

                table = pd.DataFrame({
                    "WF": summed["WF"],
                    "SIM": summed["SIM"],
                    "DWF": summed["DWF"],
                    "QWF": summed["QWF"],
                    "QSIM": summed["QSIM"],
                })
                table["DWF/WF"] = safe_ratio(table["DWF"].to_numpy(), table["WF"].to_numpy())
                table["QWF/WF"] = safe_ratio(table["QWF"].to_numpy(), table["WF"].to_numpy())
                if gt is not None:
                    gt_sums = roi_sums(gt, rois, size)
                    table["GT"] = gt_sums
                    table["SIM/GT"] = safe_ratio(table["SIM"].to_numpy(), gt_sums)
                    table["QSIM/GT"] = safe_ratio(table["QSIM"].to_numpy(), gt_sums)
                    sim_gt_r2 = compute_r2(summed["SIM"], gt_sums)
                    qsim_gt_r2 = compute_r2(summed["QSIM"], gt_sums)
                    row[f"SIM_vs_GT_{size}x{size}_R2"] = sim_gt_r2
                    row[f"QSIM_vs_GT_{size}x{size}_R2"] = qsim_gt_r2
                    r2_all[f"SIM_vs_GT_{size}x{size}"].append(sim_gt_r2)
                    r2_all[f"QSIM_vs_GT_{size}x{size}"].append(qsim_gt_r2)

                table_path = roi_tables_dir / f"{stem}_roi_sum_{size}x{size}.xlsx"
                table.to_excel(table_path, index=False)

                if SAVE_ROI_MARKED_IMAGES:
                    for method in METHODS:
                        marked = draw_rois_on_image(images[method], rois, size)
                        save_rgb_tiff(marked_dir / f"{stem}_{method}_rois_{size}x{size}.tif", marked)

            if gt is not None:
                for method, metric_prefix, values_key in (
                    ("SIM", "SIM_vs_GT", "SIM_vs_GT"),
                    ("QSIM", "QSIM_vs_GT", "QSIM_vs_GT"),
                ):
                    score = img_comp(gt, images[method])
                    for metric, value in zip(("PSNR", "SSIM", "MSE", "NRMSE"), score):
                        row[f"{metric_prefix}_{metric}"] = float(value)
                        r2_all[f"{metric_prefix}_{metric}"].append(float(value))
                    pixel_r2 = compute_r2(images[method], gt)
                    row[f"{metric_prefix}_pixel_R2"] = pixel_r2
                    r2_all[f"{metric_prefix}_pixel"].append(pixel_r2)

            if SAVE_DIFFERENCE_IMAGES:
                tifffile.imwrite(difference_dir / f"{stem}_DWF_minus_WF.tif", np.abs(images["DWF"] - wf).astype(np.float32))
                tifffile.imwrite(difference_dir / f"{stem}_QWF_minus_WF.tif", np.abs(images["QWF"] - wf).astype(np.float32))
                if gt is not None:
                    for method in ("SIM", "QSIM", "QWF"):
                        tifffile.imwrite(
                            difference_dir / f"{stem}_{method}_minus_GT.tif",
                            np.abs(images[method] - gt).astype(np.float32),
                        )

            if SAVE_PIXEL_VALUES and gt is not None:
                margin = 30
                height, width = wf.shape
                if height > 2 * margin and width > 2 * margin:
                    section = (slice(margin, -margin), slice(margin, -margin))
                    pixel_data = {
                        "WF": wf[section].ravel(),
                        "SIM": images["SIM"][section].ravel(),
                        "DWF": images["DWF"][section].ravel(),
                        "QWF": images["QWF"][section].ravel(),
                        "QSIM": images["QSIM"][section].ravel(),
                        "GT": gt[section].ravel(),
                    }
                    pixel_data["DWF/WF"] = safe_ratio(pixel_data["DWF"], pixel_data["WF"])
                    pixel_data["QWF/WF"] = safe_ratio(pixel_data["QWF"], pixel_data["WF"])
                    pixel_data["SIM/GT"] = safe_ratio(pixel_data["SIM"], pixel_data["GT"])
                    pixel_data["QSIM/GT"] = safe_ratio(pixel_data["QSIM"], pixel_data["GT"])
                    pixel_frame = pd.DataFrame(pixel_data)
                    if pixel_writer is None:
                        pixel_writer = pd.ExcelWriter(pixel_excel_path, engine="openpyxl")
                    # Excel sheet names are limited to 31 characters.
                    sheet_name = re.sub(r"[^A-Za-z0-9_]", "_", f"Sample_{stem}")[:31]
                    pixel_frame.to_excel(pixel_writer, sheet_name=sheet_name, index=False)
                else:
                    logger.warning("%s: image too small for pixel export with 30-pixel border; skipped.", stem)

            summary_rows.append(row)

    finally:
        if pixel_writer is not None:
            pixel_writer.close()
            logger.info("Pixel-level values saved to %s", pixel_excel_path)

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_excel(output_dir / "R2_summary.xlsx", index=False)
    summary_frame.to_csv(output_dir / "per_sample_metrics.csv", index=False, encoding="utf-8-sig")

    summary_lines = [
        "QHiFi-SIM quantitative analysis summary",
        f"Samples analyzed: {len(summary_rows)}",
        f"ROI sizes: {', '.join(f'{s}x{s}' for s in roi_sizes)}",
        f"GT analysis: {'enabled' if gt_files is not None else 'disabled'}",
        "",
        "DWF versus WF (before correction):",
    ]
    for metric in ("psnr", "ssim", "mse", "nrmse"):
        summary_lines.append(
            f"  {metric.upper()}: mean={mean_ignore_nan(before_metrics[metric]):.6g}, "
            f"std={std_ignore_nan(before_metrics[metric]):.6g}"
        )
    summary_lines.append("QWF versus WF (after correction):")
    for metric in ("psnr", "ssim", "mse", "nrmse"):
        summary_lines.append(
            f"  {metric.upper()}: mean={mean_ignore_nan(after_metrics[metric]):.6g}, "
            f"std={std_ignore_nan(after_metrics[metric]):.6g}"
        )
    summary_lines.append("")
    summary_lines.append("R-squared summary (mean +/- standard deviation):")
    for key, values in r2_all.items():
        if values:
            summary_lines.append(f"  {key}: {mean_ignore_nan(values):.6g} +/- {std_ignore_nan(values):.6g}")
    (output_dir / "analysis_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    logger.info("Analysis complete. Results saved to %s", output_dir)
    logger.info("Summary workbook: %s", output_dir / "R2_summary.xlsx")


if __name__ == "__main__":
    main()
