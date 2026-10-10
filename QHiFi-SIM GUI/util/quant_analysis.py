import os
import glob
import re
import numpy as np
import tifffile
import pandas as pd
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import logging


def setup_logger(output_dir):
    logger = logging.getLogger("QuantAnalysisLogger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "quant_analysis_log.txt")
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode='w')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def extract_number(filename):
    basename = os.path.basename(filename)
    numbers = re.findall(r'\d+', basename)
    return int(numbers[0]) if numbers else 0


def compute_r2(img1, img2):
    x = img1.ravel()
    y = img2.ravel()
    corr = np.corrcoef(x, y)[0, 1]
    return corr ** 2


def generate_rois(shape, patch_size, num_rois=500, border=20, image=None, bg_threshold=0.05, min_foreground_ratio=0.8):
    h, w = shape
    max_y = h - patch_size - border
    max_x = w - patch_size - border
    if max_y < border or max_x < border:
        raise ValueError(f"Image too small for border={border} and patch_size={patch_size}")

    if image is not None:
        max_val = image.max()
        fg_mask = image > (bg_threshold * max_val)
    else:
        fg_mask = None

    def is_valid_roi(y, x):
        if fg_mask is None:
            return True
        patch = fg_mask[y:y + patch_size, x:x + patch_size]
        return patch.mean() >= min_foreground_ratio

    rois = []
    attempt = 0
    max_attempts = 2000 * num_rois
    while len(rois) < num_rois and attempt < max_attempts:
        y = np.random.randint(border, max_y + 1)
        x = np.random.randint(border, max_x + 1)
        if is_valid_roi(y, x):
            rois.append((y, x))
        attempt += 1
    return rois


def patch_sum_r2(img1, img2, rois, patch_size):
    sums1 = np.array([img1[y:y + patch_size, x:x + patch_size].sum() for (y, x) in rois])
    sums2 = np.array([img2[y:y + patch_size, x:x + patch_size].sum() for (y, x) in rois])
    return compute_r2(sums1, sums2)


def collect_roi_sums(image, rois, patch_size):
    return [image[y:y + patch_size, x:x + patch_size].sum() for (y, x) in rois]


def save_roi_sum_table(roi_sums_dict, output_path):
    df = pd.DataFrame(roi_sums_dict)
    df.to_excel(output_path, index=False)


def draw_rois_on_image(image, rois, patch_size):
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
    for (y, x) in rois:
        rgb[y, x:x + patch_size, 0] = 1.0
        rgb[y, x:x + patch_size, 1] = 1.0
        rgb[y, x:x + patch_size, 2] = 0.0
        rgb[y + patch_size - 1, x:x + patch_size, 0] = 1.0
        rgb[y + patch_size - 1, x:x + patch_size, 1] = 1.0
        rgb[y + patch_size - 1, x:x + patch_size, 2] = 0.0
        rgb[y:y + patch_size, x, 0] = 1.0
        rgb[y:y + patch_size, x, 1] = 1.0
        rgb[y:y + patch_size, x, 2] = 0.0
        rgb[y:y + patch_size, x + patch_size - 1, 0] = 1.0
        rgb[y:y + patch_size, x + patch_size - 1, 1] = 1.0
        rgb[y:y + patch_size, x + patch_size - 1, 2] = 0.0
    return rgb


def compute_metrics(img1, img2):
    data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    if data_range == 0:
        data_range = 1.0
    psnr = peak_signal_noise_ratio(img1, img2, data_range=data_range)
    ssim = structural_similarity(img1, img2, data_range=data_range)
    mse = np.mean((img1 - img2) ** 2)
    nrmse = np.sqrt(mse) / data_range
    return psnr, ssim, mse, nrmse


def run_quant_analysis(wf_dir, sim_dir, output_dir):
    logger = setup_logger(output_dir)
    logger.info("Starting quantitative analysis...")

    wf_files = sorted(glob.glob(os.path.join(wf_dir, "*.tif")), key=extract_number)
    sim_files = sorted(glob.glob(os.path.join(sim_dir, "*.tif")), key=extract_number)

    if len(wf_files) != len(sim_files):
        logger.error(f"File count mismatch: {len(wf_files)} WF vs {len(sim_files)} SIM")
        return False, "File count mismatch"

    analysis_dir = os.path.join(output_dir, "Quant_Analysis")
    roi_dir = os.path.join(analysis_dir, "ROI_sum_tables")
    diff_dir = os.path.join(analysis_dir, "Difference_Images")
    marked_dir = os.path.join(analysis_dir, "ROI_marked_images")
    for d in [analysis_dir, roi_dir, diff_dir, marked_dir]:
        os.makedirs(d, exist_ok=True)

    results = []

    for idx in range(len(wf_files)):
        wf_orig = tifffile.imread(wf_files[idx]).astype(np.float32)
        sim_raw = tifffile.imread(sim_files[idx]).astype(np.float32)

        base = f"{idx:04d}"
        qwf_path = os.path.join(output_dir, "QWF", f"qwf_{base}.tif")
        qsim_path = os.path.join(output_dir, "QHiFi", f"qHiFi_{base}.tif")
        dwf_path = os.path.join(output_dir, "DWF", f"dwf_{base}.tif")
        sim_path = os.path.join(output_dir, "HiFi", f"HiFi_{base}.tif")

        if not all(os.path.exists(p) for p in [qwf_path, qsim_path, dwf_path, sim_path]):
            logger.warning(f"Missing output for sample {idx}, skipping...")
            continue

        qwf_final = tifffile.imread(qwf_path).astype(np.float32)
        qsim_final = tifffile.imread(qsim_path).astype(np.float32)
        dwf_final = tifffile.imread(dwf_path).astype(np.float32)
        sim_final = tifffile.imread(sim_path).astype(np.float32)

        # 检查所有图像的尺寸是否一致
        shapes = [wf_orig.shape, sim_final.shape, dwf_final.shape, qwf_final.shape, qsim_final.shape]
        if not all(s == shapes[0] for s in shapes):
            logger.warning(f"Sample {idx} has inconsistent dimensions: {shapes}. Skipping.")
            continue

        psnr_qwf, ssim_qwf, mse_qwf, nrmse_qwf = compute_metrics(wf_orig, qwf_final)
        psnr_dwf, ssim_dwf, mse_dwf, nrmse_dwf = compute_metrics(wf_orig, dwf_final)

        logger.info(f"Sample {idx} - DWF vs WF: PSNR={psnr_dwf:.4f}, SSIM={ssim_dwf:.4f}, MSE={mse_dwf:.4f}, NRMSE={nrmse_dwf:.4f}")
        logger.info(f"Sample {idx} - QWF vs WF: PSNR={psnr_qwf:.4f}, SSIM={ssim_qwf:.4f}, MSE={mse_qwf:.4f}, NRMSE={nrmse_qwf:.4f}")

        rois_6 = generate_rois(wf_orig.shape, 6, 500, border=30, image=wf_orig)
        rois_12 = generate_rois(wf_orig.shape, 12, 500, border=30, image=wf_orig)

        dwf_r2_pixel = compute_r2(dwf_final, wf_orig)
        qwf_r2_pixel = compute_r2(qwf_final, wf_orig)
        dwf_r2_6 = patch_sum_r2(dwf_final, wf_orig, rois_6, 6)
        dwf_r2_12 = patch_sum_r2(dwf_final, wf_orig, rois_12, 12)
        qwf_r2_6 = patch_sum_r2(qwf_final, wf_orig, rois_6, 6)
        qwf_r2_12 = patch_sum_r2(qwf_final, wf_orig, rois_12, 12)

        result = {
            'sample_index': idx,
            'DWF_vs_WF_pixel': dwf_r2_pixel,
            'DWF_vs_WF_6x6': dwf_r2_6,
            'DWF_vs_WF_12x12': dwf_r2_12,
            'QWF_vs_WF_pixel': qwf_r2_pixel,
            'QWF_vs_WF_6x6': qwf_r2_6,
            'QWF_vs_WF_12x12': qwf_r2_12,
            'DWF_vs_WF_PSNR': psnr_dwf,
            'DWF_vs_WF_SSIM': ssim_dwf,
            'QWF_vs_WF_PSNR': psnr_qwf,
            'QWF_vs_WF_SSIM': ssim_qwf,
        }
        results.append(result)

        # Save ROI tables (no GT)
        sums_6 = {
            'WF': collect_roi_sums(wf_orig, rois_6, 6),
            'SIM': collect_roi_sums(sim_final, rois_6, 6),
            'DWF': collect_roi_sums(dwf_final, rois_6, 6),
            'QWF': collect_roi_sums(qwf_final, rois_6, 6),
            'QSIM': collect_roi_sums(qsim_final, rois_6, 6)
        }
        wf_sums = sums_6['WF']
        dwf_sums = sums_6['DWF']
        qwf_sums = sums_6['QWF']
        sums_6['DWF/WF'] = [d / w if w != 0 else np.nan for d, w in zip(dwf_sums, wf_sums)]
        sums_6['QWF/WF'] = [q / w if w != 0 else np.nan for q, w in zip(qwf_sums, wf_sums)]
        save_roi_sum_table(sums_6, os.path.join(roi_dir, f'{base}_roi_sum_6x6.xlsx'))

        sums_12 = {
            'WF': collect_roi_sums(wf_orig, rois_12, 12),
            'SIM': collect_roi_sums(sim_final, rois_12, 12),
            'DWF': collect_roi_sums(dwf_final, rois_12, 12),
            'QWF': collect_roi_sums(qwf_final, rois_12, 12),
            'QSIM': collect_roi_sums(qsim_final, rois_12, 12)
        }
        wf_sums = sums_12['WF']
        dwf_sums = sums_12['DWF']
        qwf_sums = sums_12['QWF']
        sums_12['DWF/WF'] = [d / w if w != 0 else np.nan for d, w in zip(dwf_sums, wf_sums)]
        sums_12['QWF/WF'] = [q / w if w != 0 else np.nan for q, w in zip(qwf_sums, wf_sums)]
        save_roi_sum_table(sums_12, os.path.join(roi_dir, f'{base}_roi_sum_12x12.xlsx'))

        # Save difference images
        tifffile.imwrite(os.path.join(diff_dir, f'{base}_DWF_minus_WF.tif'), np.abs(dwf_final - wf_orig).astype(np.float32))
        tifffile.imwrite(os.path.join(diff_dir, f'{base}_QWF_minus_WF.tif'), np.abs(qwf_final - wf_orig).astype(np.float32))

        # Save ROI marked images
        for name, img in [('DWF', dwf_final), ('QWF', qwf_final), ('SIM', sim_final), ('QSIM', qsim_final)]:
            marked_6 = draw_rois_on_image(img, rois_6, 6)
            tifffile.imwrite(os.path.join(marked_dir, f'{base}_{name}_rois_6x6.tif'), marked_6.astype(np.float32))
            marked_12 = draw_rois_on_image(img, rois_12, 12)
            tifffile.imwrite(os.path.join(marked_dir, f'{base}_{name}_rois_12x12.tif'), marked_12.astype(np.float32))

    df = pd.DataFrame(results)
    df.to_excel(os.path.join(analysis_dir, "R2_summary.xlsx"), index=False)
    logger.info(f"Quantitative analysis completed. Results saved to {analysis_dir}")
    return True, f"Analysis completed. Results saved to {analysis_dir}"