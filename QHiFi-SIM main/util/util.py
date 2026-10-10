from __future__ import print_function
import numpy as np
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import normalized_root_mse as compare_nrmse


def img_norm(x):
    max_val = np.max(x)
    x_norm = x / (max_val + 1e-8)
    return x_norm


def _init_metrics():
    return [], [], [], []  # mses, nrmses, psnrs, ssims


def _check_shape_consistency(gt, pr):
    if gt.shape != pr.shape:
        raise ValueError(f"GT shape {gt.shape} != PR shape {pr.shape}")


def _get_data_range(gt, data_range=None):
    if data_range is not None:
        return float(data_range)
    data_range = np.max(gt) - np.min(gt)
    if np.isclose(data_range, 0):
        return 1e-8
    return data_range


def _safe_divide(numerator, denominator, default=0.0):
    if np.isclose(denominator, 0):
        return default
    return numerator / denominator


def img_comp(gt, pr, data_range=None,):
    mses, nrmses, psnrs, ssims = _init_metrics()
    gt = np.squeeze(gt).astype(np.float32)
    pr = np.squeeze(pr).astype(np.float32)
    _check_shape_consistency(gt, pr)

    if gt.ndim == 2:
        gt = np.expand_dims(gt, 0)
        pr = np.expand_dims(pr, 0)
    elif gt.ndim != 3:
        raise ValueError(f"Input must be 2D (single frame) or 3D (multi-frame). Got GT dimension: {gt.ndim}")

    for i in range(gt.shape[0]):
        gt_frame = gt[i]
        pr_frame = pr[i]

        frame_data_range = _get_data_range(gt_frame, data_range)
        gt_is_constant = np.allclose(gt_frame, gt_frame.flat[0])

        mse_val = compare_mse(gt_frame, pr_frame)
        mses.append(mse_val)
        if gt_is_constant:
            if np.isclose(mse_val, 0):
                psnr_val = float('inf')
                ssim_val = 1.0
                nrmse_val = 0.0
            else:
                psnr_val = 0.0
                ssim_val = 0.0
                nrmse_val = compare_nrmse(
                    gt_frame, pr_frame, normalization='standard_deviation'
                )
        else:
            if np.isclose(mse_val, 0):
                psnr_val = float('inf')
            else:
                psnr_val = compare_psnr(
                    gt_frame, pr_frame, data_range=frame_data_range
                )

            ssim_val = compare_ssim(
                gt_frame, pr_frame,
                data_range=frame_data_range,
                channel_axis=None
            )

            try:
                nrmse_val = compare_nrmse(
                    gt_frame, pr_frame, normalization='min-max'
                )
            except ValueError:
                nrmse_val = compare_nrmse(
                    gt_frame, pr_frame, normalization='standard_deviation'
                )

        psnrs.append(psnr_val)
        ssims.append(ssim_val)
        nrmses.append(nrmse_val)

    psnrs_finite = [p if not np.isinf(p) else 100.0 for p in psnrs]

    return (
        np.mean(psnrs_finite) if psnrs_finite else 0.0,
        np.mean(ssims) if ssims else 0.0,
        np.mean(mses) if mses else 0.0,
        np.mean(nrmses) if nrmses else 0.0,
    )

