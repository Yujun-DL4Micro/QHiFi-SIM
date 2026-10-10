"""
SIM simulation for QHiFi-SIM.

Generates simulated SIM raw frames, GT image and PSF reference
under configurable optical and noise parameters.
"""

import os
import shutil
import numpy as np
from imageio.v3 import imread, imwrite

def val_ideal_otf(v: float) -> float:

    if v < 0.0 or v > 1.0:
        return 0.0
    return (1.0 / np.pi) * (2.0 * np.arccos(v) - np.sin(2.0 * np.arccos(v)))

def get_otf_val(otf: dict, cyc: np.ndarray) -> np.ndarray:

    vals = otf['vals']
    sample_lateral = len(vals)
    cycles_per_micron = otf['cyclesPerMicron']

    cpos = cyc / cycles_per_micron + 1.0
    lpos = np.floor(cpos).astype(np.int32)
    hpos = np.ceil(cpos).astype(np.int32)
    frac = cpos - lpos

    lpos_clipped = np.clip(lpos - 1, 0, sample_lateral - 1)
    hpos_clipped = np.clip(hpos - 1, 0, sample_lateral - 1)

    val = vals[lpos_clipped] * (1.0 - frac) + vals[hpos_clipped] * frac
    val[np.ceil(cpos) > sample_lateral] = 0.0
    return val


def write_otf_central(otf: dict) -> np.ndarray:
    NPixel = otf['NPixel']
    cycles_per_micron = otf['cyclesPerMicron']
    Fc = otf['Fc']

    x = np.arange(1, NPixel + 1)
    y = np.arange(1, NPixel + 1)
    X, Y = np.meshgrid(x, y)
    cnt = NPixel // 2 + 1

    rad = np.sqrt((Y - cnt) ** 2 + (X - cnt) ** 2)
    cyc = rad * cycles_per_micron

    mask = cyc > Fc
    cyc_safe = cyc.copy()
    cyc_safe[mask] = 0.0

    val = get_otf_val(otf, cyc_safe)
    val[mask] = 0.0
    return val


def apply_detector_noise(photons: np.ndarray,
                         read_noise_std: float = 10.0,
                         dark_current: float = 5.0) -> np.ndarray:

    photons = np.asarray(photons, dtype=np.float64)
    noisy = np.random.poisson(np.maximum(photons, 0)).astype(np.float64)

    if dark_current > 0:
        dark = np.random.poisson(dark_current, size=photons.shape).astype(np.float64)
        noisy += dark
    if read_noise_std > 0:
        noisy += np.random.normal(scale=read_noise_std, size=photons.shape)

    return np.maximum(noisy, 0.0)


def main():
    # ------------------------- 光学系统参数 ----------------------------------
    NPixel = 502
    micron_per_pixel = 0.0626          # µm/pixel
    cycles_per_micron = 1.0 / (NPixel * micron_per_pixel)

    # OTF 参数
    a = 1.0
    NA = 1.41
    lamda = 525.0                      # nm
    Fc = 1000.0 / (lamda / NA / 2.0)
    sample_lateral = int(np.ceil(Fc / cycles_per_micron)) + 1

    # 结构光照明
    k0 = 0.8 * Fc
    Angle = [0.0, np.pi / 3.0, 2.0 * np.pi / 3.0]
    Phase0 = [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]
    M = 0.8

    # --------------------------- 曝光控制（防止过曝/欠曝）----------------------
    auto_exposure = True               # 自动计算 emission_scale（推荐）
    target_fraction = 0.4              # 信号峰值占满阱容量的比例（auto 时有效）
    manual_emission_scale = 200.0      # 手动指定 photon scale（auto=False 时使用）

    # --------------------------- 探测器与噪声 ---------------------------------
    add_noise = False
    max_photons_base = 5000.0           # 基准满阱容量 (e-)
    noise_read_std = 10.0               # 读出噪声 (e-)
    noise_dark_current_base = 5.0       # 基准暗电流 (e-/pixel)

    # --------------------------- 离焦背景 ------------------------------------
    enable_defocus_bg = False
    defocus_bg_photons_base = 100.0     # 基准背景光子/pixel

    # --------------------------- 曝光时间 ------------------------------------
    exposure_factor = 1.0               # 曝光倍数（1.0 = 关闭）


    input_image_path = 'GT.tif'
    SavePath = '...'
    base_subdir = 'GT'

    # Output subfolder
    folder_name = base_subdir
    if enable_defocus_bg:
        folder_name += f"_defocusBg{defocus_bg_photons_base:.0f}"
    if exposure_factor != 1.0:
        folder_name += f"_expT{exposure_factor:.1f}"
    if add_noise:
        folder_name += (f"_noise_photons{max_photons_base:.0f}"
                        f"_read{noise_read_std:.0f}"
                        f"_dark{noise_dark_current_base:.0f}")
    else:
        folder_name += "_noiseFree"
    save_path = os.path.join(SavePath, folder_name)

    # Spatial grid
    Xmax = micron_per_pixel * NPixel
    x = np.linspace(-Xmax / 2.0, Xmax / 2.0, NPixel)
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    # OTF
    v_vals = np.arange(sample_lateral) / sample_lateral
    vals = np.array([val_ideal_otf(v) * (a ** v) for v in v_vals], dtype=np.float64)
    otf_dict = {
        'vals': vals,
        'cyclesPerMicron': cycles_per_micron,
        'Fc': Fc,
        'NPixel': NPixel
    }
    otf2d = write_otf_central(otf_dict)

    # --------------------- 加载样本图像 ---------------------------------------
    Sample0 = imread(input_image_path)
    if Sample0.ndim == 3:
        Sample0 = np.mean(Sample0, axis=2)
    Sample = Sample0.astype(np.float64)          # 原始强度（任意单位）

    # --------------------- 自动曝光控制 ---------------------------------------
    if auto_exposure:
        # 计算典型信号的光子数：取第一帧（angle=0, phase=0）的峰值
        # 照明图案：1 + M*cos(kx*X + ky*Y)，平均强度 ≈ 1
        # OTF 滤波后峰值近似等于原始图像峰值，这里直接使用原始图像最大值
        # 作为安全估计，考虑可能因干涉出现更大值，再乘以 (1+M) ≈ 1.8
        peak_sample = np.max(Sample)
        safe_peak = peak_sample * (1.0 + M)  # 最亮可能值
        # 所需 photon scale = target_fraction * max_photons / safe_peak
        desired_photons = target_fraction * max_photons_base * exposure_factor
        emission_scale = desired_photons / safe_peak if safe_peak > 0 else 1.0
        print(f"自动曝光：样本峰值={peak_sample:.1f}, "
              f"计算得 emission_scale = {emission_scale:.3f}")
    else:
        emission_scale = manual_emission_scale

    # --------------------- 生成 GT 图像 ---------------------------------------
    Size1 = 2 * NPixel
    fftGT = np.fft.fftshift(np.fft.fft2(Sample))
    Temp = np.zeros((Size1, Size1), dtype=np.complex128)
    r_start = NPixel // 2
    c_start = NPixel // 2
    Temp[r_start:r_start + NPixel, c_start:c_start + NPixel] = fftGT
    Temp = np.abs(np.fft.ifft2(Temp))
    GT = 4.0 * Temp[:Size1, :Size1]

    # 创建输出目录
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    # 保存 GT（应用曝光因子，映射为 16 位）
    max_photons = max_photons_base * exposure_factor
    GT_photons = GT * emission_scale * exposure_factor
    GT_16bit = np.clip(GT_photons / max_photons * 65535, 0, 65535).astype(np.uint16)
    imwrite(os.path.join(save_path, 'GT.tif'), GT_16bit)

    # ------------- 采集循环 -------------------------------------------------
    Nang = len(Angle)
    Nphi = len(Phase0)
    dark_current_scaled = noise_dark_current_base * exposure_factor
    bg_photons_scaled = defocus_bg_photons_base * exposure_factor

    for nAng, angle in enumerate(Angle):
        kx = k0 * np.cos(angle)
        ky = k0 * np.sin(angle)

        for nPhi, phase in enumerate(Phase0):
            idx = nAng * Nphi + nPhi

            # 照明图案
            pattern = 1.0 + M * np.cos(2.0 * np.pi * (kx * X + ky * Y) + phase)

            # 荧光发射 → 光子数
            emission = pattern * Sample
            photons = emission * emission_scale * exposure_factor

            # 光学成像：OTF 滤波
            F_emission = np.fft.fftshift(np.fft.fft2(emission)) * otf2d
            image_arb = np.real(np.fft.ifft2(np.fft.ifftshift(F_emission)))
            # 将滤波后的图像重新转换为光子数（OTF 影响幅度）
            photons_filtered = image_arb * emission_scale * exposure_factor

            # 离焦背景
            if enable_defocus_bg:
                photons_filtered += bg_photons_scaled

            # 探测器噪声
            if add_noise:
                photons_filtered = apply_detector_noise(
                    photons_filtered,
                    read_noise_std=noise_read_std,
                    dark_current=dark_current_scaled
                )

            # 16 位映射与保存
            img_16bit = np.clip(photons_filtered / max_photons * 65535, 0, 65535).astype(np.uint16)
            imwrite(os.path.join(save_path, f'{idx}.tif'), img_16bit)

    print(f"仿真完成。输出文件夹：{save_path}")


if __name__ == '__main__':
    main()