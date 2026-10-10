import cupy as cp
import numpy as np
from typing import Union, Tuple, Optional


def otf2psf(otf: cp.ndarray, outsize: Optional[Union[int, Tuple[int, ...]]] = None) -> cp.ndarray:
    otf, otfSize, outSize = _parse_inputs_gpu(otf, outsize)
    if cp.all(otf == 0):
        return cp.zeros(outSize, dtype=cp.float64)
    psf = cp.fft.ifftn(otf)
    n_elem = cp.prod(cp.array(otfSize))
    n_ops = 0
    for k in range(len(otfSize)):
        n_ffts = n_elem / otfSize[k]
        n_ops += otfSize[k] * cp.log2(otfSize[k]) * n_ffts
    max_abs_psf = cp.max(cp.abs(psf))
    if max_abs_psf > 0:
        max_imag_ratio = cp.max(cp.abs(psf.imag)) / max_abs_psf
        if max_imag_ratio <= n_ops * cp.finfo(float).eps:
            psf = psf.real
    shift_amounts = tuple([int(cp.floor(s / 2)) for s in outSize])
    psf = cp.roll(psf, shift_amounts, axis=range(len(outSize)))
    slices = tuple([slice(0, s) for s in outSize])
    psf = psf[slices]
    return psf


def _parse_inputs_gpu(otf, outsize):
    if not isinstance(otf, cp.ndarray):
        try:
            otf = cp.asarray(otf)
        except:
            raise ValueError("OTF must be a numeric array")

    if not cp.all(cp.isfinite(otf)):
        raise ValueError("OTF contains non-finite values")

    if cp.iscomplexobj(otf):
        otf = otf.astype(cp.complex128)
    else:
        otf = otf.astype(cp.float64)
    otfSize = otf.shape

    if outsize is None:
        outSize = otfSize
    else:
        if isinstance(outsize, (int, np.integer)):
            if len(otfSize) == 2:
                outSize = (outsize, outsize)
            else:
                outSize = (outsize,) * len(otfSize)
        else:
            outSize = tuple(outsize)

        if any(s < 0 for s in outSize):
            raise ValueError("outsize must be non-negative real numbers")

    if otf.size == 0:
        return otf, otfSize, outSize
    padded_otfSize, padded_outSize = _padlength(otfSize, outSize)
    for i, (out_dim, otf_dim) in enumerate(zip(padded_outSize, padded_otfSize)):
        if out_dim > otf_dim:
            raise ValueError(f"Dimension {i + 1} of outsize ({out_dim}) is larger than the corresponding dimension of OTF ({otf_dim})")
    return otf, otfSize, padded_outSize


def _padlength(*args):
    num_dims = max(len(arg) for arg in args)
    result = []
    for arg in args:
        if isinstance(arg, (int, float)):
            arg = (arg,)
        padded = tuple(arg) + (1,) * (num_dims - len(arg))
        result.append(padded)

    return tuple(result)

def psf2otf(psf: np.ndarray, outsize: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    if outsize is None:
        outsize = psf.shape
    else:
        if isinstance(outsize, (int, np.integer)):
            outsize = (outsize,)
        else:
            outsize = tuple(outsize)
    if len(outsize) < psf.ndim:
        outsize = outsize + (1,) * (psf.ndim - len(outsize))
    pad_sizes = []
    for psf_dim, out_dim in zip(psf.shape, outsize):
        if out_dim >= psf_dim:
            pad_before = 0
            pad_after = out_dim - psf_dim
        else:
            raise ValueError("outsize must not be smaller than the corresponding dimension of psf")
        pad_sizes.append((pad_before, pad_after))
    padded_psf = cp.pad(psf, pad_sizes, mode='constant', constant_values=0)
    shift_amounts = tuple([-int(cp.floor(s / 2)) for s in padded_psf.shape])
    shifted_psf = cp.roll(padded_psf, shift_amounts, axis=range(padded_psf.ndim))
    otf = cp.fft.fftn(shifted_psf)
    return otf


import cupy as cp
import numpy as np

class SimOtfProvider:
    def __init__(self, param, NA, wavelength, a=1):
        self.na = NA
        self.lambda_ = wavelength         # nm
        self.cutoff = 1000 / (0.5 * self.lambda_ / self.na)
        self.img_size = param["imgSize"]    # NPixel
        self.cycles_per_micron = param["cyclesPerMicron"]
        self.sample_lateral = int(cp.ceil(self.cutoff / self.cycles_per_micron) + 1)
        self.estimate_a_value = a
        self.maxBand = 2
        self.att_strength = param["attStrength"]
        self.att_fwhm = 1.0   # 1.0
        self.useAttenuation = 1
        self._from_estimate()
        self.otf = cp.zeros((self.img_size, self.img_size), dtype=cp.float64)
        self.otfatt = cp.zeros_like(self.otf)
        self.onlyatt = cp.zeros_like(self.otf)
        self.otf = self.write_otf_vector(self.otf, kx=0, ky=0)
        self.onlyatt = self.get_only_att(kx=0, ky=0)
        self.otfatt = self.otf * self.onlyatt

    def val_ideal_otf(self, dist):
        if not isinstance(dist, cp.ndarray):
            dist = cp.asarray(dist, dtype=cp.float64)
        val = cp.zeros_like(dist, dtype=cp.float64)
        mask = (dist >= 0) & (dist <= 1)
        d = dist[mask]
        val[mask] = (1.0 / cp.pi) * (2 * cp.arccos(d) - cp.sin(2 * cp.arccos(d)))
        return val

    def val_attenuation(self, dist, strength, fwhm):
        dist = cp.asarray(dist, dtype=cp.float64)
        return 1 - strength * cp.exp(-(dist ** 2) / (0.5 * fwhm) ** 2)

    def _from_estimate(self):
        n = self.sample_lateral
        vals1 = cp.zeros(n, dtype=cp.float64)
        vals_att = cp.zeros(n, dtype=cp.float64)
        vals_only_att = cp.zeros(n, dtype=cp.float64)

        for i in range(n):
            v = abs(i) / n
            vals1[i] = self.val_ideal_otf(cp.asarray([v], dtype=cp.float64))[0] * (self.estimate_a_value ** v)

        for i in range(n):
            dist = abs(i) * self.cycles_per_micron
            vals_only_att[i] = self.val_attenuation(cp.asarray([dist], dtype=cp.float64), self.att_strength, self.att_fwhm)[0]
            vals_att[i] = vals1[i] * vals_only_att[i]

        self.vals = vals1
        self.vals_att = vals_att
        self.vals_only_att = vals_only_att

    def get_only_att(self, kx=0, ky=0):
        h = w = self.img_size
        cnt_x, cnt_y = w // 2, h // 2
        x, y = cp.meshgrid(cp.arange(w, dtype=cp.float64), cp.arange(h, dtype=cp.float64))
        rad = cp.hypot(y - (ky + cnt_y), x - (kx + cnt_x))
        cycl = rad * self.cycles_per_micron
        return self.val_attenuation(cycl, self.att_strength, self.att_fwhm)

    def write_otf_vector(self, vec, kx=0, ky=0):
        return self.otf_to_vector(vec, kx, ky, use_att=0, write=1)

    def otf_to_vector(self, vec, kx=0, ky=0, use_att=0, write=1):
        h, w = vec.shape
        cnt_x, cnt_y = w // 2, h // 2
        x, y = cp.meshgrid(cp.arange(w, dtype=cp.float64), cp.arange(h, dtype=cp.float64))
        rad = cp.hypot(y - (ky + cnt_y), x - (kx + cnt_x))
        cycl = rad * self.cycles_per_micron
        mask = cycl > self.cutoff
        cycl = cp.where(mask, 0, cycl)
        val = self.get_otf_val(cycl, use_att)
        if write == 0:
            vec = vec * val
        else:
            vec = val
        vec[mask] = 0
        return vec

    def get_otf_val(self, cycl, att):
        pos = cycl / self.cycles_per_micron
        lpos = cp.floor(pos).astype(cp.int32)
        hpos = cp.ceil(pos).astype(cp.int32)
        f = pos - lpos
        lpos = cp.clip(lpos, 0, self.sample_lateral - 1)
        hpos = cp.clip(hpos, 0, self.sample_lateral - 1)

        if att == 1:
            vals_array = self.vals_att
        else:
            vals_array = self.vals

        retl = vals_array[lpos] * (1 - f)
        reth = vals_array[hpos] * f
        val = retl + reth
        mask = cp.ceil(pos) > self.sample_lateral
        val[mask] = 0
        return val


class ParameterSet:
    def __init__(self, Iraw, Pixelsize, NA, lambda_):
        # 确保 Iraw 为 cupy float64
        if not isinstance(Iraw, cp.ndarray):
            Iraw = cp.asarray(Iraw, dtype=cp.float64)
        else:
            Iraw = Iraw.astype(cp.float64)

        NPixel = Iraw.shape[1]
        self.imgSize = NPixel
        self.micronsPerPixel = Pixelsize * 1e-3   # 假设 Pixelsize 单位为 nm
        self.cyclesPerMicron = 1 / (NPixel * self.micronsPerPixel)
        self.NA = NA
        self.lambda_ = lambda_   # nm
        self.cutoff = 1000 / (0.5 * self.lambda_ / self.NA)
        self.sampleLateral = int(np.ceil(self.cutoff / self.cyclesPerMicron) + 1)
        self.nrBands = 2
        self.phaOff = 0
        self.fac = np.ones(self.nrBands, dtype=np.float64)
        self.attStrength = 0.9
        self.OtfProvider = SimOtfProvider(self.__dict__, self.NA, self.lambda_, a=0.3)
        self.OTF = self.OtfProvider.otf
        psf_temp = otf2psf(self.OTF)
        self.psf = cp.abs(psf_temp)
        psf2_temp = otf2psf(self.OTF, outsize=32)
        self.psf2 = cp.abs(psf2_temp)
        self.psf2 = cp.asnumpy(self.psf2).astype(np.float64)
        self.psf = cp.asnumpy(self.psf).astype(np.float64)

        # ========== 新增：添加 ParameterEstimationCOR 需要的属性 ==========
        self.nrFrames = Iraw.shape[0]
        self.nrDirs = 3          # 固定为 3 个方向
        self.nrPhases = self.nrFrames // self.nrDirs  # 应为 3
        self.Dir = None          # 将在后续重建中填充