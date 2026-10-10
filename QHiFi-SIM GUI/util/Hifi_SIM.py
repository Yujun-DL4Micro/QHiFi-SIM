import numpy as np
import cupy as cp
import cupy.fft as fft

class HifiReconstruction:
    def __init__(self, IIrawFFT, param, par):
        self.IIrawFFT = IIrawFFT.astype(cp.complex128)
        self.param = param
        self.par = par

    def reconstruct(self):
        param = self.param
        Dir = self.par
        IIrawFFT = self.IIrawFFT

        Nch, Ny, Nx = IIrawFFT.shape
        assert Nch == 9, "expect 9 frames in first dim"

        Fine_frequency = cp.zeros((2*Ny, 2*Nx), dtype=cp.complex128)
        Shifted = cp.zeros((2*Ny, 2*Nx, 3), dtype=cp.complex128)

        for I in range(3):
            dir_info = Dir[I]
            phaOff = dir_info['phaOff']
            px = float(dir_info['px'])
            py = float(dir_info['py'])
            modul = dir_info['modul']

            fac = [1.0] * param.nrBands
            fac[1] = float(modul)

            frames = IIrawFFT[I*3 : (I+1)*3, :, :]
            Separate = self.separateBands(frames, phaOff, param.nrBands, fac)

            Shifted[:] = 0
            Shifted[:, :, 0] = self.placeFreq(Separate[:, :, 0])

            import os
            import tifffile as tif
            import numpy as np
            debug_dir = 'D:/1-科研工作相关/00_SIM强度校正/00_开源代码'
            aaa = Shifted[:, :, 0]
            os.makedirs(debug_dir, exist_ok=True)
            tif.imwrite(os.path.join(debug_dir, 'aaa.tif'),
                        cp.asnumpy(cp.log1p(cp.abs(aaa))).astype(np.float32))

            Shifted[:, :, 1] = self.placeFreq(Separate[:, :, 1])
            Shifted[:, :, 2] = self.placeFreq(Separate[:, :, 2])

            Shifted[:, :, 1] = self.NfourierShift(Shifted[:, :, 1], -px, -py)
            Shifted[:, :, 2] = self.NfourierShift(Shifted[:, :, 2], px, py)

            Shifted[:, :, 0] = self.otfToVector(Shifted[:, :, 0], param.OtfProvider, 0, 0, 1, 0)
            Shifted[:, :, 1] = self.otfToVector(Shifted[:, :, 1], param.OtfProvider, -px, -py, 1, 0)
            Shifted[:, :, 2] = self.otfToVector(Shifted[:, :, 2], param.OtfProvider, px, py, 1, 0)
            Fine_frequency += Shifted[:, :, 0] + Shifted[:, :, 1] + Shifted[:, :, 2]

        w1 = 1.2
        w2 = 0.1
        K0_vals = cp.array([float(d['K0']) if 'K0' in d else float(cp.sqrt(d['px']**2 + d['py']**2))
                            for d in Dir], dtype=cp.float64)
        K = int(cp.max(cp.ceil(K0_vals)))
        cutoff = np.floor(1 * K) / param.sampleLateral + 1.0

        otfHiFi = cp.zeros((2*Ny, 2*Nx), dtype=cp.float64)
        otfHiFi = self.writeApoVector(otfHiFi, param.OtfProvider, cutoff)  # Ideal OTF
        Mask = cp.zeros_like(otfHiFi)
        Mask[otfHiFi != 0] = 1.0

        # HiFi - SIM
        # Step
        wFilter1 = self.WienerFilterW1_2D(param)
        otfHiFi = cp.asarray(otfHiFi, dtype=cp.float64)
        Wk1 = otfHiFi / (wFilter1.wDenom + (w1**2))

        wFilter2 = self.WienerFilterW2_2D(param)
        ApoFWHM = 0.5 * (cutoff - 1)
        ApoFWHM = min(0.5, self.matlab_round(ApoFWHM * 100) / 100)
        apo = self.apodize_gauss([2*Ny, 2*Nx], {'rad': ApoFWHM})
        Wk2 = apo / (wFilter2.wDenom + (w2**2))

        fftInitialHiFi = Fine_frequency * Wk1 * Mask
        fftHiFi = cp.real(fft.ifft2(fft.fftshift(fftInitialHiFi * Wk2 * Mask)))

        HiFi = fftHiFi
        HiFi = cp.where(HiFi < 0, 0, HiFi)
        maxv = cp.max(HiFi)
        if maxv == 0:
            return HiFi
        HiFi = 255.0 * HiFi / maxv
        return HiFi

    def separateBands(self, IrawFFT, phaOff, bands, fac):
        phaPerBand = bands * 2 - 1
        phases = cp.array([(2 * cp.pi * p / phaPerBand + phaOff) for p in range(phaPerBand)], dtype=cp.float64)

        fac = list(fac)
        for i in range(1, bands):
            fac[i] = fac[i] * 0.5

        comp = cp.zeros(phaPerBand, dtype=cp.float64)
        compfac = cp.ones(phaPerBand, dtype=cp.float64)
        compfac[0] = fac[0]
        for i in range(1, bands):
            comp[2 * i - 1] = i
            comp[2 * i] = -i
            compfac[2 * i - 1] = fac[i]
            compfac[2 * i] = fac[i]

        W = cp.exp(1j * cp.outer(phases, comp)) * compfac
        W = W.astype(cp.complex128)

        Nframes, H, Wp = IrawFFT.shape
        IrawFFT=IrawFFT.transpose(1, 2, 0)
        # Iraw_flat = IrawFFT.reshape(H * Wp, Nframes).astype(cp.complex128)
        Iraw_flat = IrawFFT.reshape(H * Wp, Nframes, order='F').astype(cp.complex128)
        # S = Iraw_flat @ cp.linalg.pinv(W).T
        S = Iraw_flat @ cp.linalg.pinv(W).conj().T
        Sk = S.reshape(H, Wp, phaPerBand, order='F')
        return Sk

    def otfToVector(self, vec, otf, kx=0.0, ky=0.0, useAtt=1, write=0):
        h, w = vec.shape
        cnt_y, cnt_x = h // 2.0, w // 2.0
        kx = kx + cnt_x
        ky = ky + cnt_y
        x, y = cp.meshgrid(cp.arange(w, dtype=cp.float64), cp.arange(h, dtype=cp.float64))
        rad = cp.hypot(y - ky, x - kx)
        cycl = rad * otf.cycles_per_micron
        mask = cycl > otf.cutoff
        cycl[mask] = 0
        val = self.getOtfVal(otf, cycl, useAtt)
        if write == 0:
            vec = vec * val
        else:
            vec = val
        vec[mask] = 0
        return vec

    def getOtfVal(self, otf, cycl, att):
        pos = cycl / otf.cycles_per_micron
        cpos = pos + 1.0
        lpos = cp.floor(cpos).astype(cp.int32) - 1
        hpos = cp.ceil(cpos).astype(cp.int32) - 1
        f = cpos - cp.floor(cpos)

        if att == 1:
            vals = cp.asarray(otf.vals_att, dtype=cp.float64)
        else:
            vals = cp.asarray(otf.vals, dtype=cp.float64)

        retl = vals[lpos] * (1.0 - f)
        reth = vals[hpos] * f
        val = retl + reth
        mask = cp.ceil(cpos) > otf.sample_lateral
        val[mask] = 0
        return val

    def placeFreq(self, in_arr):
        h, w = in_arr.shape
        out = cp.zeros((2*h, 2*w), dtype=in_arr.dtype)
        out[h//2:h+h//2, w//2:w+w//2] = in_arr
        return out

    def NfourierShift(self, inv, kx, ky):
        inv = fft.ifft2(fft.fftshift(inv))
        h, w = inv.shape
        x, y = cp.meshgrid(cp.arange(w, dtype=cp.float64) / w,
                           cp.arange(h, dtype=cp.float64) / h)
        outv = inv * cp.exp(2j * cp.pi * (ky * y + kx * x))
        outv = fft.fftshift(fft.fft2(outv))

        return outv

    def writeApoVector(self, vec, otf, cutoff):
        h, w = vec.shape
        cnt_y, cnt_x = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]
        rad = np.hypot(y - cnt_y, x - cnt_x)
        cycl = rad * otf.cycles_per_micron
        frac = cycl / (otf.cutoff * cutoff)
        valIdealotf = np.zeros_like(frac, dtype=np.float64)
        mask = (frac >= 0) & (frac <= 1)
        frac_valid = frac[mask]
        if np.any(mask):
            acos_val = np.arccos(frac_valid)
            valIdealotf[mask] = (1.0 / np.pi) * (2 * acos_val - np.sin(2 * acos_val))
        return cp.asarray(valIdealotf, dtype=cp.float64)

    def WienerFilterW1_2D(self, sp):
        class Filter:
            def __init__(self, wDenom):
                self.wDenom = wDenom
        wDenom = self.updateCache1(sp)
        return Filter(wDenom)

    def WienerFilterW2_2D(self, sp):
        class Filter:
            def __init__(self, wDenom):
                self.wDenom = wDenom
        wDenom = self.updateCache2(sp)
        return Filter(wDenom)

    def updateCache1(self, sp):
        Temp = sp.OtfProvider.otf
        h, w = Temp.shape[:2]
        wDenom = cp.zeros((2*h, 2*w), dtype=cp.float64)
        for d in range(3):
            for b in range(2):
                wDenom = self.addWienerDenominatorWiener_2D1(wDenom, sp, d, b)
        return wDenom

    def updateCache2(self, sp):
        Temp = sp.OtfProvider.otf
        h, w = Temp.shape[:2]
        wDenom = cp.zeros((2*h, 2*w), dtype=cp.float64)
        for d in range(3):
            for b in range(2):
                wDenom = self.addWienerDenominatorWiener_2D2(wDenom, sp, d, b)
        return wDenom

    def addWienerDenominatorWiener_2D1(self, wd, sp, d, b):
        h, w = wd.shape
        dir_info = sp.Dir[d]
        cyclMicron = sp.cyclesPerMicron
        cnt_y, cnt_x = h // 2, w // 2
        x, y = cp.meshgrid(cp.arange(w, dtype=cp.float64),
                           cp.arange(h, dtype=cp.float64))
        rad1 = cp.hypot(x - cnt_x - b * dir_info['px'], y - cnt_y - b * dir_info['py']) * cyclMicron
        rad2 = cp.hypot(x - cnt_x + b * dir_info['px'], y - cnt_y + b * dir_info['py']) * cyclMicron
        otfVal1 = cp.abs(self.getOtfVval(sp.OtfProvider, rad1)) ** 2
        otfVal2 = cp.abs(self.getOtfVval(sp.OtfProvider, rad2)) ** 2
        if b == 0:
            otfVal1 /= 2
            otfVal2 /= 2
        else:
            att1 = self.valAttenuation(rad1, sp.OtfProvider.att_strength / 1.0, 1.0 * sp.OtfProvider.att_fwhm)
            att2 = self.valAttenuation(rad2, sp.OtfProvider.att_strength / 1.0, 1.0 * sp.OtfProvider.att_fwhm)
            otfVal1 *= att1 / 1.0
            otfVal2 *= att2 / 1.0
        wd = wd + 1.0*otfVal1 + 1.0*otfVal2
        return wd

    def addWienerDenominatorWiener_2D2(self, wd, sp, d, b):
        h, w = wd.shape
        dir_info = sp.Dir[d]
        cyclMicron = sp.cyclesPerMicron
        cnt_y, cnt_x = h // 2, w // 2
        x, y = cp.meshgrid(cp.arange(w, dtype=cp.float64),
                           cp.arange(h, dtype=cp.float64))
        rad1 = cp.hypot(x - cnt_x - b * dir_info['px'], y - cnt_y - b * dir_info['py']) * cyclMicron
        rad2 = cp.hypot(x - cnt_x + b * dir_info['px'], y - cnt_y + b * dir_info['py']) * cyclMicron
        otfVal1 = cp.abs(self.getOtfVval(sp.OtfProvider, rad1)) ** 2
        otfVal2 = cp.abs(self.getOtfVval(sp.OtfProvider, rad2)) ** 2
        if b == 0:
            if sp.OtfProvider.att_fwhm == 0:
                otfVal1 /= 2
                otfVal2 /= 2
            else:
                att1 = self.valAttenuation(rad1, sp.OtfProvider.att_strength / 1.0, 1.0 * sp.OtfProvider.att_fwhm)
                att2 = self.valAttenuation(rad2, sp.OtfProvider.att_strength / 1.0, 1.0 * sp.OtfProvider.att_fwhm)
                otfVal1 *= att1 / 2.0
                otfVal2 *= att2 / 2.0
        else:
            att1 = self.valAttenuation(rad1, sp.OtfProvider.att_strength / 1.0, 1.0 * sp.OtfProvider.att_fwhm)
            att2 = self.valAttenuation(rad2, sp.OtfProvider.att_strength / 1.0, 1.0 * sp.OtfProvider.att_fwhm)
            otfVal1 *= att1 / 1.0
            otfVal2 *= att2 / 1.0
        wd = wd + otfVal1 + otfVal2
        return wd

    @staticmethod
    def getOtfVval(otfProvider, cycl):
        mask = cycl > otfProvider.cutoff
        cycl_safe = cp.where(mask, 0, cycl)
        pos = cycl_safe / otfProvider.cycles_per_micron
        cpos = pos + 1.0
        lpos = cp.floor(cpos).astype(cp.int32) - 1
        hpos = cp.ceil(cpos).astype(cp.int32) - 1
        f = cpos - cp.floor(cpos)
        vals = cp.asarray(otfProvider.vals, dtype=cp.float64)
        retl = vals[lpos] * (1.0 - f)
        reth = vals[hpos] * f
        val = retl + reth
        val[mask] = 0
        return val

    @staticmethod
    def valAttenuation(dist, strength, fwhm):
        return 1 - strength * cp.exp(- (dist ** 2) / (0.5 * fwhm) ** 2)

    @staticmethod
    def apodize_gauss(siz, params):
        rad = params.get('rad', 0.5)
        offset = params.get('offset', [0, 0])
        cnt = cp.ceil((cp.array(siz) + 1) / 2) + cp.array(offset)
        x, y = cp.meshgrid(cp.arange(1, siz[1] + 1, dtype=cp.float64),
                           cp.arange(1, siz[0] + 1, dtype=cp.float64))
        r = cp.hypot((x - cnt[1]) * 2 / siz[1], (y - cnt[0]) * 2 / siz[0])
        A = cp.exp(-0.5 * (r / rad * cp.sqrt(2 * cp.log(2))) ** 2)
        return A

    @staticmethod
    def matlab_round(v):
        v = float(v)
        return int(cp.floor(v + 0.5)) if v >= 0 else int(cp.ceil(v - 0.5))