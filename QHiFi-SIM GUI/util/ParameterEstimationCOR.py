import cupy as cp
import cupy.fft as fft

class ParameterEstimationCOR:
    def __init__(self, IIrawFFT, param):
        self.IIrawFFT = IIrawFFT.astype(cp.complex128)
        self.param = param
        self.Dir = [{} for _ in range(param.nrDirs)]
        self.K0 = cp.zeros(param.nrDirs, dtype=cp.float64)

    def run(self):
        IIrawFFT = self.IIrawFFT
        param = self.param
        NPixel = IIrawFFT.shape[1]

        x, y = cp.meshgrid(cp.arange(NPixel, dtype=cp.float64),
                           cp.arange(NPixel, dtype=cp.float64))
        self.Center = cp.array([NPixel // 2, NPixel // 2], dtype=cp.float64)  # 0-based 中心
        self.R = cp.sqrt((y - self.Center[0])**2 + (x - self.Center[1])**2)

        cutoff_pix = param.cutoff / param.cyclesPerMicron
        Mask_2NA = (self.R <= 1.0 * (cutoff_pix + 1)).astype(cp.float64)
        self.NotchFilter0 = self.get_otfAtt(NPixel, param.cyclesPerMicron, 0.5 * param.cutoff).astype(cp.float64)
        self.NotchFilter = self.NotchFilter0 * Mask_2NA

        Mask_2NA2 = (self.R <= 1.10 * (cutoff_pix + 1)).astype(cp.float64)
        self.NotchFilter2 = self.NotchFilter0 * Mask_2NA2

        CrossCorrelation = []
        SeparateAll = []
        k0_init = []

        for I in range(param.nrDirs):
            frames = IIrawFFT[I * param.nrPhases : (I + 1) * param.nrPhases, :, :]
            separateII = self.separateBands(frames, 0.0, param.nrBands, [1]*param.nrBands)  # 0,1，-1
            SeparateAll.append(separateII)

            fb = 2
            c0 = separateII[:, :, 0].astype(cp.complex128)
            c_high = separateII[:, :, fb-1].astype(cp.complex128)

            c0 = c0 / cp.max(cp.abs(c0))
            c_high = c_high / cp.max(cp.abs(c_high))
            c0 = c0 * self.NotchFilter
            c_high = c_high * self.NotchFilter

            c0_sp = fft.fftshift(fft.fft2(c0))
            c_high_sp = fft.fftshift(fft.fft2(c_high))

            corr_sp = c_high_sp * cp.conj(c0_sp)
            corr_sp = corr_sp / cp.max(cp.abs(corr_sp))
            vec = fft.fftshift(fft.ifft2(fft.fftshift(corr_sp)))
            CrossCorrelation.append(vec)
            temp = vec * self.NotchFilter2
            temp = cp.log1p(cp.abs(temp))
            temp = temp / cp.max(temp)
            yPos, xPos = cp.unravel_index(cp.argmax(temp), temp.shape)
            k0_init.append(cp.sqrt((xPos - self.Center[1]) ** 2 + (yPos - self.Center[0]) ** 2))
        k0_init = cp.array(k0_init)

        Flag = 0
        if param.nrDirs > 2:
            if cp.max(k0_init) - cp.min(k0_init) > 8:
                Flag = 1
                Kobject = cp.min(k0_init)
                Mask1 = (self.R >= (Kobject + 1)).astype(cp.float64)
                Mask2 = (self.R <= (Kobject - 1)).astype(cp.float64)

        for I in range(param.nrDirs):
            vec = CrossCorrelation[I]
            if Flag == 1:
                vec = vec * (1 - Mask1) * (1 - Mask2)
            temp = vec * self.NotchFilter2
            temp = cp.log1p(cp.abs(temp))
            temp = temp / cp.max(temp)

            yPos, xPos = cp.unravel_index(cp.argmax(temp), temp.shape)
            cnt = self.Center
            kx_init = xPos - cnt[1]
            ky_init = yPos - cnt[0]

            separateII = SeparateAll[I]
            fb = 2
            lb = 2
            bn1 = 2

            cntrl = cp.zeros((10, 30))
            band0 = separateII[:, :, 0] / cp.max(cp.abs(separateII[:, :, 0]))
            band_high = separateII[:, :, fb-1] / cp.max(cp.abs(separateII[:, :, fb-1]))

            peak, _ = self.fitPeak(band0, band_high, 1, bn1, self.param.OtfProvider,
                                   -kx_init, -ky_init, 0.15, 2.5, cntrl)

            p1 = self.getPeak(separateII[:, :, 0], separateII[:, :, lb-1],
                              1, lb, self.param.OtfProvider, peak.kx, peak.ky, 0.15)
            self.Dir[I]['px'] = -peak.kx
            self.Dir[I]['py'] = -peak.ky
            self.Dir[I]['phaOff'] = -cp.angle(p1)
            modul = float(cp.abs(p1))
            modul = min(modul, 1.0)
            if modul < 0.35: modul = 0.7
            self.Dir[I]['modul'] = modul
            self.K0[I] = float(cp.sqrt(peak.kx**2 + peak.ky**2))

        return self.Dir, self.K0

    # ---------------- 辅助函数 ----------------
    def get_otfAtt(self, imgSize, cyclesPerMicron, attfwhm, kx=0.0, ky=0.0):
        imgSize = int(imgSize)
        y, x = cp.meshgrid(cp.arange(imgSize, dtype=cp.float64),
                           cp.arange(imgSize, dtype=cp.float64))
        cnt = imgSize / 2.0
        kx = float(kx) + cnt
        ky = float(ky) + cnt
        rad = cp.hypot(y - ky, x - kx)
        cycl = rad * cp.float64(cyclesPerMicron)
        attfwhm = cp.float64(attfwhm)
        return 1 - cp.exp(-(cycl ** 4) / (2 * attfwhm ** 4))

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

    def fourierShift(self, vec, kx, ky):
        H, W = vec.shape
        x, y = cp.meshgrid(cp.arange(W, dtype=cp.float64) / W,
                           cp.arange(H, dtype=cp.float64) / H)
        return vec * cp.exp(2j * cp.pi * (ky * y + kx * x))

    def commonRegion(self, band0, band1, bn0, bn1, otf, kx, ky, dist, weightLimit, divideByOtf=True):

        h, w = band0.shape
        cnt = [h / 2.0, w / 2.0]
        weight0 = cp.zeros((h, w), dtype=cp.complex128)
        weight1 = cp.zeros((h, w), dtype=cp.complex128)
        wt0 = cp.zeros((h, w), dtype=cp.complex128)
        wt1 = cp.zeros((h, w), dtype=cp.complex128)

        weight0 = self.otfToVector(weight0, otf, 0, 0, 0, 1)
        weight1 = self.otfToVector(weight1, otf, 0, 0, 0, 1)
        wt0 = self.otfToVector(wt0, otf, kx, ky, 0, 1)
        wt1 = self.otfToVector(wt1, otf, -kx, -ky, 0, 1)

        x, y = cp.meshgrid(cp.arange(w, dtype=cp.float64), cp.arange(h, dtype=cp.float64))
        rad = cp.sqrt((y - cnt[0])**2 + (x - cnt[1])**2)
        maxval = cp.sqrt(kx*kx + ky*ky)
        ratio = rad / maxval if maxval != 0 else cp.zeros_like(rad)

        mask1 = (cp.abs(weight0) < weightLimit) | (cp.abs(wt0) < weightLimit)
        band0[mask1] = 0
        mask2 = (cp.abs(weight1) < weightLimit) | (cp.abs(wt1) < weightLimit)
        band1[mask2] = 0

        weight0 = cp.where(weight0 == 0, 1, weight0)
        weight1 = cp.where(weight1 == 0, 1, weight1)

        if divideByOtf:
            band0 = band0 / weight0
            band1 = band1 / weight1

        mask = (ratio < dist) | (ratio > (1 - dist))
        band0[mask] = 0

        def matlab_round(v):
            v = float(v)
            return int(cp.floor(v + 0.5)) if v >= 0 else int(cp.ceil(v - 0.5))

        shift_x = matlab_round(kx)
        shift_y = matlab_round(ky)

        mask0 = cp.roll(mask, shift=(-shift_y, -shift_x), axis=(0, 1))
        band1[mask0] = 0
        return band0, band1

    def fitPeak(self, band0, band1, bn0, bn1, otf, kx, ky, weightLimit, search, cntrl=None):
        class Peak:
            pass
        peak = Peak()
        resPhase = 0.0
        resMag = 0.0

        for it in range(3):
            b0 = band0.copy()
            b1 = band1.copy()
            dist = 0.15
            divideByOtf = True
            b0, b1 = self.commonRegion(b0, b1, bn0, bn1, otf, kx, ky, dist, weightLimit, divideByOtf)

            b0 = fft.ifft2(fft.fftshift(b0))
            b1 = fft.ifft2(fft.fftshift(b1))

            corr = cp.zeros((10, 10), dtype=cp.complex128)
            denom = cp.sum(cp.real(b0)**2 + cp.imag(b0)**2)
            scal = 1.0 / denom if denom != 0 else 0.0

            cmax = 0.0
            cmin = cp.inf
            newKx = 0.0
            newKy = 0.0

            tkx = kx
            tky = ky
            ts = search

            for yi_idx in range(10):
                for xi_idx in range(10):
                    xpos = tkx + ((xi_idx - 4.5) / 4.5) * ts
                    ypos = tky + ((yi_idx - 4.5) / 4.5) * ts

                    b1s = self.fourierShift(b1, xpos, ypos)
                    b1s = b1s * cp.conj(b0)
                    corr[xi_idx, yi_idx] = cp.sum(b1s) * scal

            for yi_idx in range(10):
                for xi_idx in range(10):
                    val = cp.abs(corr[xi_idx, yi_idx])
                    if val > cmax:
                        cmax = val
                        newKx = tkx + ((xi_idx - 4.5) / 4.5) * ts
                        newKy = tky + ((yi_idx - 4.5) / 4.5) * ts
                        resPhase = cp.angle(corr[xi_idx, yi_idx])
                        resMag = val
                    if val < cmin:
                        cmin = val

            if cntrl is not None and cntrl.size > 0:
                for yi_idx in range(10):
                    for xi_idx in range(10):
                        if cmax > cmin:
                            cntrl[yi_idx, xi_idx + it * 10] = (cp.abs(corr[xi_idx, yi_idx]) - cmin) / (cmax - cmin)
                        else:
                            cntrl[yi_idx, xi_idx + it * 10] = 0.0

            kx = float(newKx)
            ky = float(newKy)
            search = search / 3.0

        peak.kx = kx
        peak.ky = ky
        peak.resPhase = resPhase
        peak.resMag = resMag
        return peak, cntrl

    def getPeak(self, band0, band1, bn0, bn1, otf, kx, ky, weightLimit):
        b0, b1 = self.commonRegion(band0, band1, bn0, bn1, otf, kx, ky, 0.15, weightLimit)
        b0 = fft.ifft2(fft.fftshift(b0))
        b1 = fft.ifft2(fft.fftshift(b1))
        b1 = self.fourierShift(b1, kx, ky)
        b1 *= cp.conj(b0)
        scal = 1.0 / cp.sum(cp.real(b0)**2 + cp.imag(b0)**2)
        return cp.sum(b1) * scal

    def otfToVector(self, vec, otf, kx=0.0, ky=0.0, useAtt=0, write=1):
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