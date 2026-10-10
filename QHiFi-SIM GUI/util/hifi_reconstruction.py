import os
import cupy as cp
import numpy as np
import tifffile as tif
from PyQt6.QtCore import QThread, pyqtSignal
from util.ParameterSet import ParameterSet
from util.edgefilter import edgefilter
from util.ParameterEstimationCOR import ParameterEstimationCOR
from util.Hifi_SIM import HifiReconstruction
from datetime import datetime
from skimage import restoration
import cupy.fft as fft


def zero_padding_upsample(image, factor=2, debug_dir=None):
    h, w = image.shape
    fft_img = fft.fft2(image)
    fft_shifted = fft.fftshift(fft_img)
    new_h, new_w = int(h * factor), int(w * factor)
    fft_padded = cp.zeros((new_h, new_w), dtype=cp.complex128)
    h_start = (new_h - h) // 2
    w_start = (new_w - w) // 2
    fft_padded[h_start:h_start + h, w_start:w_start + w] = fft_shifted
    fft_unshifted = fft.ifftshift(fft_padded)
    upsampled = fft.ifft2(fft_unshifted)
    upsampled_real = cp.abs(upsampled)
    upsampled_real *= factor ** 2
    return upsampled_real.astype(cp.float64)


class HifiReconstructionWorker(QThread):
    progress_signal = pyqtSignal(int)
    message_signal = pyqtSignal(str)
    file_progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(bool, str, str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.total_files = 0
        self.saved_count = 0
        self.timestamp_dir = None

    def run(self):
        try:
            total_files = len(self.params['raw_files'])
            self.message_signal.emit(f"Start processing {total_files} files...")
            output_path = self.params['output_path']
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp_dir = os.path.join(output_path, f'Results_{current_time}')
            os.makedirs(timestamp_dir, exist_ok=True)
            wf_dir = os.path.join(timestamp_dir, 'WF')
            sim_dir = os.path.join(timestamp_dir, 'HiFi-SIM')
            os.makedirs(wf_dir, exist_ok=True)
            os.makedirs(sim_dir, exist_ok=True)

            self.timestamp_dir = timestamp_dir
            self.total_files = total_files
            self.saved_count = 0

            for idx, raw_file in enumerate(self.params['raw_files']):
                self.process_single_file(idx, raw_file, total_files, wf_dir, sim_dir)
            self.message_signal.emit("All files processed!")
            self.finished_signal.emit(True, "HIFI-SIM reconstruction completed successfully", timestamp_dir)

        except Exception as e:
            timestamp_dir = getattr(self, 'timestamp_dir', "")
            self.message_signal.emit(f"Unexpected error: {str(e)}")
            self.finished_signal.emit(False, f"Processing failed: {str(e)}", timestamp_dir)

    def process_single_file(self, idx, raw_file, total_files, wf_dir, sim_dir):
        file_name = os.path.basename(raw_file)
        self.message_signal.emit(f"Processing file {idx + 1}/{total_files}: {file_name}")
        img_data = tif.imread(raw_file)
        if len(img_data.shape) != 3 or img_data.shape[0] != 9:
            raise ValueError(
                f"Raw data frames is not correct! Expected 9 frames (3 orientations × 3 phases), "
                f"but got shape {img_data.shape}. File: {file_name}"
            )

        Iraw = cp.asarray(img_data).astype(cp.float64)

        # --------------------------------------------  Wide-field  ===================================================
        wf = cp.mean(Iraw, axis=0)
        wf = edgefilter(wf)
        wf_2 = zero_padding_upsample(wf, factor=2)
        wf_2 = edgefilter(wf_2)
        file_base = os.path.splitext(file_name)[0]
        wf_output_path = os.path.join(wf_dir, f'{file_base}_WF.tif')
        tif.imwrite(wf_output_path, cp.asnumpy(wf_2).astype(np.float64), dtype=np.float64)

        # -=----------------------------------------  Preprocessing  --------------------------------------------------
        param_set = ParameterSet(
            Iraw=Iraw,
            Pixelsize=self.params['pixel_size'],
            NA=self.params['na'],
            lambda_=self.params['wavelength'],
        )
        psf = param_set.psf.astype(cp.float64)

        Iraw_2 = edgefilter(Iraw).astype(cp.float64)
        Iraw_deconvolved = Iraw_2
        '''
        Iraw_deconvolved = cp.zeros_like(Iraw_2)
        psf_cpu = cp.asnumpy(psf)
        for i in range(9):
            frame_cpu = cp.asnumpy(Iraw_2[i]).astype(np.float64)
            deconvolved_cpu = restoration.richardson_lucy(
                frame_cpu, psf_cpu, num_iter=5, clip=False
            )
            Iraw_deconvolved[i] = cp.asarray(deconvolved_cpu)
        '''

        # ----------------------------------  Parameter  Estimation  --------------------------------------------------
        IIrawFFT = fft.fftshift(fft.fft2(Iraw_deconvolved, axes=(-2, -1)), axes=(-2, -1))
        par = ParameterEstimationCOR(IIrawFFT, param_set)
        Dir, K0 = par.run()

        for i in range(len(Dir)):
            Dir[i]['K0'] = float(K0[i])
        param_set.Dir = Dir

        # ======================================================================================================
        print(f"\n===== Parameter Estimation for: {file_name} =====")
        for i in range(len(Dir)):
            px = float(Dir[i]['px'])
            py = float(Dir[i]['py'])
            phaOff = float(Dir[i]['phaOff'])
            modul = Dir[i]['modul']
            k0 = float(K0[i])
            angle_deg = float(np.degrees(np.arctan2(py, px)))
            print(
                f"Dir[{i}]: "
                f"px={px:+.6f}, py={py:+.6f}, "
                f"K0={k0:.6f}, "
                f"angle={angle_deg:+.4f}°, "
                f"phaOff={phaOff:+.6f}, "
                f"modul={modul:.6f}"
            )
        print("================================================\n")

        # --------------------------------------------  Reconstruction  ----------------------------------------------
        hifi_recon = HifiReconstruction(IIrawFFT, param_set, Dir)
        HIFI = hifi_recon.reconstruct()
        hifi_output_path = os.path.join(sim_dir, f'{file_base}_HiFi.tif')

        tif.imwrite(hifi_output_path, cp.asnumpy(HIFI).astype(np.float64), dtype=np.float64)

        self.saved_count += 1
        progress = int(self.saved_count / self.total_files * 100)
        self.file_progress_signal.emit(self.saved_count, self.total_files)
        self.progress_signal.emit(progress)
        self.message_signal.emit(f"File processed successfully: {hifi_output_path}")