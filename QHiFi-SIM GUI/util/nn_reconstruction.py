import os
import torch
import numpy as np
import tifffile as tif
from PyQt6.QtCore import QThread, pyqtSignal
from util.data_processor import ImageLoader
from util.QHiFi import QHiFi
from datetime import datetime
from scipy.signal import convolve2d


class NNReconstructionWorker(QThread):
    progress_signal = pyqtSignal(int)
    message_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.output_dir = None
        self.total_images = 0
        self.saved_count = 0

    def run(self):
        try:
            psf_params = self.params['hifi_params']
            self.image_loader = ImageLoader(
                self.params['wf_dir'],
                self.params['sim_dir'],
                hifi_params=psf_params
            )

            if self.params['mode'] == 'hifi_results':
                self.message_signal.emit("Starting QHiFi reconstruction from HIFI-SIM results...")
                self.process_hifi_results()
            else:
                self.message_signal.emit("Starting QHiFi reconstruction from SIM images...")
                self.process_external_images()

            self.finished_signal.emit(True, "QHiFi reconstruction completed successfully")

        except Exception as e:
            self.finished_signal.emit(False, str(e))
            self.message_signal.emit(f"Error during QHiFi reconstruction: {str(e)}")
            self.finished_signal.emit(False, f"QHiFi reconstruction failed: {str(e)}")

    def process_hifi_results(self):
        try:
            loader = self.image_loader
            image_pairs = []
            for idx in range(len(loader)):
                wf_img, sim_img, wf_file, sim_file = loader.load_image_pair(idx)
                wf_tensor = torch.from_numpy(wf_img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                sim_tensor = torch.from_numpy(sim_img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                image_pairs.append((wf_tensor, sim_tensor, {'wf_file': wf_file, 'sim_file': sim_file}))
            self._process_with_model(image_pairs, loader.get_psf_tensor())
        except Exception as e:
            raise RuntimeError(f"Error in HIFI results reconstruction: {str(e)}")

    def process_external_images(self):
        try:
            loader = self.image_loader
            image_pairs = []
            for idx in range(len(loader)):
                wf_img, sim_img, wf_file, sim_file = loader.load_image_pair(idx)
                wf_tensor = torch.from_numpy(wf_img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                sim_tensor = torch.from_numpy(sim_img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                image_pairs.append((wf_tensor, sim_tensor, {'wf_file': wf_file, 'sim_file': sim_file}))
            self._process_with_model(image_pairs, loader.get_psf_tensor())
        except Exception as e:
            raise RuntimeError(f"Error in external image reconstruction: {str(e)}")

    def _process_with_model(self, image_pairs, psf_tensor):
        try:
            total_images = len(image_pairs)
            self.total_images = total_images
            self.saved_count = 0
            if total_images == 0:
                raise ValueError("No image pairs found in specified directories!")
            self.progress_signal.emit(0)
            self.message_signal.emit(f"Starting QHiFi reconstruction for {total_images} image pairs...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_path = self.params['model_path']
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            checkpoint = torch.load(model_path, weights_only=True)
            model = QHiFi().to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.params['output_path'], f'Results_{current_time}')
            os.makedirs(output_dir, exist_ok=True)
            subdirs = ['QWF', 'QHiFi', 'DWF', 'HiFi']
            for subdir in subdirs:
                os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
            psf_input = psf_tensor.to(device).contiguous()
            with torch.no_grad():
                for idx, (wf_tensor, sim_tensor, file_info) in enumerate(image_pairs):
                    progress = int((idx + 1) / total_images * 100)
                    self.progress_signal.emit(progress)
                    wf_file = file_info['wf_file']
                    sim_file = file_info['sim_file']
                    self.message_signal.emit(f"Processing image {idx + 1}/{total_images}: {os.path.basename(wf_file)}")
                    wf_input = wf_tensor.to(device)
                    sim_input = sim_tensor.to(device)
                    wf_input = wf_input.contiguous()
                    sim_input = sim_input.contiguous()
                    assert wf_input.dim() == 4, f"wf_input should be 4D, got {wf_input.dim()}D"
                    assert sim_input.dim() == 4, f"sim_input should be 4D, got {sim_input.dim()}D"
                    assert psf_input.dim() == 4, f"psf_input should be 4D, got {psf_input.dim()}D"

                    qsim, qwf, dwf = model(sim_input, wf_input, psf_input)

                    qwf_np = qwf.detach().cpu().numpy()
                    qsim_np = qsim.detach().cpu().numpy()

                    qwf_img = np.squeeze(qwf_np)
                    qsim_img = np.squeeze(qsim_np)

                    if qwf_img.ndim == 3:
                        qwf_img = qwf_img[0] if qwf_img.shape[0] == 1 else qwf_img[:, :, 0]
                    if qsim_img.ndim == 3:
                        qsim_img = qsim_img[0] if qsim_img.shape[0] == 1 else qsim_img[:, :, 0]

                    if qwf_img.ndim != 2 or qsim_img.ndim != 2:
                        raise ValueError(f"Unexpected shape: qwf={qwf_img.shape}, qsim={qsim_img.shape}")

                    wf_original = tif.imread(wf_file).astype(np.float32)
                    sim_raw = tif.imread(sim_file).astype(np.float32)
                    kkk = np.percentile(wf_original, 100)

                    qwf_final = qwf_img * kkk
                    qsim_final = qsim_img * kkk

                    psf_np = psf_tensor.cpu().numpy()[0, 0]
                    dwf_raw = convolve2d(sim_raw, psf_np, mode='same', boundary='wrap')

                    vec_dwf = dwf_raw.ravel()
                    vec_wf = wf_original.ravel()
                    c_opt = np.dot(vec_dwf, vec_wf) / (np.dot(vec_dwf, vec_dwf) + 1e-12)
                    if c_opt <= 0:
                        c_opt = 1e-3

                    dwf_final = dwf_raw * c_opt
                    sim_final = sim_raw * c_opt

                    output_name = f"{idx:04d}"
                    output_paths = [
                        (os.path.join(output_dir, 'QWF', f"qwf_{output_name}.tif"), qwf_final),
                        (os.path.join(output_dir, 'QHiFi', f"qHiFi_{output_name}.tif"), qsim_final),
                        (os.path.join(output_dir, 'DWF', f"dwf_{output_name}.tif"), dwf_final),
                        (os.path.join(output_dir, 'HiFi', f"HiFi_{output_name}.tif"), sim_final)
                    ]

                    for output_path, output_data in output_paths:
                        tif.imwrite(output_path, output_data.astype('float32'), dtype='float32')
                    self.saved_count += 1
                    progress = int(self.saved_count / self.total_images * 100)
                    self.progress_signal.emit(progress)
            self.output_dir = output_dir
            self.message_signal.emit(f"QHiFi reconstruction completed! Results saved to {output_dir}")
            return True
        except Exception as e:
            raise RuntimeError(f"Error in model processing: {str(e)}")

