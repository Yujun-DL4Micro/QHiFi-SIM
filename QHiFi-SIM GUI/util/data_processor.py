import os
import numpy as np
import cupy as cp
import tifffile
import glob
import re
import torch
from util.ParameterSet import ParameterSet


class ImageLoader:
    def __init__(self, wf_dir, sim_dir,hifi_params, norm=True, device='cuda:0'):
        self.wf_dir = wf_dir
        self.sim_dir = sim_dir
        self.hifi_params = hifi_params
        self.norm = norm
        self.device = device
        self.wf_files = self._get_sorted_files(wf_dir)
        self.sim_files = self._get_sorted_files(sim_dir)

        if len(self.wf_files) != len(self.sim_files):
            raise ValueError(
                f"File count mismatch: {len(self.wf_files)} WF files vs {len(self.sim_files)} SIM files"
            )

        self.psf_tensor = self._generate_psf_gpu(device=device)

    def _get_sorted_files(self, directory):
        files = glob.glob(os.path.join(directory, "*.tif"))

        def extract_number(filename):
            basename = os.path.basename(filename)
            numbers = re.findall(r'\d+', basename)
            return int(numbers[0]) if numbers else 0

        return sorted(files, key=extract_number)

    def _generate_psf_gpu(self, device='cuda:0'):
        pixel_size = self.hifi_params['pixel_size']
        na = self.hifi_params['na']
        wavelength = self.hifi_params['wavelength']

        npixel = 1004
        if len(self.wf_files) > 0:
            try:
                wf_sample = tifffile.imread(self.wf_files[0])
                if wf_sample.ndim == 2:
                    npixel = wf_sample.shape[0]
                elif wf_sample.ndim == 3:
                    npixel = wf_sample.shape[1]
            except:
                pass

        fake_Iraw_gpu = cp.zeros((1, npixel, npixel), dtype=cp.float32)

        param_set = ParameterSet(
            Iraw=fake_Iraw_gpu,
            Pixelsize=pixel_size,
            NA=na,
            lambda_=wavelength
        )

        psf_32 = param_set.psf2
        if not isinstance(psf_32, cp.ndarray):
            psf_32 = cp.asarray(psf_32)

        psf_31 = psf_32[1:32, 1:32]
        psf_sum = cp.sum(psf_31)
        if psf_sum != 0:
            psf_31 = psf_31 / psf_sum

        psf_tensor = torch.as_tensor(psf_31, device=device)
        psf_tensor = psf_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 31, 31)

        return psf_tensor.contiguous()

    def get_psf_tensor(self):
        return self.psf_tensor

    def get_psf_tensor_for_device(self, device=None):
        if device is None:
            return self.psf_tensor
        else:
            return self.psf_tensor.to(device)

    def load_image_pair(self, idx):
        wf_file = self.wf_files[idx]
        sim_file = self.sim_files[idx]

        wf_img = tifffile.imread(wf_file).astype(np.float32)
        sim_img = tifffile.imread(sim_file).astype(np.float32)

        if self.norm:
            wf_img = self.img_norm(wf_img)
            sim_img = self.img_norm(sim_img)

        return wf_img, sim_img, wf_file, sim_file

    def img_norm(self, img):
        max_val = np.max(img)
        img_norm = img / (max_val + 1e-8)
        return img_norm

    def __len__(self):
        return len(self.wf_files)
