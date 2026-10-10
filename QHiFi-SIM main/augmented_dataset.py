import os
import random
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from util.util import img_norm


class AugmentedDataset(Dataset):
    def __init__(self, datapath, datalist, mode='train', norm=True, augment=True,
                 patch_size=256, stride=64, psf_size=31):
        self.datapath = datapath
        self.datalist = datalist
        self.mode = mode
        self.norm = norm
        self.augment = augment
        self.patch_size = patch_size
        self.stride = stride
        self.psf_size = psf_size
        self.wf_path, self.sim_path, self.psf_path = self._read_datalist()
        self.augment_combos = self._get_augmentation_combinations()
        self.non_identity_combos = self.augment_combos[1:]
        self.items = []
        self._prepare_items()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wf_img = tifffile.imread(item['wf'])
        sim_img = tifffile.imread(item['sim'])
        psf = tifffile.imread(item['psf']).astype(np.float32)

        wf_img, sim_img = self._preprocess(wf_img, sim_img)
        psf_tensor = self._process_psf(psf)

        if self.mode == 'test' and item['x'] is None:
            wf_tensor = torch.from_numpy(wf_img[None]).contiguous()
            sim_tensor = torch.from_numpy(sim_img[None]).contiguous()
            return wf_tensor, sim_tensor, psf_tensor

        y_start = item['y']
        x_start = item['x']

        wf_patch = wf_img[y_start:y_start + self.patch_size,
                   x_start:x_start + self.patch_size]
        sim_patch = sim_img[y_start:y_start + self.patch_size,
                    x_start:x_start + self.patch_size]

        if self.mode == 'train' and self.augment:
            if item['aug_type'] == 'random':
                aug = random.choice(self.non_identity_combos)
                wf_patch, sim_patch = self._apply_augmentation(wf_patch, sim_patch, aug)
            elif item['aug_type'] == 'identity':
                pass

        wf_patch = np.expand_dims(wf_patch, 0)
        sim_patch = np.expand_dims(sim_patch, 0)

        wf_patch = np.ascontiguousarray(wf_patch)
        sim_patch = np.ascontiguousarray(sim_patch)
        psf_tensor = psf_tensor.contiguous()

        return wf_patch, sim_patch, psf_tensor

    def _read_datalist(self):
        with open(self.datalist, 'r') as f:
            wf_p, sim_p, psf_p = [], [], []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    wf, sim, psf = parts
                else:
                    wf = sim = psf = parts[0]
                wf_p.append(os.path.join(self.datapath, wf))
                sim_p.append(os.path.join(self.datapath, sim))
                psf_p.append(os.path.join(self.datapath, psf))
        return wf_p, sim_p, psf_p

    def _get_augmentation_combinations(self):
        combos = [
            {'hflip': False, 'vflip': False, 'angle': 0, 'name': 'identity'},
            {'hflip': True, 'vflip': False, 'angle': 0, 'name': 'hflip'},
            {'hflip': False, 'vflip': True, 'angle': 0, 'name': 'vflip'},
            {'hflip': True, 'vflip': True, 'angle': 0, 'name': 'hflip_vflip'},
            {'hflip': False, 'vflip': False, 'angle': 90, 'name': 'rot90'},
            {'hflip': True, 'vflip': False, 'angle': 90, 'name': 'hflip_rot90'},
            {'hflip': False, 'vflip': True, 'angle': 90, 'name': 'vflip_rot90'},
            {'hflip': True, 'vflip': True, 'angle': 90, 'name': 'hflip_vflip_rot90'},
        ]
        return combos

    def _prepare_items(self):
        for img_idx in range(len(self.wf_path)):
            with tifffile.TiffFile(self.wf_path[img_idx]) as tif:
                h_raw, w_raw = tif.pages[0].shape

            if h_raw > 256 and w_raw > 256:
                h = h_raw - 50
                w = w_raw - 50
            else:
                h = h_raw
                w = w_raw

            positions_set = set()
            y_starts = []
            if h <= self.patch_size:
                y_starts = [0]
            else:
                y = 0
                while y + self.patch_size <= h:
                    y_starts.append(y)
                    y += self.stride
                last_y = h - self.patch_size
                if last_y not in y_starts:
                    y_starts.append(last_y)
            x_starts = []
            if w <= self.patch_size:
                x_starts = [0]
            else:
                x = 0
                while x + self.patch_size <= w:
                    x_starts.append(x)
                    x += self.stride
                last_x = w - self.patch_size
                if last_x not in x_starts:
                    x_starts.append(last_x)
            for y in y_starts:
                for x in x_starts:
                    positions_set.add((x, y))

            positions = list(positions_set)

            if self.mode == 'train' and self.augment:
                for x, y in positions:
                    self.items.append({
                        'wf': self.wf_path[img_idx],
                        'sim': self.sim_path[img_idx],
                        'psf': self.psf_path[img_idx],
                        'x': x,
                        'y': y,
                        'aug_type': 'identity'
                    })
                    self.items.append({
                        'wf': self.wf_path[img_idx],
                        'sim': self.sim_path[img_idx],
                        'psf': self.psf_path[img_idx],
                        'x': x,
                        'y': y,
                        'aug_type': 'random'
                    })
            else:
                for x, y in positions:
                    self.items.append({
                        'wf': self.wf_path[img_idx],
                        'sim': self.sim_path[img_idx],
                        'psf': self.psf_path[img_idx],
                        'x': x,
                        'y': y,
                        'aug_type': 'none'
                    })

                if self.mode == 'test':
                    self.items.append({
                        'wf': self.wf_path[img_idx],
                        'sim': self.sim_path[img_idx],
                        'psf': self.psf_path[img_idx],
                        'x': None,
                        'y': None,
                        'aug_type': 'none'
                    })

    def _apply_augmentation(self, wf_patch, sim_patch, aug):
        if aug['hflip']:
            wf_patch = np.fliplr(wf_patch)
            sim_patch = np.fliplr(sim_patch)

        if aug['vflip']:
            wf_patch = np.flipud(wf_patch)
            sim_patch = np.flipud(sim_patch)

        if aug['angle'] > 0:
            k = aug['angle'] // 90
            wf_patch = np.rot90(wf_patch, k)
            sim_patch = np.rot90(sim_patch, k)

        return wf_patch, sim_patch

    def _preprocess(self, wf_img, sim_img):
        wf_img = np.array(wf_img, dtype=np.float32)
        sim_img = np.array(sim_img, dtype=np.float32)

        if self.norm:
            wf_img = img_norm(wf_img)
            sim_img = img_norm(sim_img)

        h, w = wf_img.shape
        if h > 256 and w > 256:
            if h - 50 < 256:
                start_h = max(0, h - 256)
                h = 256
            else:
                start_h = 25
                h = h - 50

            if w - 50 < 256:
                start_w = max(0, w - 256)
                w = 256
            else:
                start_w = 25
                w = w - 50

            wf_img = wf_img[start_h:start_h + h, start_w:start_w + w]
            sim_img = sim_img[start_h:start_h + h, start_w:start_w + w]

        return wf_img, sim_img

    def _process_psf(self, psf):
        if psf.ndim != 2 or psf.shape[0] != psf.shape[1]:
            raise ValueError(f'PSF shape error {psf.shape}')

        if psf.shape[0] >= 32:
            psf = psf[1:, 1:]

        s = self.psf_size
        if s > psf.shape[0]:
            raise ValueError(f'psf_size={s} exceeds available PSF size {psf.shape[0]}')
        if s % 2 == 0:
            raise ValueError(f'psf_size must be odd, got {s}')

        if s < psf.shape[0]:
            start = (psf.shape[0] - s) // 2
            psf = psf[start:start + s, start:start + s]

        psf = psf.astype(np.float32)
        psf = psf / (psf.sum() + 1e-12)

        return torch.from_numpy(psf).unsqueeze(0).contiguous()
