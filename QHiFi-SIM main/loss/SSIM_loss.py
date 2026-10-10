import torch
import torch.nn.functional as F


def gaussian(window_size, sigma):
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=1.5, dtype=torch.float32, device='cpu'):
    if window_size % 2 == 0:
        raise ValueError(f"Window size must be odd, got {window_size}")

    _1D_window = gaussian(window_size, sigma).unsqueeze(1).to(device, dtype=dtype)
    _2D_window = _1D_window @ _1D_window.t()
    window = _2D_window.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, data_range=1.0):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    sigma1_sq = torch.clamp(sigma1_sq, min=1e-8)
    sigma2_sq = torch.clamp(sigma2_sq, min=1e-8)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=(1, 2, 3))


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, sigma=1.5, data_range=1.0):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.sigma = sigma
        self.data_range = data_range
        self.register_buffer('window', torch.Tensor())
        self.register_buffer('channel', torch.tensor(1))

    def forward(self, img1, img2):
        if img1.shape != img2.shape:
            raise ValueError(f"img1 shape {img1.shape} != img2 shape {img2.shape}")

        b, c, h, w = img1.shape
        dtype = img1.dtype
        device = img1.device

        if c != self.channel.item() or self.window.numel() == 0 or self.window.dtype != dtype or self.window.device != device:
            self.channel = torch.tensor(c, device=device)
            self.window = create_window(
                window_size=self.window_size,
                channel=c,
                sigma=self.sigma,
                dtype=dtype,
                device=device
            )

        return _ssim(img1, img2, self.window, self.window_size, c, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True, sigma=1.5, data_range=1.0):
    if img1.shape != img2.shape:
        raise ValueError(f"img1 shape {img1.shape} != img2 shape {img2.shape}")

    b, c, h, w = img1.shape
    window = create_window(
        window_size=window_size,
        channel=c,
        sigma=sigma,
        dtype=img1.dtype,
        device=img1.device
    )
    return _ssim(img1, img2, window, window_size, c, size_average, data_range)
