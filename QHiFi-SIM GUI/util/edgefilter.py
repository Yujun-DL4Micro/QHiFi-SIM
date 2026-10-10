import numpy as np
import cupy as cp


def edgefilter(images):
    xp = cp if isinstance(images, cp.ndarray) else np
    original_ndim = images.ndim
    if original_ndim == 2:
        images = images[None, ...]
        was_2d = True
    else:
        was_2d = False

    Z, H, W = images.shape
    images = images.astype(xp.float32)
    for I in range(Z):
        curImg = images[I, :, :].copy()
        px = 10 if H > 256 else 0

        if px > 0:
            fac = xp.pi / (2 * px)

            y = xp.arange(px)[:, None]
            top_fac = xp.sin(y * fac) ** 2
            curImg[:px, :] *= top_fac

            y = xp.arange(px)[::-1][:, None]
            bottom_fac = xp.sin(y * fac) ** 2
            curImg[-px:, :] *= bottom_fac

            x = xp.arange(px)[None, :]
            left_fac = xp.sin(x * fac) ** 2
            curImg[:, :px] *= left_fac

            x = xp.arange(px)[::-1][None, :]
            right_fac = xp.sin(x * fac) ** 2
            curImg[:, -px:] *= right_fac

        images[I, :, :] = curImg

    if was_2d:
        images = images[0]

    return images
