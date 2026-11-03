#
# Mesh-guided Gaussian Splatting - Initialization module
# Image appearance-complexity (Sobel + local variance)
#

import numpy as np
import cv2


_SOBEL_KERNEL = None


def _sobel_kernel():
    """Cached 3x3 Sobel kernels (Gx, Gy).  Constructed by outer-product of
    a 1-D Gaussian-derivative kernel with itself."""
    global _SOBEL_KERNEL
    if _SOBEL_KERNEL is None:
        g1 = cv2.getGaussianKernel(3, 1.0).flatten().astype(np.float32)  # (3,)
        gx = np.outer(g1, np.array([-1.0, 0.0, 1.0], dtype=np.float32))   # (3, 3)
        gy = np.outer(np.array([-1.0, 0.0, 1.0], dtype=np.float32), g1)    # (3, 3)
        _SOBEL_KERNEL = (gx, gy)
    return _SOBEL_KERNEL


def _sobel_magnitude(gray01):
    gx, gy = _sobel_kernel()
    sx = cv2.filter2D(gray01, cv2.CV_32F, gx)
    sy = cv2.filter2D(gray01, cv2.CV_32F, gy)
    return np.sqrt(sx * sx + sy * sy)


def _local_variance(gray01, kernel_size=5):
    mu  = cv2.boxFilter(gray01,         cv2.CV_32F, (kernel_size, kernel_size))
    mu2 = cv2.boxFilter(gray01 * gray01, cv2.CV_32F, (kernel_size, kernel_size))
    return np.maximum(mu2 - mu * mu, 0.0)


def compute_image_complexity(image_rgb01, sobel_weight=0.5, var_weight=0.5,
                              var_kernel=5, eps=1e-6):
    """
    Compute a per-pixel appearance complexity map.

    Parameters
    ----------
    image_rgb01 : (H, W, 3) float32 in [0, 1]
    sobel_weight : float
    var_weight   : float
    var_kernel   : int  (odd number, >1)
    eps          : float

    Returns
    -------
    (H, W) float32 in [0, 1]
    """
    assert image_rgb01.ndim == 3 and image_rgb01.shape[2] == 3, \
        f"expected (H, W, 3) image, got {image_rgb01.shape}"
    img = image_rgb01.astype(np.float32)

    gray_u8 = cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8),
                           cv2.COLOR_RGB2GRAY)
    gray01 = gray_u8.astype(np.float32) / 255.0

    sob = _sobel_magnitude(gray01)                 # usually in [0, ~3]
    var = _local_variance(gray01, kernel_size=var_kernel)  # in [0, ~0.5]

    sob_n = sob / (sob.max() + eps)
    var_n = var / (var.max() + eps)

    complexity = sobel_weight * sob_n + var_weight * var_n
    complexity = complexity / (complexity.max() + eps)
    return complexity.astype(np.float32)
