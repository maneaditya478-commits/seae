"""Reference implementations for image processing kernels.

All reference kernels serve as golden ground truth for validation and testing.
Implementations prioritize correctness and mathematical clarity over performance.
"""

from typing import Tuple, Optional
import numpy as np


def gaussian_kernel_2d(ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Generate a normalized 2D Gaussian kernel.
    
    Args:
        ksize: Size of the square kernel (must be odd).
        sigma: Standard deviation of Gaussian.
        
    Returns:
        2D numpy array of shape (ksize, ksize) summing to 1.0.
    """
    if ksize % 2 == 0:
        raise ValueError(f"ksize must be odd, got {ksize}")
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def ref_gaussian_blur(
    img: np.ndarray,
    ksize: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """Golden reference for Gaussian blur using 2D spatial convolution.
    
    Supports 2D grayscale images or 3D multi-channel images.
    Border handling: edge replication / clamp.
    """
    if img.ndim == 2:
        h, w = img.shape
        channels = 1
        work_img = img[:, :, None]
    elif img.ndim == 3:
        h, w, channels = img.shape
        work_img = img
    else:
        raise ValueError(f"Input image must be 2D or 3D, got {img.shape}")

    pad = ksize // 2
    kernel = gaussian_kernel_2d(ksize, sigma)
    
    # Pad image with replicate
    padded = np.pad(work_img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(work_img, dtype=np.float32)

    for c in range(channels):
        for i in range(h):
            for j in range(w):
                region = padded[i:i + ksize, j:j + ksize, c]
                out[i, j, c] = np.sum(region * kernel)

    if img.ndim == 2:
        out = out[:, :, 0]
        
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(out), 0, np.iinfo(img.dtype).max).astype(img.dtype)
    return out.astype(img.dtype)


def ref_threshold(
    img: np.ndarray,
    thresh: float = 128.0,
    maxval: float = 255.0
) -> np.ndarray:
    """Golden reference for binary thresholding.
    
    Pixel > thresh is set to maxval, otherwise 0.
    """
    out = np.where(img > thresh, maxval, 0)
    return out.astype(img.dtype)


def ref_conv2d(
    img: np.ndarray,
    kernel: Optional[np.ndarray] = None
) -> np.ndarray:
    """Golden reference for general 2D convolution.
    
    Border handling: edge replicate.
    """
    if kernel is None:
        # Default simple 3x3 sharpen/box filter
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    if img.ndim == 2:
        h, w = img.shape
        work_img = img[:, :, None]
        channels = 1
    else:
        h, w, channels = img.shape
        work_img = img

    padded = np.pad(work_img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="edge")
    out = np.zeros_like(work_img, dtype=np.float32)

    # Note: 2D convolution performs kernel flipping; correlation does not.
    # Standard spatial filtering convention in image processing:
    flipped_k = np.flip(kernel)

    for c in range(channels):
        for i in range(h):
            for j in range(w):
                region = padded[i:i + k_h, j:j + k_w, c]
                out[i, j, c] = np.sum(region * flipped_k)

    if img.ndim == 2:
        out = out[:, :, 0]

    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(out), 0, np.iinfo(img.dtype).max).astype(img.dtype)
    return out.astype(img.dtype)


def ref_sobel(img: np.ndarray) -> np.ndarray:
    """Golden reference for Sobel edge detection magnitude.
    
    Computes sqrt(Gx^2 + Gy^2).
    """
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    if img.ndim == 3:
        # Convert to single channel grayscale if multi-channel (standard luminance)
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float32)

    h, w = gray.shape
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(region * sobel_x)
            gy[i, j] = np.sum(region * sobel_y)

    mag = np.hypot(gx, gy)
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(mag), 0, np.iinfo(img.dtype).max).astype(img.dtype)
    return mag.astype(img.dtype)


def ref_fused_contrast_blur(
    img: np.ndarray,
    alpha: float = 1.5,
    beta: float = 10.0,
    ksize: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Golden reference for custom fused kernel: Contrast Adjustment + Gaussian Blur.

    Operation:
        1. Linear contrast stretch & brightness offset: out1 = clip(alpha * img + beta, 0, 255)
        2. Gaussian spatial smoothing blur on the enhanced result.
    No single native OpenCV library function implements this fused pipeline in a single pass.
    """
    if img.ndim == 2:
        work_img = img[:, :, None]
    else:
        work_img = img

    # 1. Contrast adjustment
    enhanced = np.clip(alpha * work_img.astype(np.float32) + beta, 0.0, 255.0)

    # 2. Gaussian blur
    h, w, c = enhanced.shape
    pad = ksize // 2
    kernel = gaussian_kernel_2d(ksize, sigma)
    padded = np.pad(enhanced, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(enhanced, dtype=np.float32)

    for ch in range(c):
        for i in range(h):
            for j in range(w):
                region = padded[i:i + ksize, j:j + ksize, ch]
                out[i, j, ch] = np.sum(region * kernel)

    if img.ndim == 2:
        out = out[:, :, 0]

    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(out), 0, 255).astype(img.dtype)
    return out.astype(img.dtype)


REFERENCE_REGISTRY = {
    "gaussian_blur": ref_gaussian_blur,
    "threshold": ref_threshold,
    "conv2d": ref_conv2d,
    "sobel": ref_sobel,
    "fused_contrast_blur": ref_fused_contrast_blur,
}


def get_reference_kernel(kernel_name: str):
    """Retrieve reference kernel function by name."""
    if kernel_name not in REFERENCE_REGISTRY:
        raise KeyError(f"Unknown kernel: {kernel_name}. Available: {list(REFERENCE_REGISTRY.keys())}")
    return REFERENCE_REGISTRY[kernel_name]
