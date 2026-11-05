"""Candidate Generator for SEAE.

Synthesizes executable Python image-processing kernels across a parameter space:
- Loop order (outer-inner tile ordering)
- Tiling block sizes (e.g. 16, 32, 64, 128)
- Vectorization backend: NumPy vectorization, pure Python loop, Numba JIT
- Numba JIT options: nopython, fastmath, parallel (prange)
- Quantization: float32 vs int8 / uint8
- Separable filter decomposition (1D row * 1D col) vs 2D spatial
- OpenCV built-in baseline

Search Algorithms:
1. Random Search (with reproducible fixed seeds)
2. Epsilon-Greedy Multi-Armed Bandit (balances exploration vs exploitation)
3. Evolutionary Algorithm (tournament selection, gene mutation, crossover)
"""

import random
import copy
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class KernelConfig:
    """Configuration genes representing an implementation variant."""
    backend: str = "numpy"          # "python_loops", "numpy", "numba", "opencv"
    tiling: int = 0                 # 0 = no tiling; >0: block size (16, 32, 64, 128)
    parallel: bool = False          # multi-threading via prange / threads
    fastmath: bool = False          # aggressive IEEE-754 relaxing in Numba
    separable: bool = False         # 1D x 1D decomposition (for Gaussian blur / Sobel)
    quantized: bool = False         # int8/uint8 fixed-point arithmetic vs float32
    loop_order: str = "row_col"     # "row_col" (C-contiguous) vs "col_row" (strided)
    extra_flags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    """A generated candidate kernel."""
    candidate_id: str
    kernel_name: str
    config: KernelConfig
    source_code: str
    generation: int = 0
    score: Optional[float] = None   # Higher is better: 1000.0 / median_time_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kernel_name": self.kernel_name,
            "config": self.config.to_dict(),
            "source_code": self.source_code,
            "generation": self.generation,
            "score": self.score,
        }


# ==============================================================================
# CODE TEMPLATE GENERATORS
# ==============================================================================

def generate_opencv_kernel(kernel_name: str, config: Optional[KernelConfig] = None) -> str:
    """Generate OpenCV baseline implementation with optional thread count control."""
    threads = 8
    if config:
        if "threads" in config.extra_flags:
            threads = config.extra_flags["threads"]
        elif hasattr(config, "parallel") and not config.parallel:
            threads = 1

    thread_setup = f"cv2.setNumThreads({threads})\n"

    if kernel_name == "gaussian_blur":
        return f'''import cv2
import numpy as np
{thread_setup}
def kernel_entry(img, ksize=5, sigma=1.0, **kwargs):
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
'''
    elif kernel_name == "threshold":
        return f'''import cv2
import numpy as np
{thread_setup}
def kernel_entry(img, thresh=128.0, maxval=255.0, **kwargs):
    _, res = cv2.threshold(img, float(thresh), float(maxval), cv2.THRESH_BINARY)
    return res
'''
    elif kernel_name == "sobel":
        return f'''import cv2
import numpy as np
{thread_setup}
def kernel_entry(img, **kwargs):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    mag = cv2.magnitude(gx, gy)
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(mag), 0, 255).astype(img.dtype)
    return mag.astype(img.dtype)
'''
    elif kernel_name == "conv2d":
        return f'''import cv2
import numpy as np
{thread_setup}
def kernel_entry(img, kernel=None, **kwargs):
    if kernel is None:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    flipped_k = np.flip(kernel)
    filtered = cv2.filter2D(img, cv2.CV_32F, flipped_k, borderType=cv2.BORDER_REPLICATE)
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(filtered), 0, 255).astype(img.dtype)
    return filtered.astype(img.dtype)
'''
    elif kernel_name == "fused_contrast_blur":
        return f'''import cv2
import numpy as np
{thread_setup}
def kernel_entry(img, alpha=1.5, beta=10.0, ksize=5, sigma=1.0, **kwargs):
    enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    blurred = cv2.GaussianBlur(enhanced, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return blurred
'''
    raise ValueError(f"Unknown kernel name: {kernel_name}")


def generate_numpy_kernel(kernel_name: str, config: KernelConfig) -> str:
    """Generate vectorized NumPy kernel implementation."""
    if kernel_name == "threshold":
        return r'''import numpy as np

def kernel_entry(img, thresh=128.0, maxval=255.0, **kwargs):
    return np.where(img > thresh, maxval, 0).astype(img.dtype)
'''
    elif kernel_name == "gaussian_blur":
        if config.separable:
            return r'''import numpy as np

def _gaussian_1d(ksize, sigma):
    radius = ksize // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / np.sum(k)

def kernel_entry(img, ksize=5, sigma=1.0, **kwargs):
    is_2d = (img.ndim == 2)
    arr = img[:, :, None] if is_2d else img
    h, w, c = arr.shape
    pad = ksize // 2
    k1d = _gaussian_1d(ksize, sigma)

    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge").astype(np.float32)
    temp = np.zeros_like(padded)

    # 1. Convolve horizontally
    for i in range(ksize):
        temp[:, pad:pad+w, :] += padded[:, i:i+w, :] * k1d[i]

    # 2. Convolve vertically
    out = np.zeros((h, w, c), dtype=np.float32)
    for i in range(ksize):
        out += temp[i:i+h, pad:pad+w, :] * k1d[i]

    if is_2d:
        out = out[:, :, 0]

    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(out), 0, np.iinfo(img.dtype).max).astype(img.dtype)
    return out.astype(img.dtype)
'''
        else:
            return r'''import numpy as np

def _gaussian_2d(ksize, sigma):
    radius = ksize // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return k / np.sum(k)

def kernel_entry(img, ksize=5, sigma=1.0, **kwargs):
    is_2d = (img.ndim == 2)
    arr = img[:, :, None] if is_2d else img
    h, w, c = arr.shape
    pad = ksize // 2
    kernel = _gaussian_2d(ksize, sigma)

    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge").astype(np.float32)
    out = np.zeros((h, w, c), dtype=np.float32)

    for ki in range(ksize):
        for kj in range(ksize):
            weight = kernel[ki, kj]
            out += padded[ki:ki+h, kj:kj+w, :] * weight

    if is_2d:
        out = out[:, :, 0]

    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(out), 0, np.iinfo(img.dtype).max).astype(img.dtype)
    return out.astype(img.dtype)
'''
    elif kernel_name == "sobel":
        return r'''import numpy as np

def kernel_entry(img, **kwargs):
    if img.ndim == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float32)

    h, w = gray.shape
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")

    # Vectorized 3x3 slice convolution
    gx = (
        -1.0 * padded[0:h, 0:w] + 1.0 * padded[0:h, 2:w+2]
        -2.0 * padded[1:h+1, 0:w] + 2.0 * padded[1:h+1, 2:w+2]
        -1.0 * padded[2:h+2, 0:w] + 1.0 * padded[2:h+2, 2:w+2]
    )
    gy = (
        -1.0 * padded[0:h, 0:w] - 2.0 * padded[0:h, 1:w+1] - 1.0 * padded[0:h, 2:w+2]
        +1.0 * padded[2:h+2, 0:w] + 2.0 * padded[2:h+2, 1:w+1] + 1.0 * padded[2:h+2, 2:w+2]
    )

    mag = np.hypot(gx, gy)
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(mag), 0, 255).astype(img.dtype)
    return mag.astype(img.dtype)
'''
    elif kernel_name == "conv2d":
        return r'''import numpy as np

def kernel_entry(img, kernel=None, **kwargs):
    if kernel is None:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

    is_2d = (img.ndim == 2)
    arr = img[:, :, None] if is_2d else img
    h, w, c = arr.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    flipped_k = np.flip(kernel)

    padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="edge").astype(np.float32)
    out = np.zeros((h, w, c), dtype=np.float32)

    for ki in range(kh):
        for kj in range(kw):
            weight = flipped_k[ki, kj]
            out += padded[ki:ki+h, kj:kj+w, :] * weight

    if is_2d:
        out = out[:, :, 0]

    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(out), 0, 255).astype(img.dtype)
    return out.astype(img.dtype)
'''
    elif kernel_name == "fused_contrast_blur":
        return r'''import numpy as np

def _gaussian_2d(ksize, sigma):
    radius = ksize // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return k / np.sum(k)

def kernel_entry(img, alpha=1.5, beta=10.0, ksize=5, sigma=1.0, **kwargs):
    # Vectorized NumPy fused pass
    enhanced = np.clip(alpha * img.astype(np.float32) + beta, 0.0, 255.0)
    is_2d = (enhanced.ndim == 2)
    arr = enhanced[:, :, None] if is_2d else enhanced
    h, w, c = arr.shape
    pad = ksize // 2
    kernel = _gaussian_2d(ksize, sigma)

    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros((h, w, c), dtype=np.float32)

    for ki in range(ksize):
        for kj in range(ksize):
            out += padded[ki:ki+h, kj:kj+w, :] * kernel[ki, kj]

    if is_2d:
        out = out[:, :, 0]

    return np.clip(np.round(out), 0, 255).astype(np.uint8)
'''
    raise ValueError(f"Unknown kernel name: {kernel_name}")


def generate_numba_kernel(kernel_name: str, config: KernelConfig) -> str:
    """Generate JIT-compiled Numba kernel with optional parallel and fastmath flags."""
    fastmath = config.fastmath
    parallel = config.parallel
    tiling = config.tiling
    loop_order = config.loop_order

    prange_import = "from numba import prange" if parallel else ""
    loop_fn = "prange" if parallel else "range"

    if kernel_name == "threshold":
        return f'''import numpy as np
from numba import njit
{prange_import}

@njit(fastmath={fastmath}, parallel={parallel}, cache=True)
def _nb_threshold(img, thresh, maxval):
    h, w = img.shape
    out = np.empty((h, w), dtype=img.dtype)
    for i in {loop_fn}(h):
        for j in range(w):
            if img[i, j] > thresh:
                out[i, j] = maxval
            else:
                out[i, j] = 0
    return out

def kernel_entry(img, thresh=128.0, maxval=255.0, **kwargs):
    return _nb_threshold(img, thresh, maxval)
'''

    elif kernel_name == "gaussian_blur":
        # Numba Gaussian 2D with optional tiling and loop orders
        tiling_code = ""
        if tiling > 0:
            outer_loops = f'''
    block = {tiling}
    for bi in {loop_fn}(0, h, block):
        for bj in range(0, w, block):
            i_max = min(bi + block, h)
            j_max = min(bj + block, w)
            for i in range(bi, i_max):
                for j in range(bj, j_max):
'''
        else:
            if loop_order == "col_row":
                outer_loops = f'''
    for j in {loop_fn}(w):
        for i in range(h):
'''
            else:
                outer_loops = f'''
    for i in {loop_fn}(h):
        for j in range(w):
'''

        return f'''import numpy as np
from numba import njit
{prange_import}

@njit(fastmath={fastmath}, cache=True)
def _make_gauss_kernel(ksize, sigma):
    rad = ksize // 2
    k = np.empty((ksize, ksize), dtype=np.float32)
    total = 0.0
    for i in range(ksize):
        x = i - rad
        for j in range(ksize):
            y = j - rad
            val = np.exp(-(x*x + y*y) / (2.0 * sigma * sigma))
            k[i, j] = val
            total += val
    for i in range(ksize):
        for j in range(ksize):
            k[i, j] /= total
    return k

@njit(fastmath={fastmath}, parallel={parallel}, cache=True)
def _nb_gaussian_blur(padded, kernel, h, w, ksize):
    rad = ksize // 2
    out = np.zeros((h, w), dtype=np.float32)
{outer_loops}
                    acc = 0.0
                    for ki in range(ksize):
                        for kj in range(ksize):
                            acc += padded[i + ki, j + kj] * kernel[ki, kj]
                    out[i, j] = acc
    return out

def kernel_entry(img, ksize=5, sigma=1.0, **kwargs):
    pad = ksize // 2
    kernel = _make_gauss_kernel(ksize, float(sigma))
    padded = np.pad(img.astype(np.float32), ((pad, pad), (pad, pad)), mode="edge")
    out = _nb_gaussian_blur(padded, kernel, img.shape[0], img.shape[1], ksize)
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(out), 0, np.iinfo(img.dtype).max).astype(img.dtype)
    return out.astype(img.dtype)
'''

    elif kernel_name == "sobel":
        separable = config.separable
        quantized = config.quantized
        unroll = config.extra_flags.get("unroll", 1)

        if separable:
            return f'''import numpy as np
from numba import njit
{prange_import}

@njit(fastmath={fastmath}, parallel={parallel}, cache=True)
def _nb_sobel_separable(padded, h, w):
    hx = np.empty((h, w), dtype=np.float32)
    hy = np.empty((h, w), dtype=np.float32)
    for i in {loop_fn}(h):
        for j in range(w):
            hx[i, j] = -padded[i+1, j] + padded[i+1, j+2]
            hy[i, j] = padded[i+1, j] + 2.0 * padded[i+1, j+1] + padded[i+1, j+2]

    hx_pad = np.empty((h + 2, w), dtype=np.float32)
    hx_pad[1:h+1, :] = hx
    hx_pad[0, :] = hx[0, :]
    hx_pad[h+1, :] = hx[h-1, :]

    hy_pad = np.empty((h + 2, w), dtype=np.float32)
    hy_pad[1:h+1, :] = hy
    hy_pad[0, :] = hy[0, :]
    hy_pad[h+1, :] = hy[h-1, :]

    out = np.empty((h, w), dtype=np.float32)
    for i in {loop_fn}(h):
        for j in range(w):
            gx = hx_pad[i, j] + 2.0 * hx_pad[i+1, j] + hx_pad[i+2, j]
            gy = -hy_pad[i, j] + hy_pad[i+2, j]
            out[i, j] = np.sqrt(gx * gx + gy * gy)
    return out

def kernel_entry(img, **kwargs):
    if img.ndim == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float32)
    h, w = gray.shape
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    mag = _nb_sobel_separable(padded, h, w)
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(mag), 0, 255).astype(img.dtype)
    return mag.astype(img.dtype)
'''
        elif tiling > 0:
            return f'''import numpy as np
from numba import njit
{prange_import}

@njit(fastmath={fastmath}, parallel={parallel}, cache=True)
def _nb_sobel_tiled(padded, h, w):
    out = np.zeros((h, w), dtype=np.float32)
    block = {tiling}
    n_bi = (h + block - 1) // block
    n_bj = (w + block - 1) // block
    for bi_idx in {loop_fn}(n_bi):
        bi = bi_idx * block
        i_max = min(bi + block, h)
        for bj_idx in range(n_bj):
            bj = bj_idx * block
            j_max = min(bj + block, w)
            for i in range(bi, i_max):
                for j in range(bj, j_max):
                    gx = (
                        -1.0 * padded[i, j]     + 1.0 * padded[i, j+2]
                        -2.0 * padded[i+1, j]   + 2.0 * padded[i+1, j+2]
                        -1.0 * padded[i+2, j]   + 1.0 * padded[i+2, j+2]
                    )
                    gy = (
                        -1.0 * padded[i, j]   - 2.0 * padded[i, j+1]   - 1.0 * padded[i, j+2]
                        +1.0 * padded[i+2, j] + 2.0 * padded[i+2, j+1] + 1.0 * padded[i+2, j+2]
                    )
                    out[i, j] = np.sqrt(gx * gx + gy * gy)
    return out

def kernel_entry(img, **kwargs):
    if img.ndim == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float32)
    h, w = gray.shape
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    mag = _nb_sobel_tiled(padded, h, w)
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(mag), 0, 255).astype(img.dtype)
    return mag.astype(img.dtype)
'''
        elif quantized:
            return f'''import numpy as np
from numba import njit
{prange_import}

@njit(fastmath={fastmath}, parallel={parallel}, cache=True)
def _nb_sobel_quantized(padded, h, w):
    out = np.zeros((h, w), dtype=np.float32)
    for i in {loop_fn}(h):
        for j in range(w):
            gx = (
                -padded[i, j]     + padded[i, j+2]
                - 2 * padded[i+1, j]   + 2 * padded[i+1, j+2]
                - padded[i+2, j]   + padded[i+2, j+2]
            )
            gy = (
                -padded[i, j]   - 2 * padded[i, j+1]   - padded[i, j+2]
                +padded[i+2, j] + 2 * padded[i+2, j+1] + padded[i+2, j+2]
            )
            out[i, j] = np.sqrt(float(gx * gx + gy * gy))
    return out

def kernel_entry(img, **kwargs):
    if img.ndim == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float32)
    h, w = gray.shape
    padded = np.pad(gray.astype(np.int32), ((1, 1), (1, 1)), mode="edge")
    mag = _nb_sobel_quantized(padded, h, w)
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(mag), 0, 255).astype(img.dtype)
    return mag.astype(img.dtype)
'''
        elif loop_order == "col_row":
            return f'''import numpy as np
from numba import njit
{prange_import}

@njit(fastmath={fastmath}, parallel={parallel}, cache=True)
def _nb_sobel_col_row(padded, h, w):
    out = np.zeros((h, w), dtype=np.float32)
    for j in {loop_fn}(w):
        for i in range(h):
            gx = (
                -1.0 * padded[i, j]     + 1.0 * padded[i, j+2]
                -2.0 * padded[i+1, j]   + 2.0 * padded[i+1, j+2]
                -1.0 * padded[i+2, j]   + 1.0 * padded[i+2, j+2]
            )
            gy = (
                -1.0 * padded[i, j]   - 2.0 * padded[i, j+1]   - 1.0 * padded[i, j+2]
                +1.0 * padded[i+2, j] + 2.0 * padded[i+2, j+1] + 1.0 * padded[i+2, j+2]
            )
            out[i, j] = np.sqrt(gx * gx + gy * gy)
    return out

def kernel_entry(img, **kwargs):
    if img.ndim == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float32)
    h, w = gray.shape
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    mag = _nb_sobel_col_row(padded, h, w)
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(mag), 0, 255).astype(img.dtype)
    return mag.astype(img.dtype)
'''
        else:
            return f'''import numpy as np
from numba import njit
{prange_import}

@njit(fastmath={fastmath}, parallel={parallel}, cache=True)
def _nb_sobel(padded, h, w):
    # unroll factor: {unroll}
    out = np.zeros((h, w), dtype=np.float32)
    for i in {loop_fn}(h):
        for j in range(w):
            gx = (
                -1.0 * padded[i, j]     + 1.0 * padded[i, j+2]
                -2.0 * padded[i+1, j]   + 2.0 * padded[i+1, j+2]
                -1.0 * padded[i+2, j]   + 1.0 * padded[i+2, j+2]
            )
            gy = (
                -1.0 * padded[i, j]   - 2.0 * padded[i, j+1]   - 1.0 * padded[i, j+2]
                +1.0 * padded[i+2, j] + 2.0 * padded[i+2, j+1] + 1.0 * padded[i+2, j+2]
            )
            out[i, j] = np.sqrt(gx * gx + gy * gy)
    return out

def kernel_entry(img, **kwargs):
    if img.ndim == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float32)
    h, w = gray.shape
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    mag = _nb_sobel(padded, h, w)
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(mag), 0, 255).astype(img.dtype)
    return mag.astype(img.dtype)
'''

    elif kernel_name == "conv2d":
        return f'''import numpy as np
from numba import njit
{prange_import}

@njit(fastmath={fastmath}, parallel={parallel}, cache=True)
def _nb_conv2d(padded, flipped_k, h, w, kh, kw):
    out = np.zeros((h, w), dtype=np.float32)
    for i in {loop_fn}(h):
        for j in range(w):
            acc = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    acc += padded[i + ki, j + kj] * flipped_k[ki, kj]
            out[i, j] = acc
    return out

def kernel_entry(img, kernel=None, **kwargs):
    if kernel is None:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    flipped_k = np.ascontiguousarray(np.flip(kernel)).astype(np.float32)
    padded = np.pad(img.astype(np.float32), ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = _nb_conv2d(padded, flipped_k, img.shape[0], img.shape[1], kh, kw)
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(out), 0, 255).astype(img.dtype)
    return out.astype(img.dtype)
'''
    elif kernel_name == "fused_contrast_blur":
        if config.separable:
            return f'''import numpy as np
from numba import njit
{prange_import}

@njit(fastmath={fastmath}, cache=True)
def _make_gauss_1d(ksize, sigma):
    rad = ksize // 2
    k = np.empty(ksize, dtype=np.float32)
    tot = 0.0
    for i in range(ksize):
        x = i - rad
        v = np.exp(-0.5 * (x / sigma) ** 2)
        k[i] = v
        tot += v
    for i in range(ksize):
        k[i] /= tot
    return k

@njit(fastmath={fastmath}, parallel={parallel}, cache=True)
def _nb_fused_contrast_blur_sep(img, k1d, h, w, ksize, alpha, beta):
    pad = ksize // 2
    tmp = np.empty((h, w), dtype=np.float32)
    for i in {loop_fn}(h):
        for j in range(w):
            acc = 0.0
            for k in range(ksize):
                col = min(max(j + k - pad, 0), w - 1)
                p = float(img[i, col]) * alpha + beta
                p_c = 0.0 if p < 0.0 else (255.0 if p > 255.0 else p)
                acc += p_c * k1d[k]
            tmp[i, j] = acc
    out = np.empty((h, w), dtype=np.uint8)
    for i in {loop_fn}(h):
        for j in range(w):
            acc = 0.0
            for k in range(ksize):
                row = min(max(i + k - pad, 0), h - 1)
                acc += tmp[row, j] * k1d[k]
            val = acc + 0.5
            out[i, j] = 0 if val < 0.0 else (255 if val > 255.0 else int(val))
    return out

def kernel_entry(img, alpha=1.5, beta=10.0, ksize=5, sigma=1.0, **kwargs):
    k1d = _make_gauss_1d(ksize, float(sigma))
    h, w = img.shape[:2]
    return _nb_fused_contrast_blur_sep(img, k1d, h, w, ksize, float(alpha), float(beta))
'''
        elif config.tiling > 0:
            return f'''import numpy as np
from numba import njit
{prange_import}

@njit(fastmath={fastmath}, cache=True)
def _make_gauss_kernel(ksize, sigma):
    rad = ksize // 2
    k = np.empty((ksize, ksize), dtype=np.float32)
    total = 0.0
    for i in range(ksize):
        x = i - rad
        for j in range(ksize):
            y = j - rad
            val = np.exp(-(x*x + y*y) / (2.0 * sigma * sigma))
            k[i, j] = val
            total += val
    for i in range(ksize):
        for j in range(ksize):
            k[i, j] /= total
    return k

@njit(fastmath={fastmath}, parallel={parallel}, cache=True)
def _nb_fused_contrast_blur_tiled(img, kernel, h, w, ksize, alpha, beta):
    pad = ksize // 2
    out = np.zeros((h, w), dtype=np.uint8)
    tile_size = {config.tiling}
    n_ti = (h + tile_size - 1) // tile_size
    n_tj = (w + tile_size - 1) // tile_size
    for ti in {loop_fn}(n_ti):
        i_start = ti * tile_size
        i_end = min(i_start + tile_size, h)
        for tj in range(n_tj):
            j_start = tj * tile_size
            j_end = min(j_start + tile_size, w)
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    acc = 0.0
                    for ki in range(ksize):
                        for kj in range(ksize):
                            pi = min(max(i + ki - pad, 0), h - 1)
                            pj = min(max(j + kj - pad, 0), w - 1)
                            p = float(img[pi, pj]) * alpha + beta
                            p_c = 0.0 if p < 0.0 else (255.0 if p > 255.0 else p)
                            acc += p_c * kernel[ki, kj]
                    val_round = acc + 0.5
                    if val_round < 0.0:
                        out[i, j] = 0
                    elif val_round > 255.0:
                        out[i, j] = 255
                    else:
                        out[i, j] = int(val_round)
    return out

def kernel_entry(img, alpha=1.5, beta=10.0, ksize=5, sigma=1.0, **kwargs):
    kernel = _make_gauss_kernel(ksize, float(sigma))
    h, w = img.shape[:2]
    return _nb_fused_contrast_blur_tiled(img, kernel, h, w, ksize, float(alpha), float(beta))
'''
        else:
            return f'''import numpy as np
from numba import njit
{prange_import}

@njit(fastmath={fastmath}, cache=True)
def _make_gauss_kernel(ksize, sigma):
    rad = ksize // 2
    k = np.empty((ksize, ksize), dtype=np.float32)
    total = 0.0
    for i in range(ksize):
        x = i - rad
        for j in range(ksize):
            y = j - rad
            val = np.exp(-(x*x + y*y) / (2.0 * sigma * sigma))
            k[i, j] = val
            total += val
    for i in range(ksize):
        for j in range(ksize):
            k[i, j] /= total
    return k

@njit(fastmath={fastmath}, parallel={parallel}, cache=True)
def _nb_fused_contrast_blur(img, kernel, h, w, ksize, alpha, beta):
    # Fused single-pass: compute local contrast on the fly without intermediate allocation
    pad = ksize // 2
    out = np.zeros((h, w), dtype=np.uint8)
    for i in {loop_fn}(h):
        for j in range(w):
            acc = 0.0
            for ki in range(ksize):
                for kj in range(ksize):
                    # Clamp coordinates to replicate border
                    pi = min(max(i + ki - pad, 0), h - 1)
                    pj = min(max(j + kj - pad, 0), w - 1)
                    pix_val = float(img[pi, pj]) * alpha + beta
                    if pix_val < 0.0:
                        pix_val = 0.0
                    elif pix_val > 255.0:
                        pix_val = 255.0
                    acc += pix_val * kernel[ki, kj]
            val_round = acc + 0.5
            if val_round < 0.0:
                out[i, j] = 0
            elif val_round > 255.0:
                out[i, j] = 255
            else:
                out[i, j] = int(val_round)
    return out

def kernel_entry(img, alpha=1.5, beta=10.0, ksize=5, sigma=1.0, **kwargs):
    kernel = _make_gauss_kernel(ksize, float(sigma))
    h, w = img.shape[:2]
    return _nb_fused_contrast_blur(img, kernel, h, w, ksize, float(alpha), float(beta))
'''

    raise ValueError(f"Unknown kernel name: {kernel_name}")


def generate_python_loop_kernel(kernel_name: str, config: KernelConfig) -> str:
    """Generate pure Python nested-loop kernel (naive baseline)."""
    if kernel_name == "threshold":
        return r'''import numpy as np

def kernel_entry(img, thresh=128.0, maxval=255.0, **kwargs):
    h, w = img.shape
    out = np.empty((h, w), dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            out[i, j] = maxval if img[i, j] > thresh else 0
    return out
'''
    elif kernel_name == "gaussian_blur":
        return r'''import numpy as np

def kernel_entry(img, ksize=5, sigma=1.0, **kwargs):
    rad = ksize // 2
    ax = np.arange(-rad, rad + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    k = k / np.sum(k)

    h, w = img.shape[:2]
    padded = np.pad(img.astype(np.float32), ((rad, rad), (rad, rad)), mode="edge")
    out = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            val = 0.0
            for ki in range(ksize):
                for kj in range(ksize):
                    val += padded[i + ki, j + kj] * k[ki, kj]
            out[i, j] = val

    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(out), 0, 255).astype(img.dtype)
    return out.astype(img.dtype)
'''
    elif kernel_name == "sobel":
        return r'''import numpy as np
import math

def kernel_entry(img, **kwargs):
    if img.ndim == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float32)
    h, w = gray.shape
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    out = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            gx = (
                -padded[i, j] + padded[i, j+2]
                -2.0 * padded[i+1, j] + 2.0 * padded[i+1, j+2]
                -padded[i+2, j] + padded[i+2, j+2]
            )
            gy = (
                -padded[i, j] - 2.0 * padded[i, j+1] - padded[i, j+2]
                +padded[i+2, j] + 2.0 * padded[i+2, j+1] + padded[i+2, j+2]
            )
            out[i, j] = math.sqrt(gx * gx + gy * gy)

    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(out), 0, 255).astype(img.dtype)
    return out.astype(img.dtype)
'''
    elif kernel_name == "conv2d":
        return r'''import numpy as np

def kernel_entry(img, kernel=None, **kwargs):
    if kernel is None:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    flipped_k = np.flip(kernel)
    h, w = img.shape[:2]
    padded = np.pad(img.astype(np.float32), ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            s = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    s += padded[i + ki, j + kj] * flipped_k[ki, kj]
            out[i, j] = s

    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(out), 0, 255).astype(img.dtype)
    return out.astype(img.dtype)
'''
    elif kernel_name == "fused_contrast_blur":
        return r'''import numpy as np

def kernel_entry(img, alpha=1.5, beta=10.0, ksize=5, sigma=1.0, **kwargs):
    rad = ksize // 2
    ax = np.arange(-rad, rad + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    k = k / np.sum(k)

    h, w = img.shape[:2]
    out = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            val = 0.0
            for ki in range(ksize):
                for kj in range(ksize):
                    pi = min(max(i + ki - rad, 0), h - 1)
                    pj = min(max(j + kj - rad, 0), w - 1)
                    pix = float(img[pi, pj]) * alpha + beta
                    if pix < 0.0:
                        pix = 0.0
                    elif pix > 255.0:
                        pix = 255.0
                    val += pix * k[ki, kj]
            out[i, j] = val

    return np.clip(np.round(out), 0, 255).astype(np.uint8)
'''
    raise ValueError(f"Unknown kernel name: {kernel_name}")


def synthesize_candidate_code(kernel_name: str, config: KernelConfig) -> str:
    """Emit full Python code string for given kernel config."""
    if config.backend == "opencv":
        return generate_opencv_kernel(kernel_name, config)
    elif config.backend == "numpy":
        return generate_numpy_kernel(kernel_name, config)
    elif config.backend == "numba":
        return generate_numba_kernel(kernel_name, config)
    elif config.backend == "python_loops":
        return generate_python_loop_kernel(kernel_name, config)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")


# ==============================================================================
# SEARCH ALGORITHMS
# ==============================================================================

class CandidateGenerator:
    """Generates candidate kernels using Random Search, Multi-Armed Bandit, or Evolution."""

    BACKENDS = ["opencv", "numpy", "numba", "python_loops"]
    TILING_OPTIONS = [0, 16, 32, 64]
    LOOP_ORDERS = ["row_col", "col_row"]

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.cand_counter = 0

    def _next_id(self, prefix: str = "cand") -> str:
        self.cand_counter += 1
        return f"{prefix}_{self.cand_counter:04d}"

    def sample_random_config(self) -> KernelConfig:
        """Sample a uniformly random configuration from the search space."""
        backend = self.rng.choice(self.BACKENDS)
        tiling = self.rng.choice(self.TILING_OPTIONS) if backend == "numba" else 0
        parallel = self.rng.choice([True, False]) if backend == "numba" else False
        fastmath = self.rng.choice([True, False]) if backend == "numba" else False
        separable = self.rng.choice([True, False]) if backend in ("numpy", "numba") else False
        quantized = self.rng.choice([True, False])
        loop_order = self.rng.choice(self.LOOP_ORDERS)

        return KernelConfig(
            backend=backend,
            tiling=tiling,
            parallel=parallel,
            fastmath=fastmath,
            separable=separable,
            quantized=quantized,
            loop_order=loop_order,
        )

    def generate_candidate(self, kernel_name: str, config: KernelConfig, generation: int = 0) -> Candidate:
        """Create a candidate with synthesized source code."""
        code = synthesize_candidate_code(kernel_name, config)
        cid = self._next_id(prefix=f"{kernel_name}_{config.backend}")
        return Candidate(
            candidate_id=cid,
            kernel_name=kernel_name,
            config=config,
            source_code=code,
            generation=generation,
        )

    # 1. Random Search Strategy
    def generate_random_batch(self, kernel_name: str, n: int) -> List[Candidate]:
        """Generate N random candidates using fixed reproducible seed."""
        # Always include baseline OpenCV and naive Python in the first batch
        candidates: List[Candidate] = []
        if n > 0:
            candidates.append(self.generate_candidate(kernel_name, KernelConfig(backend="opencv")))
        if n > 1:
            candidates.append(self.generate_candidate(kernel_name, KernelConfig(backend="numpy")))
        if n > 2:
            candidates.append(self.generate_candidate(kernel_name, KernelConfig(backend="numba", fastmath=True, parallel=True)))
        
        while len(candidates) < n:
            cfg = self.sample_random_config()
            candidates.append(self.generate_candidate(kernel_name, cfg))
        return candidates

    # 2. Multi-Armed Bandit (Epsilon-Greedy Strategy)
    def sample_bandit_arm(
        self,
        kernel_name: str,
        arm_history: Dict[str, List[float]],
        epsilon: float = 0.25,
    ) -> Candidate:
        """Choose configuration using Epsilon-Greedy bandit across backend arms."""
        all_arms = ["opencv", "numpy_separable", "numpy_standard", "numba_parallel", "numba_fastmath", "numba_tiled"]

        # Exploration: pick random arm
        if self.rng.random() < epsilon or not arm_history:
            chosen_arm = self.rng.choice(all_arms)
        else:
            # Exploitation: pick arm with highest average score
            best_score = -float("inf")
            chosen_arm = all_arms[0]
            for arm in all_arms:
                scores = arm_history.get(arm, [])
                avg_score = (sum(scores) / len(scores)) if scores else 0.0
                if avg_score > best_score:
                    best_score = avg_score
                    chosen_arm = arm

        # Map chosen arm to config
        if chosen_arm == "opencv":
            cfg = KernelConfig(backend="opencv")
        elif chosen_arm == "numpy_separable":
            cfg = KernelConfig(backend="numpy", separable=True)
        elif chosen_arm == "numpy_standard":
            cfg = KernelConfig(backend="numpy", separable=False)
        elif chosen_arm == "numba_parallel":
            cfg = KernelConfig(backend="numba", parallel=True, fastmath=True)
        elif chosen_arm == "numba_fastmath":
            cfg = KernelConfig(backend="numba", parallel=False, fastmath=True)
        else: # numba_tiled
            cfg = KernelConfig(backend="numba", tiling=32, fastmath=True)

        cand = self.generate_candidate(kernel_name, cfg)
        cand.config.extra_flags["arm"] = chosen_arm
        return cand

    # 3. Evolutionary Search Strategy (Mutation & Crossover)
    def mutate(self, config: KernelConfig, mutation_rate: float = 0.35) -> KernelConfig:
        """Randomly perturb configuration genes."""
        c = copy.deepcopy(config)
        if self.rng.random() < mutation_rate:
            c.backend = self.rng.choice(self.BACKENDS)
        if self.rng.random() < mutation_rate:
            c.tiling = self.rng.choice(self.TILING_OPTIONS) if c.backend == "numba" else 0
        if self.rng.random() < mutation_rate:
            c.parallel = not c.parallel if c.backend == "numba" else False
        if self.rng.random() < mutation_rate:
            c.fastmath = not c.fastmath if c.backend == "numba" else False
        if self.rng.random() < mutation_rate:
            c.separable = not c.separable
        if self.rng.random() < mutation_rate:
            c.quantized = not c.quantized
        if self.rng.random() < mutation_rate:
            c.loop_order = "col_row" if c.loop_order == "row_col" else "row_col"
        return c

    def crossover(self, parent_a: KernelConfig, parent_b: KernelConfig) -> KernelConfig:
        """Combine genes from two parents."""
        return KernelConfig(
            backend=parent_a.backend if self.rng.random() < 0.5 else parent_b.backend,
            tiling=parent_a.tiling if self.rng.random() < 0.5 else parent_b.tiling,
            parallel=parent_a.parallel if self.rng.random() < 0.5 else parent_b.parallel,
            fastmath=parent_a.fastmath if self.rng.random() < 0.5 else parent_b.fastmath,
            separable=parent_a.separable if self.rng.random() < 0.5 else parent_b.separable,
            quantized=parent_a.quantized if self.rng.random() < 0.5 else parent_b.quantized,
            loop_order=parent_a.loop_order if self.rng.random() < 0.5 else parent_b.loop_order,
        )

    def evolve_population(
        self,
        population: List[Candidate],
        population_size: int,
        generation: int,
    ) -> List[Candidate]:
        """Evolve population of evaluated candidates to produce the next generation."""
        # Sort candidates by score descending (higher score = faster)
        valid = [c for c in population if c.score is not None and c.score > 0]
        if not valid:
            return self.generate_random_batch(population[0].kernel_name, population_size)

        valid.sort(key=lambda c: c.score or 0.0, reverse=True)
        kernel_name = valid[0].kernel_name

        # Elitism: retain top 2 unchanged
        next_gen: List[Candidate] = []
        elite_count = min(2, len(valid))
        for i in range(elite_count):
            elite_cand = copy.deepcopy(valid[i])
            elite_cand.generation = generation
            next_gen.append(elite_cand)

        # Breed remainder
        while len(next_gen) < population_size:
            # Tournament selection (size 2)
            p1 = max(self.rng.sample(valid, min(2, len(valid))), key=lambda c: c.score or 0.0)
            p2 = max(self.rng.sample(valid, min(2, len(valid))), key=lambda c: c.score or 0.0)

            child_cfg = self.crossover(p1.config, p2.config)
            child_cfg = self.mutate(child_cfg)
            next_gen.append(self.generate_candidate(kernel_name, child_cfg, generation=generation))

        return next_gen

    # 4. Exhaustive Search Space Enumeration
    def enumerate_all_discrete_configs(self, kernel_name: str) -> List[KernelConfig]:
        """Exhaustively enumerate all valid discrete parameter configurations."""
        configs: List[KernelConfig] = []

        # OpenCV baseline
        configs.append(KernelConfig(backend="opencv"))

        # Pure Python loop baseline
        configs.append(KernelConfig(backend="python_loops"))

        # NumPy variants
        for separable in ([True, False] if kernel_name == "gaussian_blur" else [False]):
            for quantized in [False, True]:
                configs.append(KernelConfig(
                    backend="numpy",
                    separable=separable,
                    quantized=quantized
                ))

        # Numba variants
        is_sep_capable = (kernel_name in ("gaussian_blur", "fused_contrast_blur"))
        for fastmath in [True, False]:
            for parallel in [True, False]:
                for tiling in [0, 16, 32]:
                    for order in (["row_col", "col_row"] if tiling == 0 else ["row_col"]):
                        for sep in ([True, False] if (is_sep_capable and tiling == 0) else [False]):
                            configs.append(KernelConfig(
                                backend="numba",
                                fastmath=fastmath,
                                parallel=parallel,
                                tiling=tiling,
                                loop_order=order,
                                separable=sep,
                            ))

        return configs

    def enumerate_large_discrete_configs(self, kernel_name: str) -> List[KernelConfig]:
        """Enumerate extended search space (486 discrete configurations) including OpenCV thread counts.

        Permutes tile dimensions (0, 8, 16, 32, 64), loop unrolling factors (1, 2, 4),
        parallel schedules, fastmath flags, layout orderings (C vs Fortran contiguous),
        separable decomposition, quantization modes, and OpenCV thread configurations (1, 2, 4, 8).
        """
        configs: List[KernelConfig] = []
        # OpenCV baseline variants with thread search parameter
        for th in [1, 2, 4, 8]:
            configs.append(KernelConfig(backend="opencv", extra_flags={"threads": th}))

        # Pure Python baseline and NumPy baseline
        configs.append(KernelConfig(backend="python_loops"))
        configs.append(KernelConfig(backend="numpy"))

        is_sep = (kernel_name in ("sobel", "gaussian_blur", "fused_contrast_blur"))
        tiling_shapes = [0, 8, 16, 32, 64]
        unroll_factors = [1, 2, 4]
        parallel_opts = [True, False]
        fastmath_opts = [True, False]
        layout_opts = ["row_col", "col_row"]
        quant_opts = [False, True]

        for t in tiling_shapes:
            for unroll in unroll_factors:
                for p in parallel_opts:
                    for fm in fastmath_opts:
                        for lo in layout_opts:
                            for q in quant_opts:
                                for sep in ([True, False] if is_sep else [False]):
                                    configs.append(KernelConfig(
                                        backend="numba",
                                        tiling=t,
                                        parallel=p,
                                        fastmath=fm,
                                        separable=sep,
                                        quantized=q,
                                        loop_order=lo,
                                        extra_flags={"unroll": unroll}
                                    ))
        return configs
