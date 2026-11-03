"""Tests verifying reference kernels against OpenCV built-ins.

Documents and tests numerical tolerances:
1. Gaussian Blur: ref vs cv2.GaussianBlur (borderType=cv2.BORDER_REPLICATE).
   Tolerance: Max absolute error <= 1 intensity level (due to fixed-point vs float32 kernel rounding).
2. Threshold: ref vs cv2.threshold (cv2.THRESH_BINARY).
   Tolerance: Exact match (Max absolute error == 0).
3. Sobel: ref vs cv2.Sobel (borderType=cv2.BORDER_REPLICATE).
   Tolerance: Max absolute error <= 1.0 (gradient magnitude calculation rounding).
4. Conv2d: ref vs cv2.filter2D (borderType=cv2.BORDER_REPLICATE).
   Tolerance: Max absolute error <= 1.0 (filter rounding).
"""

import pytest
import numpy as np
import cv2

from seae.references import ref_gaussian_blur, ref_threshold, ref_sobel, ref_conv2d


def test_ref_gaussian_blur_vs_opencv():
    """Verify ref_gaussian_blur against cv2.GaussianBlur with BORDER_REPLICATE."""
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    ksize = 5
    sigma = 1.0

    ref_out = ref_gaussian_blur(img, ksize=ksize, sigma=sigma)
    cv_out = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

    max_diff = np.max(np.abs(ref_out.astype(np.float32) - cv_out.astype(np.float32)))
    # OpenCV uses fixed-point integers for 8-bit convolution coefficients, so +/- 1 gray level is expected
    assert max_diff <= 1.0, f"Gaussian blur discrepancy too high: max_diff={max_diff}"


def test_ref_threshold_vs_opencv():
    """Verify ref_threshold against cv2.threshold (exact match expected)."""
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    thresh = 128
    maxval = 255

    ref_out = ref_threshold(img, thresh=thresh, maxval=maxval)
    _, cv_out = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)

    max_diff = np.max(np.abs(ref_out.astype(np.float32) - cv_out.astype(np.float32)))
    assert max_diff == 0.0, f"Threshold must be exact match, got max_diff={max_diff}"


def test_ref_conv2d_vs_opencv():
    """Verify ref_conv2d against cv2.filter2D with flipped kernel and BORDER_REPLICATE."""
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    ref_out = ref_conv2d(img, kernel=kernel)
    # cv2.filter2D computes cross-correlation; with flipped kernel it's convolution
    flipped_k = np.flip(kernel)
    # OpenCV filter2D with CV_32F then clamped to uint8
    filtered = cv2.filter2D(img, cv2.CV_32F, flipped_k, borderType=cv2.BORDER_REPLICATE)
    cv_out = np.clip(np.round(filtered), 0, 255).astype(np.uint8)

    max_diff = np.max(np.abs(ref_out.astype(np.float32) - cv_out.astype(np.float32)))
    assert max_diff <= 1.0, f"Conv2D discrepancy too high: max_diff={max_diff}"


def test_ref_sobel_vs_opencv():
    """Verify ref_sobel against cv2.Sobel gradient magnitude."""
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    ref_out = ref_sobel(img)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    cv_mag = np.clip(np.round(np.hypot(gx, gy)), 0, 255).astype(np.uint8)

    max_diff = np.max(np.abs(ref_out.astype(np.float32) - cv_mag.astype(np.float32)))
    assert max_diff <= 1.0, f"Sobel magnitude discrepancy too high: max_diff={max_diff}"
