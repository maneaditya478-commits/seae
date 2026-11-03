"""Tests for Milestone 1: references, task analyzer, hardware manager, sandbox, and verifier."""

import pytest
import numpy as np
import time

from seae.task_analyzer import TaskAnalyzer, parse_resolution
from seae.hardware_manager import HardwareManager
from seae.references import ref_gaussian_blur, ref_threshold, ref_conv2d, ref_sobel
from seae.sandbox import Sandbox, SandboxResult
from seae.verifier import Verifier, calculate_psnr


def test_parse_resolution():
    h, w = parse_resolution("1920x1080")
    assert (w, h) == (1920, 1080)
    h, w = parse_resolution("512X512")
    assert (w, h) == (512, 512)
    with pytest.raises(ValueError):
        parse_resolution("invalid_size")


def test_task_analyzer():
    spec = TaskAnalyzer.analyze(kernel_name="gaussian_blur", size_str="640x480", dtype_str="uint8")
    assert spec.height == 480
    assert spec.width == 640
    assert spec.shape == (480, 640)
    assert spec.total_pixels == 480 * 640
    assert spec.memory_bytes == 480 * 640


def test_hardware_manager():
    profile = HardwareManager.get_hardware_profile()
    assert profile.logical_cores >= 1
    assert profile.ram_gb > 0
    assert len(profile.fingerprint) > 0
    assert isinstance(profile.simd_features, list)


def test_references_execution():
    img = np.random.randint(0, 255, size=(32, 32), dtype=np.uint8)
    
    blur_out = ref_gaussian_blur(img, ksize=3, sigma=1.0)
    assert blur_out.shape == (32, 32)
    assert blur_out.dtype == np.uint8

    thresh_out = ref_threshold(img, thresh=128, maxval=255)
    assert thresh_out.shape == (32, 32)
    assert set(np.unique(thresh_out)).issubset({0, 255})

    sobel_out = ref_sobel(img)
    assert sobel_out.shape == (32, 32)

    conv_out = ref_conv2d(img)
    assert conv_out.shape == (32, 32)


def test_sandbox_success():
    code = """
import numpy as np
def kernel_entry(img, **kwargs):
    return img.copy()
"""
    sb = Sandbox(timeout_seconds=5.0)
    inp = np.ones((10, 10), dtype=np.uint8)
    res = sb.run_candidate(code, inp)
    assert res.success is True
    assert res.output_array is not None
    assert np.array_equal(res.output_array, inp)
    assert res.execution_time_ms > 0


def test_sandbox_timeout():
    code = """
import time
def kernel_entry(img, **kwargs):
    time.sleep(3.0)
    return img
"""
    sb = Sandbox(timeout_seconds=0.6)
    inp = np.ones((5, 5), dtype=np.uint8)
    res = sb.run_candidate(code, inp)
    assert res.success is False
    assert res.timed_out is True
    assert "timed out" in res.error_message.lower()


def test_sandbox_network_blocking():
    code = """
import socket
def kernel_entry(img, **kwargs):
    s = socket.socket()
    return img
"""
    sb = Sandbox(timeout_seconds=4.0)
    inp = np.ones((5, 5), dtype=np.uint8)
    res = sb.run_candidate(code, inp)
    assert res.success is False
    assert "network" in res.error_message.lower() or "prohibited" in res.error_message.lower()


def test_sandbox_syntax_error():
    code = """
def kernel_entry(img, **kwargs)
    this is invalid syntax
"""
    sb = Sandbox(timeout_seconds=4.0)
    inp = np.ones((5, 5), dtype=np.uint8)
    res = sb.run_candidate(code, inp)
    assert res.success is False
    assert res.error_message is not None


def test_verifier_pass():
    # Pass an exact identity / threshold implementation
    valid_threshold_code = """
import numpy as np
def kernel_entry(img, thresh=128.0, maxval=255.0, **kwargs):
    return np.where(img > thresh, maxval, 0).astype(img.dtype)
"""
    verifier = Verifier()
    spec = TaskAnalyzer.analyze(kernel_name="threshold", size_str="64x64", dtype_str="uint8")
    report = verifier.verify_candidate("cand_thresh_ok", valid_threshold_code, spec)
    assert report.all_passed is True
    assert report.rejection_reason is None


def test_verifier_rejects_wrong_calculation():
    # Incorrect kernel that returns all zeros
    bad_code = """
import numpy as np
def kernel_entry(img, thresh=128.0, maxval=255.0, **kwargs):
    return np.zeros_like(img)
"""
    verifier = Verifier()
    spec = TaskAnalyzer.analyze(kernel_name="threshold", size_str="64x64", dtype_str="uint8")
    report = verifier.verify_candidate("cand_bad_calc", bad_code, spec)
    assert report.all_passed is False
    assert "Error tolerance exceeded" in report.rejection_reason or "Failed" in report.rejection_reason


def test_psnr_identical():
    a = np.ones((10, 10), dtype=np.uint8) * 100
    b = a.copy()
    assert calculate_psnr(a, b) == float("inf")
