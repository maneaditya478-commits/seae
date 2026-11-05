"""Tests for stricter verifier tolerances and parameter-aware caching."""

import hashlib
import inspect
import numpy as np
import pytest

from seae.verifier import Verifier, ToleranceConfig, calculate_psnr
from seae.task_analyzer import TaskAnalyzer, TaskSpec
from seae.references import get_reference_kernel


def test_one_wrong_pixel_must_fail():
    """A single corrupted pixel (e.g. exceeding 3x limit) must fail despite high overall PSNR."""
    # Candidate code: golden threshold, but tampers with exactly one pixel: out[0, 0] = 255 - out[0, 0]
    corrupted_code = """
import numpy as np

def kernel_entry(img, thresh=128.0, maxval=255.0, **kwargs):
    out = np.where(img > thresh, maxval, 0).astype(img.dtype)
    # Corrupt exactly one pixel
    out[0, 0] = 255 if out[0, 0] == 0 else 0
    return out
"""
    spec = TaskAnalyzer.analyze(kernel_name="threshold", size_str="48x48", dtype_str="uint8")
    verifier = Verifier()
    report = verifier.verify_candidate("cand_one_wrong_pixel", corrupted_code, spec)

    # Must be rejected under the new verifier
    assert report.all_passed is False
    assert "Tolerance failed" in report.rejection_reason or "Failed on" in report.rejection_reason


def test_uniform_error_of_one_must_pass():
    """Uniform error of 1 on quantized candidate must pass (error <= 2.0, PSNR ~48 dB)."""
    # Candidate code: golden threshold, but with uniform offset of +1 (clipped to 255)
    offset_code = """
import numpy as np

def kernel_entry(img, thresh=128.0, maxval=255.0, **kwargs):
    base = np.where(img > thresh, maxval, 0).astype(np.int32)
    # Shift by 1
    shifted = np.clip(base + 1, 0, 255).astype(img.dtype)
    return shifted
"""
    spec = TaskAnalyzer.analyze(kernel_name="threshold", size_str="48x48", dtype_str="uint8")
    verifier = Verifier()
    report = verifier.verify_candidate("cand_uniform_one", offset_code, spec)

    assert report.all_passed is True
    assert report.rejection_reason is None
    # Verify metrics recorded
    for res in report.test_case_results:
        metrics = res["metrics"]
        assert metrics["max_abs_error"] <= 1.0
        assert metrics["fraction_exceeding_tol"] == 0.0
        assert metrics["passed"] is True


def test_quantized_excess_percentage_rejected():
    """Reject candidate when more than 0.1% of pixels exceed the max error limit."""
    # In a 48x48 image (2304 pixels), 0.1% is ~2.3 pixels. If 5 pixels have error 2.5 (> 2.0, <= 6.0), it must fail.
    excess_code = """
import numpy as np

def kernel_entry(img, thresh=128.0, maxval=255.0, **kwargs):
    out = np.where(img > thresh, maxval, 0).astype(np.int32)
    # Introduce an error of 3 (which is > max_allowed 2.0, but <= 3x 6.0) on 5 pixels (> 0.1% of 2304)
    out[0, 0] = np.clip(out[0, 0] + 3, 0, 255)
    out[0, 1] = np.clip(out[0, 1] + 3, 0, 255)
    out[0, 2] = np.clip(out[0, 2] + 3, 0, 255)
    out[0, 3] = np.clip(out[0, 3] + 3, 0, 255)
    out[0, 4] = np.clip(out[0, 4] + 3, 0, 255)
    return out.astype(img.dtype)
"""
    spec = TaskAnalyzer.analyze(kernel_name="threshold", size_str="48x48", dtype_str="uint8")
    verifier = Verifier()
    report = verifier.verify_candidate("cand_excess_pct", excess_code, spec)

    assert report.all_passed is False
    assert "pixels exceeded max_err limit" in report.rejection_reason


def test_float_candidate_requires_both_mae_and_psnr():
    """Floating point candidates require BOTH max_abs_error <= limit AND PSNR >= limit."""
    # Candidate with small distributed noise that fails floating-point max_abs_tol_float (1e-3)
    float_noisy_code = """
import numpy as np

def kernel_entry(img, **kwargs):
    # Error of 0.01 (exceeds 1e-3, but PSNR will still be > 40 dB)
    return (img.astype(np.float32) + 0.01).astype(img.dtype)
"""
    spec = TaskSpec(kernel_name="conv2d", height=48, width=48, channels=1, dtype_str="float32")
    verifier = Verifier()
    report = verifier.verify_candidate("cand_float_noisy", float_noisy_code, spec)

    assert report.all_passed is False
    assert "Tolerance failed for floating_point" in report.rejection_reason


def test_cache_key_includes_kernel_params_and_reference_hash():
    """Ensure verifier cache discriminates by kernel parameters and reference implementation hash."""
    code = """
import numpy as np
def kernel_entry(img, thresh=128.0, maxval=255.0, **kwargs):
    return np.where(img > thresh, maxval, 0).astype(img.dtype)
"""
    import dataclasses
    verifier = Verifier()
    base_spec = TaskAnalyzer.analyze(kernel_name="threshold", size_str="48x48", dtype_str="uint8")
    spec1 = dataclasses.replace(base_spec, thresh=100.0)
    spec2 = dataclasses.replace(base_spec, thresh=200.0)

    verifier.verify_candidate("cand_c1", code, spec1)
    verifier.verify_candidate("cand_c2", code, spec2)

    # Verifier cache should have 2 distinct entries because thresh differed
    assert len(verifier._verification_cache) == 2
    keys = list(verifier._verification_cache.keys())
    assert keys[0] != keys[1]
    # Check that thresh is in the key
    assert 100.0 in keys[0]
    assert 200.0 in keys[1]
    # Check that reference hash is in the key
    ref_fn = get_reference_kernel("threshold")
    ref_hash = hashlib.sha256(inspect.getsource(ref_fn).encode("utf-8")).hexdigest()[:16]
    assert keys[0][-1] == ref_hash
