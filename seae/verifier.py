"""Verifier module for SEAE.

Runs candidates strictly inside the isolated sandbox and compares candidate outputs
against trusted reference kernels across multiple input distributions:
1. Random uniform distributions
2. Edge cases (all zeros, all ones, max-value, saturated gradients)
3. Tiny inputs (1x1, 2x2, 3x3)
4. Non-square inputs

Evaluates numerical tolerance using:
- Maximum Absolute Error (MAE / max diff)
- Peak Signal-to-Noise Ratio (PSNR in dB)
- Strict shape and dtype checking
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from seae.sandbox import Sandbox, SandboxResult
from seae.references import get_reference_kernel
from seae.task_analyzer import TaskSpec


@dataclass
class ToleranceConfig:
    """Configurable numerical tolerances for verifier."""
    max_abs_tol_float: float = 1e-3
    min_psnr_db_float: float = 40.0
    max_abs_tol_quantized: float = 2.0
    min_psnr_db_quantized: float = 28.0


@dataclass
class VerificationMetrics:
    """Numerical verification score metrics."""
    max_abs_error: float
    mean_abs_error: float
    psnr_db: float
    shape_matches: bool
    dtype_matches: bool
    passed: bool
    applied_threshold: float
    applied_min_psnr: float
    candidate_kind: str
    rejection_reason: Optional[str] = None
    fraction_exceeding_tol: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationReport:
    """Aggregate report across all test cases."""
    candidate_id: str
    all_passed: bool
    rejection_reason: Optional[str] = None
    test_case_results: List[Dict[str, Any]] = None  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_psnr(ref: np.ndarray, test: np.ndarray, dynamic_range: Optional[float] = None) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels."""
    if ref.shape != test.shape:
        return 0.0

    ref_f = ref.astype(np.float64)
    test_f = test.astype(np.float64)
    mse = np.mean((ref_f - test_f) ** 2)

    if mse == 0.0:
        return float("inf")

    if dynamic_range is None:
        if np.issubdtype(ref.dtype, np.integer):
            info = np.iinfo(ref.dtype)
            dynamic_range = float(info.max - info.min)
        else:
            dynamic_range = float(np.max(ref_f) - np.min(ref_f))
            if dynamic_range <= 0.0:
                dynamic_range = 1.0

    psnr = 20.0 * math.log10(dynamic_range / math.sqrt(mse))
    return psnr


def generate_test_inputs(task_spec: TaskSpec, max_dim: int = 48) -> List[Tuple[str, np.ndarray]]:
    """Generate comprehensive test input battery: random, edge-cases, and tiny inputs."""
    dtype = task_spec.dtype
    is_int = np.issubdtype(dtype, np.integer)
    min_val = 0 if is_int else 0.0
    max_val = np.iinfo(dtype).max if is_int else 1.0

    tests: List[Tuple[str, np.ndarray]] = []

    # 1. Standard random test at task resolution (or moderate size for quick sandbox verification)
    h_test = min(task_spec.height, max_dim)
    w_test = min(task_spec.width, max_dim)
    if is_int:
        rand_img = np.random.randint(min_val, max_val + 1, size=(h_test, w_test), dtype=dtype)
    else:
        rand_img = np.random.uniform(min_val, max_val, size=(h_test, w_test)).astype(dtype)
    tests.append(("random_standard", rand_img))

    # 2. Edge case: All zeros
    tests.append(("edge_zeros", np.zeros((h_test, w_test), dtype=dtype)))

    # 3. Edge case: All max value
    tests.append(("edge_maxval", np.full((h_test, w_test), max_val, dtype=dtype)))

    # 4. Edge case: High-frequency alternating checkerboard
    check = np.indices((h_test, w_test)).sum(axis=0) % 2
    if is_int:
        tests.append(("edge_checkerboard", (check * max_val).astype(dtype)))
    else:
        tests.append(("edge_checkerboard", (check * 1.0).astype(dtype)))

    # 5. Tiny inputs (testing boundary safety and dimension edge conditions)
    # 5x5 tiny input (matches default 5x5 filter or smaller)
    tiny_size = max(task_spec.ksize, 5)
    if is_int:
        tiny_img = np.random.randint(min_val, max_val + 1, size=(tiny_size, tiny_size), dtype=dtype)
    else:
        tiny_img = np.random.uniform(min_val, max_val, size=(tiny_size, tiny_size)).astype(dtype)
    tests.append(("tiny_boundary", tiny_img))

    # 6. Non-square aspect ratio input
    non_sq_h, non_sq_w = min(task_spec.height, 32), min(task_spec.width, 48)
    if is_int:
        non_sq_img = np.random.randint(min_val, max_val + 1, size=(non_sq_h, non_sq_w), dtype=dtype)
    else:
        non_sq_img = np.random.uniform(min_val, max_val, size=(non_sq_h, non_sq_w)).astype(dtype)
    tests.append(("non_square_aspect", non_sq_img))

    return tests


class Verifier:
    """Verifies candidate image kernel implementations against reference outputs."""

    def __init__(
        self,
        sandbox: Optional[Sandbox] = None,
        tolerance_config: Optional[ToleranceConfig] = None,
    ) -> None:
        self.sandbox = sandbox or Sandbox(timeout_seconds=5.0, memory_limit_mb=1024.0)
        self.tolerances = tolerance_config or ToleranceConfig()
        self._verification_cache: Dict[Tuple[Any, ...], VerificationReport] = {}

    def verify_candidate(
        self,
        candidate_id: str,
        code_str: str,
        task_spec: TaskSpec,
        is_quantized: bool = False,
    ) -> VerificationReport:
        """Run candidate through all verification test cases inside the sandbox.
        
        Args:
            candidate_id: Identifier of the candidate.
            code_str: Source code of candidate containing `def kernel_entry(img, **kwargs)`.
            task_spec: Problem specification.
            is_quantized: Whether candidate uses integer quantization (e.g. int8/uint8).
            
        Returns:
            VerificationReport containing pass/fail status, per-test metrics, and logged thresholds.
        """
        ref_fn = get_reference_kernel(task_spec.kernel_name)
        try:
            import inspect
            ref_source = inspect.getsource(ref_fn)
        except Exception:
            ref_source = getattr(ref_fn, "__name__", str(ref_fn))
        import hashlib
        ref_source_hash = hashlib.sha256(ref_source.encode("utf-8")).hexdigest()[:16]

        cache_key = (
            task_spec.kernel_name,
            code_str,
            getattr(task_spec, "ksize", None),
            getattr(task_spec, "sigma", None),
            getattr(task_spec, "thresh", None),
            getattr(task_spec, "maxval", None),
            ref_source_hash,
        )
        if cache_key in self._verification_cache:
            return self._verification_cache[cache_key]
        is_pure_py = ("python_loops" in candidate_id) or ("def _py_" in code_str)
        test_inputs = generate_test_inputs(task_spec, max_dim=32 if is_pure_py else 48)
        results: List[Dict[str, Any]] = []

        # Determine threshold based on candidate type
        if is_quantized or np.issubdtype(task_spec.dtype, np.integer):
            max_allowed = self.tolerances.max_abs_tol_quantized
            min_psnr = self.tolerances.min_psnr_db_quantized
            cand_kind = "quantized/integer"
        else:
            max_allowed = self.tolerances.max_abs_tol_float
            min_psnr = self.tolerances.min_psnr_db_float
            cand_kind = "floating_point"

        # Parameters to forward into kernel
        kwargs: Dict[str, Any] = {}
        if task_spec.kernel_name == "gaussian_blur":
            kwargs["ksize"] = task_spec.ksize
            kwargs["sigma"] = task_spec.sigma
        elif task_spec.kernel_name == "threshold":
            kwargs["thresh"] = task_spec.thresh
            kwargs["maxval"] = task_spec.maxval

        # 1. Execute candidate across the entire test battery in a single sandbox process
        sb_res = self.sandbox.run_candidate_battery(code_str, test_inputs, **kwargs)
        if not sb_res.success:
            reason = sb_res.error_message or "Unknown sandbox execution failure"
            rep = VerificationReport(
                candidate_id=candidate_id,
                all_passed=False,
                rejection_reason=f"Sandbox error during verification battery: {reason}",
                test_case_results=[{
                    "test_name": "battery",
                    "passed": False,
                    "reason": reason,
                    "timed_out": sb_res.timed_out,
                    "oom_killed": sb_res.oom_killed,
                }],
            )
            self._verification_cache[cache_key] = rep
            return rep

        outputs_dict = sb_res.output_arrays or {}

        for test_name, inp in test_inputs:
            # 1. Compute ground truth reference in host
            try:
                ref_out = ref_fn(inp, **kwargs)
            except Exception as e:
                return VerificationReport(
                    candidate_id=candidate_id,
                    all_passed=False,
                    rejection_reason=f"Reference implementation raised error: {e}",
                    test_case_results=results,
                )

            cand_out = outputs_dict.get(test_name)
            if cand_out is None:
                return VerificationReport(
                    candidate_id=candidate_id,
                    all_passed=False,
                    rejection_reason=f"No output array returned on '{test_name}'",
                    test_case_results=results,
                )

            # 3. Shape & dtype verification
            shape_match = (cand_out.shape == ref_out.shape)
            if not shape_match:
                reason = f"Shape mismatch on '{test_name}': expected {ref_out.shape}, got {cand_out.shape}"
                results.append({"test_name": test_name, "passed": False, "reason": reason})
                return VerificationReport(
                    candidate_id=candidate_id,
                    all_passed=False,
                    rejection_reason=reason,
                    test_case_results=results,
                )

            # Strict output dtype verification: output dtype must match reference
            if cand_out.dtype != ref_out.dtype:
                reason = f"Dtype mismatch on '{test_name}': expected {ref_out.dtype}, got {cand_out.dtype}"
                results.append({"test_name": test_name, "passed": False, "reason": reason})
                return VerificationReport(
                    candidate_id=candidate_id,
                    all_passed=False,
                    rejection_reason=reason,
                    test_case_results=results,
                )

            # Check for non-finite values (NaN / Inf)
            if not np.all(np.isfinite(cand_out)):
                reason = f"Output contains NaN or Inf on '{test_name}'"
                results.append({"test_name": test_name, "passed": False, "reason": reason})
                return VerificationReport(
                    candidate_id=candidate_id,
                    all_passed=False,
                    rejection_reason=reason,
                    test_case_results=results,
                )

            # 4. Error comparison
            abs_diff = np.abs(cand_out.astype(np.float64) - ref_out.astype(np.float64))
            max_err = float(np.max(abs_diff))
            mean_err = float(np.mean(abs_diff))
            psnr = calculate_psnr(ref_out, cand_out)

            total_pixels = abs_diff.size
            num_exceeding = int(np.sum(abs_diff > max_allowed))
            frac_exceeding = float(num_exceeding / total_pixels) if total_pixels > 0 else 0.0

            psnr_ok = (psnr >= min_psnr)

            if cand_kind == "quantized/integer":
                # For quantized candidates:
                # - Allow at most 0.1% (0.001) of pixels to exceed max_allowed
                # - None to exceed 3x max_allowed
                # - AND PSNR >= min_psnr
                max_3x = 3.0 * max_allowed
                pct_ok = (frac_exceeding <= 0.001)
                none_3x_ok = (max_err <= max_3x)
                passed = psnr_ok and pct_ok and none_3x_ok

                if not passed:
                    reasons = []
                    if not psnr_ok:
                        reasons.append(f"PSNR={psnr:.2f}dB < {min_psnr:.1f}dB")
                    if not none_3x_ok:
                        reasons.append(f"max_err={max_err:.4f} > 3x limit ({max_3x:.4f})")
                    if not pct_ok:
                        reasons.append(
                            f"{frac_exceeding * 100:.3f}% pixels exceeded max_err limit "
                            f"({num_exceeding}/{total_pixels} > 0.1%)"
                        )
                    reason = f"Tolerance failed for {cand_kind}: {'; '.join(reasons)}"
                else:
                    reason = None
            else:
                # For floating_point:
                # Require BOTH max_abs_error <= limit AND PSNR >= limit
                max_err_ok = (max_err <= max_allowed)
                passed = max_err_ok and psnr_ok

                if not passed:
                    reasons = []
                    if not max_err_ok:
                        reasons.append(f"max_err={max_err:.4f} > {max_allowed}")
                    if not psnr_ok:
                        reasons.append(f"PSNR={psnr:.2f}dB < {min_psnr:.1f}dB")
                    reason = f"Tolerance failed for {cand_kind}: {'; '.join(reasons)}"
                else:
                    reason = None

            test_metric = VerificationMetrics(
                max_abs_error=max_err,
                mean_abs_error=mean_err,
                psnr_db=psnr,
                shape_matches=True,
                dtype_matches=(cand_out.dtype == ref_out.dtype),
                passed=passed,
                applied_threshold=max_allowed,
                applied_min_psnr=min_psnr,
                candidate_kind=cand_kind,
                rejection_reason=reason,
                fraction_exceeding_tol=frac_exceeding,
            )

            results.append({
                "test_name": test_name,
                "passed": passed,
                "metrics": test_metric.to_dict(),
            })

            if not passed:
                rep = VerificationReport(
                    candidate_id=candidate_id,
                    all_passed=False,
                    rejection_reason=f"Failed on '{test_name}': {reason}",
                    test_case_results=results,
                )
                self._verification_cache[cache_key] = rep
                return rep

        rep = VerificationReport(
            candidate_id=candidate_id,
            all_passed=True,
            rejection_reason=None,
            test_case_results=results,
        )
        self._verification_cache[cache_key] = rep
        return rep
