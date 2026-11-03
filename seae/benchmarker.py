"""Benchmarker module for SEAE.

Accurately measures:
- Warm-up execution runs to prime JIT caches (Numba/BLAS).
- Multi-run median runtime (seconds and ms).
- Interquartile range (IQR) or standard deviation of latency.
- Peak resident memory (RSS MB).
- RAPL energy measurement on supported Linux Intel/AMD platforms,
  with transparent and honest skipping and messaging on Windows or non-RAPL machines.
"""

import os
import sys
import time
import statistics
import tempfile
import json
import subprocess
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from seae.sandbox import Sandbox
from seae.task_analyzer import TaskSpec


@dataclass
class EnergyMeasurement:
    """Hardware energy monitoring reading."""
    supported: bool
    energy_joules: Optional[float] = None
    average_power_watts: Optional[float] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkMetrics:
    """Full performance profile of a kernel candidate."""
    candidate_id: str
    num_runs: int
    warmup_runs: int
    median_time_ms: float
    mean_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    iqr_time_ms: float
    peak_memory_mb: float
    warmup_compile_time_ms: float
    first_run_overhead_ms: float
    active_threads: int
    is_pure_python_baseline: bool
    energy: EnergyMeasurement
    all_times_ms: List[float]

    @property
    def speedup_over(self, baseline_median_ms: float) -> float:
        if self.median_time_ms <= 0:
            return 1.0
        return baseline_median_ms / self.median_time_ms

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["energy"] = self.energy.to_dict()
        return d


INTERLEAVED_RUNNER_TEMPLATE = r'''
import sys
import os
import time
import json
import numpy as np

def main():
    if len(sys.argv) < 6:
        sys.exit(1)

    c1_file = sys.argv[1]
    c2_file = sys.argv[2]
    input_file = sys.argv[3]
    status_file = sys.argv[4]
    num_runs = int(sys.argv[5])
    warmup_runs = int(sys.argv[6]) if len(sys.argv) > 6 else 3

    npz = np.load(input_file)
    img = npz["img"]
    kwargs = {}
    for key in npz.files:
        if key != "img":
            val = npz[key]
            kwargs[key] = val.item() if val.ndim == 0 else val

    target_threads = int(os.environ.get("SEAE_NUM_THREADS", "8"))
    try:
        import cv2
        cv2.setNumThreads(target_threads)
    except Exception:
        pass
    try:
        import numba
        numba.set_num_threads(target_threads)
    except Exception:
        pass

    import importlib.util
    spec1 = importlib.util.spec_from_file_location("c1_mod", c1_file)
    mod1 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(mod1)
    entry1 = mod1.kernel_entry

    spec2 = importlib.util.spec_from_file_location("c2_mod", c2_file)
    mod2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod2)
    entry2 = mod2.kernel_entry

    # Warmup both
    for _ in range(warmup_runs):
        _ = entry1(img, **kwargs)
        _ = entry2(img, **kwargs)

    # Interleaved measurement with randomized ABBA block order
    times1 = []
    times2 = []
    times1_when_first = []
    times1_when_second = []
    times2_when_first = []
    times2_when_second = []

    np.random.seed(42)
    # Ensure even number of runs for balanced ABBA blocks
    effective_runs = num_runs if num_runs % 2 == 0 else num_runs + 1

    for pair_idx in range(effective_runs // 2):
        # Balanced ABBA: alternate [A, B] then [B, A]
        order_pair = [("A", "B"), ("B", "A")] if pair_idx % 2 == 0 else [("B", "A"), ("A", "B")]
        for first, second in order_pair:
            if first == "A":
                t0 = time.perf_counter()
                _ = entry1(img, **kwargs)
                t1 = time.perf_counter()
                dur1 = (t1 - t0) * 1000.0
                times1.append(dur1)
                times1_when_first.append(dur1)

                t2 = time.perf_counter()
                _ = entry2(img, **kwargs)
                t3 = time.perf_counter()
                dur2 = (t3 - t2) * 1000.0
                times2.append(dur2)
                times2_when_second.append(dur2)
            else:
                t2 = time.perf_counter()
                _ = entry2(img, **kwargs)
                t3 = time.perf_counter()
                dur2 = (t3 - t2) * 1000.0
                times2.append(dur2)
                times2_when_first.append(dur2)

                t0 = time.perf_counter()
                _ = entry1(img, **kwargs)
                t1 = time.perf_counter()
                dur1 = (t1 - t0) * 1000.0
                times1.append(dur1)
                times1_when_second.append(dur1)

    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({
            "status": "SUCCESS",
            "times1": times1[:num_runs],
            "times2": times2[:num_runs],
            "times1_when_first": times1_when_first,
            "times1_when_second": times1_when_second,
            "times2_when_first": times2_when_first,
            "times2_when_second": times2_when_second,
        }, f)

if __name__ == "__main__":
    main()
'''


@dataclass
class InterleavedComparisonResult:
    """Rigorous final verification comparing baseline vs candidate over interleaved runs."""
    cand1_times_ms: List[float]
    cand2_times_ms: List[float]
    cand1_mean_ms: float
    cand1_std_ms: float
    cand2_mean_ms: float
    cand2_std_ms: float
    cand2_beats_by_10pct: bool
    cand2_outside_std: bool
    promotable: bool
    rejection_reason: Optional[str] = None
    mean_diff_ms: float = 0.0
    std_diff_ms: float = 0.0
    ci_95_ms: float = 0.0
    order_effect_check: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


BENCHMARK_RUNNER_TEMPLATE = r'''
import sys
import os
import time
import json
import numpy as np

def main():
    if len(sys.argv) < 5:
        sys.exit(1)

    candidate_file = sys.argv[1]
    input_file = sys.argv[2]
    status_file = sys.argv[3]
    num_runs = int(sys.argv[4])
    warmup_runs = int(sys.argv[5])

    npz = np.load(input_file)
    img = npz["img"]
    kwargs = {}
    for key in npz.files:
        if key != "img":
            val = npz[key]
            kwargs[key] = val.item() if val.ndim == 0 else val

    target_threads = int(os.environ.get("SEAE_NUM_THREADS", "8"))
    try:
        import cv2
        cv2.setNumThreads(target_threads)
    except Exception:
        pass
    try:
        import numba
        numba.set_num_threads(target_threads)
    except Exception:
        pass

    import importlib.util
    cand_spec = importlib.util.spec_from_file_location("cand_mod", candidate_file)
    cand_mod = importlib.util.module_from_spec(cand_spec)
    cand_spec.loader.exec_module(cand_mod)
    kernel_entry = cand_mod.kernel_entry

    # 1. Warm-up and JIT compile measurement
    warmup_times_ms = []
    t_warmup_start = time.perf_counter()
    for _ in range(warmup_runs):
        tw0 = time.perf_counter()
        _ = kernel_entry(img, **kwargs)
        tw1 = time.perf_counter()
        warmup_times_ms.append((tw1 - tw0) * 1000.0)
    t_warmup_end = time.perf_counter()
    total_warmup_compile_ms = (t_warmup_end - t_warmup_start) * 1000.0
    first_run_compile_overhead_ms = warmup_times_ms[0] if warmup_times_ms else 0.0

    # 2. Timed benchmarking runs
    run_times_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = kernel_entry(img, **kwargs)
        t1 = time.perf_counter()
        run_times_ms.append((t1 - t0) * 1000.0)

    # Thread count detection if possible
    active_threads = 1
    try:
        import numba
        active_threads = numba.get_num_threads()
    except Exception:
        active_threads = os.cpu_count() or 1

    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({
            "status": "SUCCESS",
            "run_times_ms": run_times_ms,
            "warmup_compile_time_ms": total_warmup_compile_ms,
            "first_run_overhead_ms": first_run_compile_overhead_ms,
            "active_threads": active_threads,
        }, f)

if __name__ == "__main__":
    main()
'''


class RAPLEnergySensor:
    """Linux Intel RAPL energy reader, gracefully skips on Windows or unsupported systems."""

    RAPL_ROOT = "/sys/class/powercap/intel-rapl"

    @classmethod
    def is_available(cls) -> bool:
        return sys.platform.startswith("linux") and os.path.exists(cls.RAPL_ROOT)

    @classmethod
    def read_energy_uj(cls) -> Optional[int]:
        """Read energy in microjoules if sysfs RAPL path is available."""
        if not cls.is_available():
            return None
        energy_file = os.path.join(cls.RAPL_ROOT, "intel-rapl:0", "energy_uj")
        try:
            with open(energy_file, "r") as f:
                return int(f.read().strip())
        except Exception:
            return None


class Benchmarker:
    """Executes candidates in sandbox to measure runtimes, peak RSS, and energy."""

    def __init__(
        self,
        sandbox: Optional[Sandbox] = None,
        default_runs: int = 15,
        default_warmup: int = 3,
    ) -> None:
        self.sandbox = sandbox or Sandbox(timeout_seconds=30.0, memory_limit_mb=2048.0)
        self.default_runs = default_runs
        self.default_warmup = default_warmup

    def benchmark_candidate(
        self,
        candidate_id: str,
        code_str: str,
        task_spec: TaskSpec,
        runs: Optional[int] = None,
        warmup: Optional[int] = None,
        is_pure_python: bool = False,
        seed: int = 42,
        bound_pure_python: bool = True,
    ) -> BenchmarkMetrics:
        """Measure performance across repeated runs in a fresh isolated process."""
        import psutil

        # Rule 3: Benchmark pure-Python candidates only at small sizes with a hard timeout; mark as baseline
        is_pure_py = is_pure_python or ("python_loops" in candidate_id) or ("def _py_" in code_str)
        if is_pure_py and bound_pure_python:
            # Restrict pure-python loops to at most 64x64 to prevent long hangs while preserving baseline comparison
            effective_shape = (min(task_spec.height, 64), min(task_spec.width, 64))
            if task_spec.channels > 1:
                effective_shape = (*effective_shape, task_spec.channels)
            hard_timeout = min(self.sandbox.timeout_seconds, 4.0)
            n_runs = min(runs or self.default_runs, 2)
            n_warmup = 1
        elif is_pure_py and not bound_pure_python:
            # Full resolution pure-Python evaluation with dedicated single-pass timing
            effective_shape = task_spec.shape
            hard_timeout = max(self.sandbox.timeout_seconds, 35.0)
            n_runs = min(runs or 1, 1)
            n_warmup = 0
        else:
            effective_shape = task_spec.shape
            hard_timeout = self.sandbox.timeout_seconds
            n_runs = runs or self.default_runs
            n_warmup = warmup or self.default_warmup

        # Rule 5: Run each candidate benchmark on identical seeded inputs
        rng = np.random.RandomState(seed)
        dtype = task_spec.dtype
        is_int = np.issubdtype(dtype, np.integer)
        if is_int:
            img = rng.randint(0, np.iinfo(dtype).max + 1, size=effective_shape, dtype=dtype)
        else:
            img = rng.uniform(0.0, 1.0, size=effective_shape).astype(dtype)

        kwargs: Dict[str, Any] = {}
        if task_spec.kernel_name == "gaussian_blur":
            kwargs["ksize"] = task_spec.ksize
            kwargs["sigma"] = task_spec.sigma
        elif task_spec.kernel_name == "threshold":
            kwargs["thresh"] = task_spec.thresh
            kwargs["maxval"] = task_spec.maxval

        with tempfile.TemporaryDirectory(prefix="seae_bench_", ignore_cleanup_errors=True) as tmpdir:
            cand_path = os.path.join(tmpdir, "candidate.py")
            bench_runner_path = os.path.join(tmpdir, "bench_runner.py")
            input_path = os.path.join(tmpdir, "input.npz")
            status_path = os.path.join(tmpdir, "status.json")

            with open(cand_path, "w", encoding="utf-8") as f:
                f.write(code_str)

            with open(bench_runner_path, "w", encoding="utf-8") as f:
                f.write(BENCHMARK_RUNNER_TEMPLATE)

            save_dict = {"img": img}
            for k, v in kwargs.items():
                save_dict[k] = np.array(v)
            np.savez_compressed(input_path, **save_dict)

            cmd = [
                self.sandbox.python_executable,
                bench_runner_path,
                cand_path,
                input_path,
                status_path,
                str(n_runs),
                str(n_warmup)
            ]

            # Check RAPL energy before
            rapl_start_uj = RAPLEnergySensor.read_energy_uj()
            t_wall_start = time.perf_counter()

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            try:
                ps_proc = psutil.Process(proc.pid)
            except Exception:
                ps_proc = None

            peak_rss_mb = 0.0
            poll_interval = 0.02
            deadline = time.perf_counter() + hard_timeout

            while True:
                ret = proc.poll()
                if ret is not None:
                    break

                if time.perf_counter() > deadline:
                    self.sandbox._kill_process_tree(proc.pid)
                    if is_pure_py:
                        timeout_ms = hard_timeout * 1000.0
                        return BenchmarkMetrics(
                            candidate_id=candidate_id,
                            num_runs=1,
                            warmup_runs=0,
                            median_time_ms=timeout_ms,
                            mean_time_ms=timeout_ms,
                            min_time_ms=timeout_ms,
                            max_time_ms=timeout_ms,
                            std_time_ms=0.0,
                            iqr_time_ms=0.0,
                            peak_memory_mb=peak_rss_mb,
                            warmup_compile_time_ms=0.0,
                            first_run_overhead_ms=0.0,
                            active_threads=1,
                            is_pure_python_baseline=True,
                            energy=EnergyMeasurement(supported=False, note="Timeout threshold exceeded"),
                            all_times_ms=[timeout_ms],
                        )
                    raise TimeoutError(f"Benchmarking timed out for candidate {candidate_id}")

                if ps_proc is not None:
                    try:
                        rss_mb = ps_proc.memory_info().rss / (1024 * 1024)
                        if rss_mb > peak_rss_mb:
                            peak_rss_mb = rss_mb
                    except Exception:
                        pass
                time.sleep(poll_interval)

            t_wall_end = time.perf_counter()
            rapl_end_uj = RAPLEnergySensor.read_energy_uj()
            stdout, stderr = proc.communicate(timeout=0.5)

            if proc.returncode != 0:
                raise RuntimeError(f"Benchmarking runner failed: {stderr or stdout}")

            with open(status_path, "r", encoding="utf-8") as f:
                status_data = json.load(f)

            run_times = status_data["run_times_ms"]
            warmup_compile_ms = status_data.get("warmup_compile_time_ms", 0.0)
            first_run_overhead_ms = status_data.get("first_run_overhead_ms", 0.0)
            active_threads = status_data.get("active_threads", 1)

            # Rule 7: Do not report energy numbers when RAPL is unavailable
            if rapl_start_uj is not None and rapl_end_uj is not None:
                energy_joules = (rapl_end_uj - rapl_start_uj) / 1_000_000.0
                wall_secs = t_wall_end - t_wall_start
                power_w = energy_joules / wall_secs if wall_secs > 0 else 0.0
                energy_obj = EnergyMeasurement(
                    supported=True,
                    energy_joules=energy_joules,
                    average_power_watts=power_w,
                    note="Measured via Intel RAPL powercap interface."
                )
            else:
                energy_obj = EnergyMeasurement(
                    supported=False,
                    energy_joules=None,
                    average_power_watts=None,
                    note="RAPL unavailable on this platform. Energy reporting omitted."
                )

            med = float(statistics.median(run_times))
            mean = float(statistics.mean(run_times))
            min_t = float(min(run_times))
            max_t = float(max(run_times))
            std_t = float(statistics.stdev(run_times)) if len(run_times) > 1 else 0.0

            # Interquartile range (IQR)
            q75, q25 = np.percentile(run_times, [75, 25])
            iqr_t = float(q75 - q25)

            return BenchmarkMetrics(
                candidate_id=candidate_id,
                num_runs=n_runs,
                warmup_runs=n_warmup,
                median_time_ms=med,
                mean_time_ms=mean,
                min_time_ms=min_t,
                max_time_ms=max_t,
                std_time_ms=std_t,
                iqr_time_ms=iqr_t,
                peak_memory_mb=peak_rss_mb,
                warmup_compile_time_ms=warmup_compile_ms,
                first_run_overhead_ms=first_run_overhead_ms,
                active_threads=active_threads,
                is_pure_python_baseline=is_pure_python,
                energy=energy_obj,
                all_times_ms=run_times,
            )

    def benchmark_interleaved_comparison(
        self,
        candidate1_code: str,
        candidate2_code: str,
        task_spec: TaskSpec,
        runs: int = 30,
        warmup_runs: int = 3,
        input_array: Optional[np.ndarray] = None,
    ) -> InterleavedComparisonResult:
        """Execute >=30 interleaved runs alternating candidate1 (baseline) and candidate2.

        Evaluates the strict selection rule:
        1. candidate2 must be faster than candidate1 (never select a slower candidate).
        2. candidate2 must beat candidate1 by >10% (mean2 < mean1 * 0.90).
        3. candidate2 must beat candidate1 outside run-to-run std: (mean1 - mean2) > (std1 + std2).
        """
        if runs < 30:
            runs = 30

        if input_array is None:
            dtype = task_spec.dtype
            h, w = task_spec.height, task_spec.width
            if np.issubdtype(dtype, np.integer):
                min_v = 0
                max_v = np.iinfo(dtype).max
                input_array = np.random.randint(min_v, max_v + 1, size=(h, w), dtype=dtype)
            else:
                input_array = np.random.uniform(0.0, 1.0, size=(h, w)).astype(dtype)

        kwargs = {}
        if task_spec.kernel_name == "gaussian_blur":
            kwargs["ksize"] = task_spec.ksize
            kwargs["sigma"] = task_spec.sigma
        elif task_spec.kernel_name == "threshold":
            kwargs["thresh"] = task_spec.thresh
            kwargs["maxval"] = task_spec.maxval

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            c1_path = os.path.join(tmp_dir, "cand1.py")
            c2_path = os.path.join(tmp_dir, "cand2.py")
            runner_path = os.path.join(tmp_dir, "interleaved_runner.py")
            input_path = os.path.join(tmp_dir, "input.npz")
            status_path = os.path.join(tmp_dir, "interleaved_status.json")

            with open(c1_path, "w", encoding="utf-8") as f:
                f.write(candidate1_code)
            with open(c2_path, "w", encoding="utf-8") as f:
                f.write(candidate2_code)
            with open(runner_path, "w", encoding="utf-8") as f:
                f.write(INTERLEAVED_RUNNER_TEMPLATE)

            np.savez(input_path, img=input_array, **kwargs)

            python_bin = self.sandbox.python_executable
            cmd = [
                python_bin,
                runner_path,
                c1_path,
                c2_path,
                input_path,
                status_path,
                str(runs),
                str(warmup_runs),
            ]

            env = os.environ.copy()
            env["SEAE_NUM_THREADS"] = "8"
            env["OMP_NUM_THREADS"] = "8"
            env["OPENBLAS_NUM_THREADS"] = "8"
            env["MKL_NUM_THREADS"] = "8"
            env["NUMBA_NUM_THREADS"] = "8"

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=60.0,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            if proc.returncode != 0:
                raise RuntimeError(f"Interleaved benchmarking runner failed: {proc.stderr or proc.stdout}")

            with open(status_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            times1 = data["times1"]
            times2 = data["times2"]

            mean1 = float(statistics.mean(times1))
            std1 = float(statistics.stdev(times1)) if len(times1) > 1 else 0.0
            mean2 = float(statistics.mean(times2))
            std2 = float(statistics.stdev(times2)) if len(times2) > 1 else 0.0

            beats_by_10 = (mean2 < mean1 * 0.90)
            outside_std = ((mean1 - mean2) > (std1 + std2))
            never_slower = (mean2 < mean1)

            # Paired difference analysis for signed regret with 95% Confidence Interval
            import math
            diffs = [t2 - t1 for t1, t2 in zip(times1, times2)]
            mean_diff = float(statistics.mean(diffs))
            std_diff = float(statistics.stdev(diffs)) if len(diffs) > 1 else 0.0
            ci_95 = 2.045 * (std_diff / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0

            reasons = []
            if not never_slower:
                reasons.append(f"Candidate is slower than baseline ({mean2:.4f} ms >= {mean1:.4f} ms)")
            elif not beats_by_10:
                reasons.append(f"Improvement under 10% threshold (cand: {mean2:.4f} ms, 90% base: {mean1*0.90:.4f} ms)")
            elif not outside_std:
                reasons.append(f"Improvement ({mean1 - mean2:.4f} ms) within combined std ({std1 + std2:.4f} ms)")

            promotable = never_slower and beats_by_10 and outside_std

            t1_first = data.get("times1_when_first", [])
            t1_sec = data.get("times1_when_second", [])
            t2_first = data.get("times2_when_first", [])
            t2_sec = data.get("times2_when_second", [])

            order_check = {
                "cand1_mean_when_first": float(statistics.mean(t1_first)) if t1_first else mean1,
                "cand1_mean_when_second": float(statistics.mean(t1_sec)) if t1_sec else mean1,
                "cand2_mean_when_first": float(statistics.mean(t2_first)) if t2_first else mean2,
                "cand2_mean_when_second": float(statistics.mean(t2_sec)) if t2_sec else mean2,
                "cand1_order_effect_ms": (float(statistics.mean(t1_first)) - float(statistics.mean(t1_sec))) if (t1_first and t1_sec) else 0.0,
                "cand2_order_effect_ms": (float(statistics.mean(t2_first)) - float(statistics.mean(t2_sec))) if (t2_first and t2_sec) else 0.0,
            }

            return InterleavedComparisonResult(
                cand1_times_ms=times1,
                cand2_times_ms=times2,
                cand1_mean_ms=mean1,
                cand1_std_ms=std1,
                cand2_mean_ms=mean2,
                cand2_std_ms=std2,
                cand2_beats_by_10pct=beats_by_10,
                cand2_outside_std=outside_std,
                promotable=promotable,
                rejection_reason="; ".join(reasons) if reasons else None,
                mean_diff_ms=mean_diff,
                std_diff_ms=std_diff,
                ci_95_ms=ci_95,
                order_effect_check=order_check,
            )
