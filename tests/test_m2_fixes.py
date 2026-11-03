"""Tests specifically for Milestone 2 fixes and improvements:
- Cross-platform interpreter lookup.
- JIT / warm-up time tracking and thread count logging.
- Exhaustive search space enumeration and comparison.
"""

import os
import sys
import pytest
import numpy as np

from seae.sandbox import get_default_python_executable, Sandbox
from seae.benchmarker import Benchmarker
from seae.task_analyzer import TaskAnalyzer
from seae.candidate_generator import CandidateGenerator, KernelConfig


def test_get_default_python_executable():
    exe = get_default_python_executable()
    assert os.path.exists(exe)
    assert "python" in os.path.basename(exe).lower()


def test_jit_warmup_time_and_threads_tracking():
    spec = TaskAnalyzer.analyze("gaussian_blur", size_str="64x64", dtype_str="uint8")
    gen = CandidateGenerator(seed=42)
    # Generate Numba kernel which triggers JIT compilation on warm-up
    cand = gen.generate_candidate("gaussian_blur", KernelConfig(backend="numba", fastmath=True))

    bench = Benchmarker(default_runs=3, default_warmup=2)
    metrics = bench.benchmark_candidate(cand.candidate_id, cand.source_code, spec)

    # Validate warm-up & compile time is tracked
    assert metrics.warmup_compile_time_ms > 0
    assert metrics.first_run_overhead_ms > 0
    assert metrics.active_threads >= 1
    assert metrics.is_pure_python_baseline is False


def test_pure_python_baseline_boundary_and_flag():
    spec = TaskAnalyzer.analyze("threshold", size_str="512x512", dtype_str="uint8")
    gen = CandidateGenerator(seed=42)
    cand = gen.generate_candidate("threshold", KernelConfig(backend="python_loops"))

    bench = Benchmarker()
    metrics = bench.benchmark_candidate(
        cand.candidate_id,
        cand.source_code,
        spec,
        is_pure_python=True
    )

    assert metrics.is_pure_python_baseline is True
    # Pure-python loops on large 512x512 are automatically bounded to <= 128x128
    # and fast runs so it finishes in a few milliseconds without stalling search
    assert metrics.median_time_ms > 0


def test_exhaustive_enumeration_completeness():
    gen = CandidateGenerator(seed=42)
    configs = gen.enumerate_all_discrete_configs("gaussian_blur")
    assert len(configs) >= 15

    backends = {c.backend for c in configs}
    assert "opencv" in backends
    assert "python_loops" in backends
    assert "numpy" in backends
    assert "numba" in backends

    # Check separable flag for gaussian_blur
    assert any(c.separable is True for c in configs if c.backend == "numpy")
