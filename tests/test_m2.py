"""Tests for Milestone 2: Benchmarker and Candidate Generator."""

import pytest
import numpy as np

from seae.task_analyzer import TaskAnalyzer
from seae.candidate_generator import CandidateGenerator, KernelConfig
from seae.benchmarker import Benchmarker, RAPLEnergySensor
from seae.verifier import Verifier


def test_candidate_generation_synthesizes_valid_code():
    gen = CandidateGenerator(seed=123)
    spec = TaskAnalyzer.analyze("gaussian_blur", size_str="64x64")

    # 1. Test OpenCV variant
    c_cv = gen.generate_candidate("gaussian_blur", KernelConfig(backend="opencv"))
    assert "cv2.GaussianBlur" in c_cv.source_code

    # 2. Test NumPy separable variant
    c_np_sep = gen.generate_candidate("gaussian_blur", KernelConfig(backend="numpy", separable=True))
    assert "_gaussian_1d" in c_np_sep.source_code

    # 3. Test Numba parallel variant
    c_nb = gen.generate_candidate("gaussian_blur", KernelConfig(backend="numba", parallel=True, fastmath=True))
    assert "@njit" in c_nb.source_code
    assert "prange" in c_nb.source_code


def test_random_search_batch_reproducibility():
    gen1 = CandidateGenerator(seed=42)
    batch1 = gen1.generate_random_batch("threshold", n=5)

    gen2 = CandidateGenerator(seed=42)
    batch2 = gen2.generate_random_batch("threshold", n=5)

    assert len(batch1) == 5
    assert len(batch2) == 5
    for c1, c2 in zip(batch1, batch2):
        assert c1.config.backend == c2.config.backend
        assert c1.config.loop_order == c2.config.loop_order


def test_bandit_arm_sampling():
    gen = CandidateGenerator(seed=999)
    history = {
        "opencv": [100.0, 110.0],
        "numba_parallel": [250.0, 260.0],  # higher is better
    }
    # With epsilon=0.0 (pure exploitation), it must pick numba_parallel
    cand = gen.sample_bandit_arm("conv2d", history, epsilon=0.0)
    assert cand.config.extra_flags["arm"] == "numba_parallel"


def test_evolution_mutation_and_elitism():
    gen = CandidateGenerator(seed=777)
    pop = [
        gen.generate_candidate("sobel", KernelConfig(backend="opencv")),
        gen.generate_candidate("sobel", KernelConfig(backend="numpy")),
        gen.generate_candidate("sobel", KernelConfig(backend="numba")),
    ]
    # Assign scores
    pop[0].score = 50.0
    pop[1].score = 10.0
    pop[2].score = 120.0  # best

    next_gen = gen.evolve_population(pop, population_size=4, generation=1)
    assert len(next_gen) == 4
    # The best candidate (numba, score 120) must be preserved in elitism
    assert any(c.config.backend == "numba" for c in next_gen[:2])


def test_benchmarker_execution():
    spec = TaskAnalyzer.analyze("threshold", size_str="64x64", dtype_str="uint8")
    gen = CandidateGenerator(seed=42)
    cand = gen.generate_candidate("threshold", KernelConfig(backend="numpy"))

    bench = Benchmarker(default_runs=5, default_warmup=2)
    metrics = bench.benchmark_candidate(cand.candidate_id, cand.source_code, spec)

    assert metrics.num_runs == 5
    assert metrics.warmup_runs == 2
    assert metrics.median_time_ms > 0
    assert metrics.peak_memory_mb > 0
    assert len(metrics.all_times_ms) == 5
    assert metrics.energy.note != ""  # Energy reporting must state status honestly


def test_rapl_energy_sensor_graceful_on_windows():
    # RAPL should gracefully indicate unsupported or read integer on Linux
    supported = RAPLEnergySensor.is_available()
    assert isinstance(supported, bool)
    energy = RAPLEnergySensor.read_energy_uj()
    if not supported:
        assert energy is None
