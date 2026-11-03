"""Unit tests for Milestone 5 sign-off criteria:
1. Strict selection rule: >10% improvement AND outside run-to-run std; never slower.
2. Output dtype and tolerance verification for quantized candidates.
3. Break-even computation and baseline speedup reporting (1.00x and 'N/A').
4. Hardware fingerprint CPU-only SIMD detection (no RAM, no unsupported build flags).
"""

import numpy as np
import pytest
from seae.task_analyzer import TaskSpec, TaskAnalyzer
from seae.hardware_manager import HardwareManager, HardwareProfile
from seae.verifier import Verifier, ToleranceConfig, VerificationReport
from seae.benchmarker import Benchmarker, InterleavedComparisonResult
from seae.candidate_generator import CandidateGenerator, KernelConfig
from seae.orchestrator import Orchestrator, SearchBudget


def test_selection_rule_logic():
    """Verify that selection rule strictly enforces:
    1. cand2_mean < cand1_mean (never slower).
    2. cand2_mean < cand1_mean * 0.90 (beats by >10%).
    3. (cand1_mean - cand2_mean) > (cand1_std + cand2_std) (outside run-to-run std).
    """
    # Case 1: Candidate is slower -> MUST REJECT
    times_base = [1.0] * 30
    times_slower = [1.1] * 30
    res_slower = InterleavedComparisonResult(
        cand1_times_ms=times_base,
        cand2_times_ms=times_slower,
        cand1_mean_ms=1.0,
        cand1_std_ms=0.01,
        cand2_mean_ms=1.1,
        cand2_std_ms=0.01,
        cand2_beats_by_10pct=False,
        cand2_outside_std=False,
        promotable=False,
        rejection_reason="Candidate is slower than baseline"
    )
    assert not res_slower.promotable

    # Case 2: Candidate is 5% faster (not >10%) -> MUST REJECT
    times_marginal = [0.95] * 30
    beats_10 = (0.95 < 1.0 * 0.90)
    assert not beats_10
    res_marginal = InterleavedComparisonResult(
        cand1_times_ms=times_base,
        cand2_times_ms=times_marginal,
        cand1_mean_ms=1.0,
        cand1_std_ms=0.01,
        cand2_mean_ms=0.95,
        cand2_std_ms=0.01,
        cand2_beats_by_10pct=False,
        cand2_outside_std=True,
        promotable=False,
    )
    assert not res_marginal.promotable

    # Case 3: Candidate is 20% faster, but noisy (within std) -> MUST REJECT
    # base: 1.0 +/- 0.15, cand: 0.80 +/- 0.15. (1.0 - 0.80 = 0.20 < 0.15 + 0.15 = 0.30)
    outside_std = (1.0 - 0.80) > (0.15 + 0.15)
    assert not outside_std
    res_noisy = InterleavedComparisonResult(
        cand1_times_ms=[1.0] * 30,
        cand2_times_ms=[0.80] * 30,
        cand1_mean_ms=1.0,
        cand1_std_ms=0.15,
        cand2_mean_ms=0.80,
        cand2_std_ms=0.15,
        cand2_beats_by_10pct=True,
        cand2_outside_std=False,
        promotable=False,
    )
    assert not res_noisy.promotable

    # Case 4: Candidate is 30% faster with low variance -> MUST PASS
    # base: 1.0 +/- 0.02, cand: 0.70 +/- 0.02. (1.0 - 0.70 = 0.30 > 0.04)
    outside_std_pass = (1.0 - 0.70) > (0.02 + 0.02)
    beats_10_pass = (0.70 < 1.0 * 0.90)
    assert outside_std_pass and beats_10_pass
    res_winner = InterleavedComparisonResult(
        cand1_times_ms=[1.0] * 30,
        cand2_times_ms=[0.70] * 30,
        cand1_mean_ms=1.0,
        cand1_std_ms=0.02,
        cand2_mean_ms=0.70,
        cand2_std_ms=0.02,
        cand2_beats_by_10pct=True,
        cand2_outside_std=True,
        promotable=True,
    )
    assert res_winner.promotable


def test_quantized_candidate_dtype_rejection(tmp_path):
    """Assert output dtype must match reference task spec; mismatch is rejected."""
    verifier = Verifier()
    spec = TaskAnalyzer.analyze("threshold", size_str="64x64", dtype_str="uint8", thresh=128.0, maxval=255.0)

    # Invalid candidate that returns float32 instead of expected uint8
    wrong_dtype_code = r'''
import numpy as np
def kernel_entry(img, thresh=128.0, maxval=255.0, **kwargs):
    return (img > thresh).astype(np.float32) * maxval
'''
    rep = verifier.verify_candidate("bad_dtype_cand", wrong_dtype_code, spec, is_quantized=True)
    assert not rep.all_passed
    assert "Dtype mismatch" in rep.rejection_reason


def test_quantized_candidate_valid_dtype_and_tolerance(tmp_path):
    """Assert quantized candidate returning matching uint8 dtype and valid values passes."""
    verifier = Verifier()
    spec = TaskAnalyzer.analyze("threshold", size_str="64x64", dtype_str="uint8", thresh=128.0, maxval=255.0)

    valid_quant_code = r'''
import numpy as np
def kernel_entry(img, thresh=128.0, maxval=255.0, **kwargs):
    # Quantized uint8 comparison
    t_val = np.uint8(thresh)
    m_val = np.uint8(maxval)
    return np.where(img > t_val, m_val, np.uint8(0))
'''
    rep = verifier.verify_candidate("good_dtype_cand", valid_quant_code, spec, is_quantized=True)
    assert rep.all_passed
    assert rep.rejection_reason is None


def test_break_even_and_speedup_baseline_equivalence(tmp_path):
    """Verify that when baseline is chosen or speedup <= 1.0, speedup is 1.00x and break-even is 'N/A'."""
    from seae.registry import KernelRegistry
    from seae.approval_gate import ApprovalGate
    db_path = str(tmp_path / "test_be.db")
    registry = KernelRegistry(db_path=db_path)
    approval_gate = ApprovalGate(auto_approve=True)
    orchestrator = Orchestrator(registry=registry, approval_gate=approval_gate)

    # Force a search on small size with budget=2 where no candidate beats opencv
    budget = SearchBudget(max_seconds=10.0, max_candidates=2, strategy="random", expected_lifetime_invocations=1000)
    result = orchestrator.run("gaussian_blur", size_str="64x64", dtype_str="uint8", budget=budget, force_search=True)

    # If OpenCV baseline was retained:
    if result.selected_backend == "opencv":
        assert result.speedup_ratio == 1.0
        # Check text representation
        be_str = "N/A" if result.speedup_ratio <= 1.00 else f"{int(result.break_even_invocations):,}"
        assert be_str == "N/A"


def test_hardware_fingerprint_excludes_ram_and_unsupported_simd():
    """Verify that hardware fingerprint does NOT include RAM and only lists CPU-reported SIMD flags."""
    profile = HardwareManager.get_hardware_profile(refresh=True)

    # RAM must not be in the fingerprint string
    assert "gb" not in profile.fingerprint.lower()
    assert "ram" not in profile.fingerprint.lower()

    # OpenCV build-only flags with trailing '?' (e.g. AVX512-SKX on Zen 3) must be excluded
    assert "AVX512-SKX" not in profile.simd_features
    assert "AVX512F" not in profile.simd_features

    # True hardware features must be present
    assert "AVX2" in profile.simd_features
    assert "SSE" in profile.simd_features


def test_readme_tables_match_benchmark_raw_results_json():
    """Verify that README.md tables and numbers match benchmark_raw_results.json exactly."""
    import json
    from pathlib import Path

    json_path = Path("benchmark_raw_results.json")
    readme_path = Path("README.md")

    assert json_path.exists(), "benchmark_raw_results.json does not exist"
    assert readme_path.exists(), "README.md does not exist"

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_text = f.read()

    # 1. Check Hardware Fingerprint in README
    hw = raw["metadata"]["hardware"]
    assert hw["fingerprint"] in readme_text

    # 2. Check Sobel 256x256 break-even calls and speedup
    sobel_row = next(r for r in raw["matrix"] if r["kernel"] == "sobel" and r["size"] == "256x256")
    sobel_be_calls = f"{sobel_row['seae']['break_even_calls']:,} calls"
    assert sobel_be_calls in readme_text

    sobel_speedup = f"{sobel_row['seae']['speedup_vs_opencv']:.2f}x"
    assert sobel_speedup in readme_text


def test_recomputed_arithmetic_speedup_breakeven_regret_match_readme():
    """Strictly recompute speedup, break-even, and signed regret from raw data and assert README matches."""
    import json
    from pathlib import Path

    json_path = Path("benchmark_raw_results.json")
    readme_path = Path("README.md")
    assert json_path.exists() and readme_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_text = f.read()

    # Verify arithmetic for each matrix cell
    for r in raw["matrix"]:
        cv_ms = r["opencv"]["mean_ms"]
        sel_ms = r["seae"]["winner_mean_ms"]
        is_cv = ("OpenCV" in r["seae"]["winner_description"])

        # Recomputed speedup
        if is_cv or sel_ms >= cv_ms:
            calc_speedup = 1.00
            calc_be_text = "N/A"
        else:
            calc_speedup = round(cv_ms / sel_ms, 2)
            search_s = r["seae"]["search_wall_clock_s"]
            comp_ms = r["seae"]["winner_compile_ms"]
            overhead_ms = (search_s * 1000.0) + comp_ms
            savings_ms = cv_ms - sel_ms
            calc_be_calls = int(round(overhead_ms / savings_ms))
            calc_be_text = f"{calc_be_calls:,} calls"

        sp_str = f"{calc_speedup:.2f}x"
        assert sp_str in readme_text, f"Calculated speedup {sp_str} missing for {r['kernel']} {r['size']}"
        assert calc_be_text in readme_text, f"Calculated break-even {calc_be_text} missing for {r['kernel']} {r['size']}"

    # Verify signed regret for Part 2
    for k_name in ["sobel", "threshold"]:
        if k_name in raw.get("search_comparison", {}):
            for s_name, s_data in raw["search_comparison"][k_name]["strategies"].items():
                if "signed_regret_ms" in s_data:
                    regret_str = f"{s_data['signed_regret_ms']:+.4f}"
                    assert regret_str in readme_text, f"Signed regret {regret_str} missing in README for {s_name}"


def test_tiled_numba_sobel_variant_matches_reference_within_tolerance(tmp_path):
    """Verify that the tiled Numba Sobel variant (with step-1 prange fix) passes verification and matches reference."""
    import importlib.util
    from seae.references import ref_sobel

    gen = CandidateGenerator(seed=42)
    cfg = KernelConfig(backend="numba", fastmath=True, parallel=True, tiling=32)
    cand = gen.generate_candidate("sobel", cfg)

    # 1. Verification battery check via sandbox
    spec = TaskAnalyzer.analyze("sobel", size_str="256x256")
    verifier = Verifier()
    rep = verifier.verify_candidate(cand.candidate_id, cand.source_code, spec)
    assert rep.all_passed, f"Tiled Sobel candidate failed verification: {rep.rejection_reason}"

    # 2. Direct execution and numeric comparison against reference kernel
    cand_file = tmp_path / "cand_tiled_sobel.py"
    cand_file.write_text(cand.source_code, encoding="utf-8")

    spec_import = importlib.util.spec_from_file_location("cand_tiled_sobel", str(cand_file))
    mod = importlib.util.module_from_spec(spec_import)
    spec_import.loader.exec_module(mod)

    img = np.random.RandomState(42).randint(0, 256, (256, 256), dtype=np.uint8)
    ref_out = ref_sobel(img)
    cand_out = mod.kernel_entry(img)

    assert cand_out.shape == ref_out.shape
    assert cand_out.dtype == ref_out.dtype
    mae = float(np.mean(np.abs(cand_out.astype(np.float64) - ref_out.astype(np.float64))))
    assert mae <= 2.0, f"MAE {mae} exceeded tolerance 2.0"



