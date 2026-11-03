"""Tests for Registry Milestone 3 improvements:
1. Environment version fingerprinting and STALE marking.
2. SHA-256 verification and tamper detection.
3. Single version rollback fallback to baseline.
4. ROLLED_BACK versions prevented from auto-re-promotion.
5. Single-transaction guarantee that at most one version is ACTIVE per key.
"""

import os
import gc
import json
import pytest
import tempfile
import time

from seae.registry import KernelRegistry, compute_code_sha256, get_environment_versions


def _safe_remove(path: str, max_retries: int = 5):
    gc.collect()
    for _ in range(max_retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            time.sleep(0.05)
            gc.collect()


@pytest.fixture
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield db_path
    _safe_remove(db_path)


def test_sha256_computation_and_tamper_detection(temp_db):
    reg = KernelRegistry(db_path=temp_db)
    code = "def kernel_entry(img):\n    return img * 2\n"
    expected_sha = compute_code_sha256(code)

    reg.register_candidate(
        kernel_name="gaussian_blur",
        size_str="128x128",
        hardware_fingerprint="HW_SHA_TEST",
        candidate_id="cand_sha_1",
        backend="numpy",
        source_code=code,
        median_time_ms=10.0,
        speedup_ratio=1.0,
        peak_memory_mb=25.0,
        verification_passed=True,
    )

    # Clean fetch: must succeed and SHA matches
    entry = reg.get_latest_active("gaussian_blur", "128x128", "HW_SHA_TEST")
    assert entry is not None
    assert entry.code_sha256 == expected_sha

    # Tamper with code directly in SQLite database
    with reg._get_connection() as conn:
        conn.execute("UPDATE kernel_cache SET source_code = 'MALICIOUS_CODE' WHERE entry_id = ?", (entry.entry_id,))
        conn.commit()

    # Next lookup must detect tamper, mark CORRUPTED, and refuse to serve
    tampered_entry = reg.get_latest_active("gaussian_blur", "128x128", "HW_SHA_TEST")
    assert tampered_entry is None


def test_single_version_rollback_falls_back_to_baseline(temp_db):
    reg = KernelRegistry(db_path=temp_db)

    # Register only a single version (v1)
    v1 = reg.register_candidate(
        kernel_name="threshold",
        size_str="256x256",
        hardware_fingerprint="HW_SINGLE_V",
        candidate_id="thresh_v1",
        backend="numpy",
        source_code="def kernel_entry(img): return img",
        median_time_ms=5.0,
        speedup_ratio=1.0,
        peak_memory_mb=30.0,
        verification_passed=True,
    )
    assert v1.version == 1

    # Rollback when only v1 exists
    restored, status_msg = reg.rollback("threshold", "256x256", "HW_SINGLE_V")
    assert restored is None
    assert status_msg == "FALLBACK_TO_BASELINE"

    # Active lookup must now return None (forces engine back to trusted reference/baseline)
    active = reg.get_latest_active("threshold", "256x256", "HW_SINGLE_V")
    assert active is None


def test_rolled_back_versions_never_auto_repromoted(temp_db):
    reg = KernelRegistry(db_path=temp_db)
    code = "def kernel_entry(img):\n    return img + 1\n"

    reg.register_candidate(
        kernel_name="sobel",
        size_str="64x64",
        hardware_fingerprint="HW_NO_REPROMOTE",
        candidate_id="sobel_orig",
        backend="numpy",
        source_code=code,
        median_time_ms=8.0,
        speedup_ratio=1.0,
        peak_memory_mb=15.0,
        verification_passed=True,
    )

    # Roll back
    reg.rollback("sobel", "64x64", "HW_NO_REPROMOTE")

    # Attempting to re-promote identical rolled-back code must raise ValueError
    with pytest.raises(ValueError, match="matches a previously ROLLED_BACK implementation"):
        reg.register_candidate(
            kernel_name="sobel",
            size_str="64x64",
            hardware_fingerprint="HW_NO_REPROMOTE",
            candidate_id="sobel_attempt_2",
            backend="numpy",
            source_code=code,
            median_time_ms=7.5,
            speedup_ratio=1.0,
            peak_memory_mb=15.0,
            verification_passed=True,
        )


def test_single_active_constraint_in_transaction(temp_db):
    reg = KernelRegistry(db_path=temp_db)

    # Register 3 successive versions
    for i in range(1, 4):
        reg.register_candidate(
            kernel_name="conv2d",
            size_str="64x64",
            hardware_fingerprint="HW_CONCURRENT",
            candidate_id=f"conv_v{i}",
            backend="numpy",
            source_code=f"def kernel_entry(img):\n    return img * {i}\n",
            median_time_ms=10.0 / i,
            speedup_ratio=float(i),
            peak_memory_mb=20.0,
            verification_passed=True,
        )

    # Inspect SQLite database directly: exactly ONE version must have status = 'ACTIVE'
    with reg._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM kernel_cache
            WHERE kernel_name = 'conv2d' AND size_str = '64x64'
              AND hardware_fingerprint = 'HW_CONCURRENT' AND status = 'ACTIVE'
        """)
        active_count = cur.fetchone()[0]
        assert active_count == 1

        # Check latest version is the active one
        cur.execute("""
            SELECT version FROM kernel_cache
            WHERE kernel_name = 'conv2d' AND size_str = '64x64'
              AND hardware_fingerprint = 'HW_CONCURRENT' AND status = 'ACTIVE'
        """)
        active_version = cur.fetchone()[0]
        assert active_version == 3


def test_stale_detection_on_env_version_change(temp_db):
    reg = KernelRegistry(db_path=temp_db)
    code = "def kernel_entry(img): return img"

    reg.register_candidate(
        kernel_name="threshold",
        size_str="32x32",
        hardware_fingerprint="HW_STALE_TEST",
        candidate_id="cand_env_1",
        backend="opencv",
        source_code=code,
        median_time_ms=1.0,
        speedup_ratio=1.0,
        peak_memory_mb=10.0,
        verification_passed=True,
    )

    # Modify env_versions_json to simulate an upgraded NumPy version
    fake_env = get_environment_versions()
    fake_env["numpy"] = "9.9.9"
    with reg._get_connection() as conn:
        conn.execute("UPDATE kernel_cache SET env_versions_json = ? WHERE candidate_id = 'cand_env_1'", (json.dumps(fake_env),))
        conn.commit()

    # Querying should notice the version drift, mark it STALE, and return None
    entry = reg.get_latest_active("threshold", "32x32", "HW_STALE_TEST")
    assert entry is None
