"""Tests for Milestone 3: Registry, Caching, Rollback, and Diffs."""

import os
import gc
import pytest
import tempfile
import time

from seae.registry import KernelRegistry


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


def test_registry_registration_and_lookup(temp_db):
    reg = KernelRegistry(db_path=temp_db)

    # Initial lookup on empty cache
    res = reg.get_latest_active("gaussian_blur", "1920x1080", "HW_TEST_1")
    assert res is None

    # Register v1
    v1 = reg.register_candidate(
        kernel_name="gaussian_blur",
        size_str="1920x1080",
        hardware_fingerprint="HW_TEST_1",
        candidate_id="cand_v1",
        backend="numpy",
        source_code="def kernel_entry(img): return img",
        median_time_ms=25.0,
        speedup_ratio=1.0,
        peak_memory_mb=120.0,
        verification_passed=True
    )
    assert v1.version == 1
    assert v1.status == "ACTIVE"

    # Query active
    active = reg.get_latest_active("gaussian_blur", "1920x1080", "HW_TEST_1")
    assert active is not None
    assert active.candidate_id == "cand_v1"
    assert active.version == 1


def test_registry_version_promotion_and_diff(temp_db):
    reg = KernelRegistry(db_path=temp_db)

    # Register v1
    reg.register_candidate(
        kernel_name="threshold",
        size_str="512x512",
        hardware_fingerprint="HW_M1",
        candidate_id="cand_v1",
        backend="numpy",
        source_code="def kernel_entry(img):\n    return img > 128\n",
        median_time_ms=10.0,
        speedup_ratio=1.0,
        peak_memory_mb=50.0,
        verification_passed=True
    )

    # Register v2 (faster)
    v2 = reg.register_candidate(
        kernel_name="threshold",
        size_str="512x512",
        hardware_fingerprint="HW_M1",
        candidate_id="cand_v2",
        backend="numba",
        source_code="def kernel_entry(img):\n    # Optimized Numba\n    return img > 128\n",
        median_time_ms=2.5,
        speedup_ratio=4.0,
        peak_memory_mb=52.0,
        verification_passed=True
    )
    assert v2.version == 2
    assert v2.status == "ACTIVE"
    assert "+    # Optimized Numba" in v2.diff_from_prior

    # Check v1 was marked SUPERSEDED
    history = reg.get_history("threshold", "512x512", "HW_M1")
    assert len(history) == 2
    assert history[0].version == 1
    assert history[0].status == "SUPERSEDED"
    assert history[1].version == 2
    assert history[1].status == "ACTIVE"


def test_registry_rollback(temp_db):
    reg = KernelRegistry(db_path=temp_db)

    # Register v1
    reg.register_candidate(
        kernel_name="sobel",
        size_str="256x256",
        hardware_fingerprint="HW_X",
        candidate_id="sobel_v1",
        backend="numpy",
        source_code="def kernel_entry(img): return img",
        median_time_ms=15.0,
        speedup_ratio=1.0,
        peak_memory_mb=40.0,
        verification_passed=True
    )

    # Register v2
    reg.register_candidate(
        kernel_name="sobel",
        size_str="256x256",
        hardware_fingerprint="HW_X",
        candidate_id="sobel_v2",
        backend="numba",
        source_code="def kernel_entry(img): return img * 2",
        median_time_ms=4.0,
        speedup_ratio=3.75,
        peak_memory_mb=41.0,
        verification_passed=True
    )

    # Active should currently be v2
    curr = reg.get_latest_active("sobel", "256x256", "HW_X")
    assert curr.version == 2

    # Perform rollback
    rolled, status_msg = reg.rollback("sobel", "256x256", "HW_X")
    assert rolled is not None
    assert rolled.version == 1
    assert rolled.status == "ACTIVE"
    assert status_msg == "RESTORED_PREVIOUS"

    # Current active after rollback must be v1
    new_curr = reg.get_latest_active("sobel", "256x256", "HW_X")
    assert new_curr.version == 1
    assert new_curr.candidate_id == "sobel_v1"


def test_registry_rejects_unverified_candidate(temp_db):
    reg = KernelRegistry(db_path=temp_db)
    with pytest.raises(ValueError, match="verification failed"):
        reg.register_candidate(
            kernel_name="conv2d",
            size_str="128x128",
            hardware_fingerprint="HW_Y",
            candidate_id="bad_cand",
            backend="bad",
            source_code="invalid",
            median_time_ms=1.0,
            speedup_ratio=10.0,
            peak_memory_mb=10.0,
            verification_passed=False
        )


def test_registry_json_export_and_import(temp_db):
    reg = KernelRegistry(db_path=temp_db)
    reg.register_candidate(
        kernel_name="gaussian_blur",
        size_str="100x100",
        hardware_fingerprint="HW_TEST_JSON",
        candidate_id="cand_j1",
        backend="numpy",
        source_code="code_sample",
        median_time_ms=5.0,
        speedup_ratio=1.0,
        peak_memory_mb=20.0,
        verification_passed=True
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f_json:
        json_path = f_json.name

    try:
        reg.export_to_json(json_path)
        assert os.path.exists(json_path)

        # Import into fresh DB
        fd_new, new_db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd_new)

        try:
            new_reg = KernelRegistry(db_path=new_db_path)
            imported_count = new_reg.import_from_json(json_path)
            assert imported_count == 1
            entry = new_reg.get_latest_active("gaussian_blur", "100x100", "HW_TEST_JSON")
            assert entry is not None
            assert entry.candidate_id == "cand_j1"
        finally:
            _safe_remove(new_db_path)
    finally:
        _safe_remove(json_path)
