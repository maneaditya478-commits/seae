"""Tests for Milestone 4: Orchestrator, Approval Gate, and Search Budgeting."""

import os
import gc
import tempfile
import time
import pytest

from seae.orchestrator import Orchestrator, SearchBudget
from seae.approval_gate import ApprovalGate, ApprovalDecision
from seae.registry import KernelRegistry
from seae.task_analyzer import TaskAnalyzer


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


def test_orchestrator_cache_hit_bypasses_search(temp_db):
    registry = KernelRegistry(db_path=temp_db)
    gate = ApprovalGate(auto_approve=True)
    orchestrator = Orchestrator(registry=registry, approval_gate=gate)

    # 1. First run: cache miss -> search and promote
    res1 = orchestrator.run(
        kernel_name="threshold",
        size_str="64x64",
        budget=SearchBudget(max_candidates=2, max_seconds=10.0),
    )
    assert res1.cache_hit is False
    assert res1.candidates_evaluated >= 1
    assert res1.promoted_to_cache is True

    # 2. Second run: immediate cache hit -> 0 candidates evaluated, 0s search duration
    res2 = orchestrator.run(
        kernel_name="threshold",
        size_str="64x64",
        budget=SearchBudget(max_candidates=2, max_seconds=10.0),
    )
    assert res2.cache_hit is True
    assert res2.candidates_evaluated == 0
    assert res2.search_duration_seconds == 0.0
    assert res2.selected_candidate_id == res1.selected_candidate_id


@pytest.mark.slow
def test_orchestrator_candidate_count_budget_enforcement(temp_db):
    registry = KernelRegistry(db_path=temp_db)
    gate = ApprovalGate(auto_approve=True)
    orchestrator = Orchestrator(registry=registry, approval_gate=gate)

    max_cands = 3
    # Use high expected_lifetime_invocations so ROI cutoff doesn't trigger before candidate count budget
    res = orchestrator.run(
        kernel_name="gaussian_blur",
        size_str="64x64",
        budget=SearchBudget(
            max_candidates=max_cands,
            max_seconds=30.0,
            expected_lifetime_invocations=1_000_000,
            strategy="random"
        ),
        force_search=True,
    )
    assert res.candidates_evaluated <= max_cands
    assert res.budget_exhausted is True
    assert "budget reached" in (res.early_stop_reason or "").lower()


def test_early_stopping_roi_rule(temp_db):
    """Test that search early-stops when search cost exceeds projected lifetime savings."""
    registry = KernelRegistry(db_path=temp_db)
    gate = ApprovalGate(auto_approve=True)
    orchestrator = Orchestrator(registry=registry, approval_gate=gate)

    # Set expected_invocations very low (e.g. 5 invocations)
    # Even if savings are 0.1 ms/call, 5 * 0.1 = 0.5 ms savings,
    # which will immediately be dwarfed by search cost (~20-50ms).
    budget = SearchBudget(
        max_candidates=10,
        max_seconds=20.0,
        expected_lifetime_invocations=2, # Tiny ROI amortization
        strategy="random"
    )
    res = orchestrator.run(
        kernel_name="threshold",
        size_str="64x64",
        budget=budget,
        force_search=True,
    )
    # Must stop early due to ROI cutoff
    assert res.early_stop_reason is not None
    assert "ROI cutoff" in res.early_stop_reason
    # Rule: Must have evaluated at least 2 candidates (baseline + 1 novel) before stopping
    assert res.candidates_evaluated >= 2
    assert res.candidates_evaluated < 10


def test_approval_gate_blocks_promotion_when_unapproved():
    gate = ApprovalGate(auto_approve=False)
    # Non-interactive call must safely reject promotion
    decision = gate.request_approval(
        candidate_id="cand_test",
        kernel_name="threshold",
        size_str="64x64",
        speedup_ratio=1.5,
        baseline_ms=5.0,
        new_ms=3.33,
        diff_str="+ new code",
    )
    assert decision.approved is False
    assert "blocked" in decision.reason.lower() or "rejected" in decision.reason.lower()
