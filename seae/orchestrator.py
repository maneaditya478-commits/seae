"""Central orchestrator for SEAE (Self-Evolving Algorithm Engine).

Ties all components together:
1. Checks the cache first; searches only on a miss (or when force_search=True).
2. Manages search budgets (max_seconds and max_candidates).
3. Evaluates candidates: Sandbox Verification -> Benchmarking.
4. Early search termination when search cost exceeds projected lifetime savings:
   if (search_time_spent_ms > expected_lifetime_invocations * cumulative_savings_ms),
   we abort further search to prevent amortized loss.
5. Employs Approval Gate before committing new kernels into the Registry.
"""

import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple

from seae.task_analyzer import TaskAnalyzer, TaskSpec
from seae.hardware_manager import HardwareManager, HardwareProfile
from seae.verifier import Verifier, VerificationReport, ToleranceConfig
from seae.benchmarker import Benchmarker, BenchmarkMetrics
from seae.candidate_generator import CandidateGenerator, Candidate, KernelConfig
from seae.registry import KernelRegistry, RegistryEntry, make_environment_fingerprint
from seae.approval_gate import ApprovalGate, ApprovalDecision

logger = logging.getLogger("SEAE.Orchestrator")


@dataclass
class SearchBudget:
    """Constraints on evolutionary/bandit search space exploration."""
    max_candidates: int = 15
    max_seconds: float = 30.0
    expected_lifetime_invocations: int = 10_000   # Amortization threshold for ROI
    strategy: str = "evolutionary"                # "evolutionary", "bandit", "random"


@dataclass
class ExecutionLogEntry:
    """Record of a single evaluated candidate."""
    candidate_id: str
    backend: str
    verified: bool
    rejection_reason: Optional[str]
    median_time_ms: Optional[float]
    speedup_ratio: Optional[float]
    warmup_compile_ms: float
    evaluation_cost_ms: float


@dataclass
class OrchestratorResult:
    """Outcome of orchestrator kernel request."""
    kernel_name: str
    size_str: str
    cache_hit: bool
    selected_candidate_id: str
    selected_backend: str
    source_code: str
    baseline_median_ms: float
    selected_median_ms: float
    speedup_ratio: float
    search_duration_seconds: float
    candidates_evaluated: int
    promoted_to_cache: bool
    budget_exhausted: bool
    early_stop_reason: Optional[str]
    candidate_logs: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Orchestrator:
    """Main workflow coordinator for SEAE."""

    def __init__(
        self,
        registry: Optional[KernelRegistry] = None,
        verifier: Optional[Verifier] = None,
        benchmarker: Optional[Benchmarker] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
        approval_gate: Optional[ApprovalGate] = None,
        hardware_profile: Optional[HardwareProfile] = None,
    ) -> None:
        self.registry = registry or KernelRegistry()
        self.verifier = verifier or Verifier()
        self.benchmarker = benchmarker or Benchmarker()
        self.candidate_generator = candidate_generator or CandidateGenerator(seed=42)
        self.approval_gate = approval_gate or ApprovalGate(auto_approve=False)
        self.hw_profile = hardware_profile or HardwareManager.get_hardware_profile()
        self.env_fingerprint = make_environment_fingerprint(self.hw_profile.fingerprint)

    def run(
        self,
        kernel_name: str,
        size_str: str = "1920x1080",
        dtype_str: str = "uint8",
        budget: Optional[SearchBudget] = None,
        force_search: bool = False,
        **kernel_kwargs: Any,
    ) -> OrchestratorResult:
        """Execute or optimize kernel for target machine.
        
        Steps:
        1. Query cache for active verified kernel under current hardware + library environment.
        2. If hit and not forced, return cached kernel immediately with zero search cost.
        3. If miss:
           a. Measure trusted baseline (OpenCV).
           b. Generate and explore candidate space within search budget.
           c. Monitor search cost vs projected ROI; early-stop if search time exceeds projected savings.
           d. Request approval via ApprovalGate before promoting winner to cache.
        """
        budget = budget or SearchBudget()
        task_spec = TaskAnalyzer.analyze(
            kernel_name=kernel_name,
            size_str=size_str,
            dtype_str=dtype_str,
            **kernel_kwargs
        )

        kernel_params = {}
        if kernel_name == "gaussian_blur":
            kernel_params = {"ksize": task_spec.ksize, "sigma": task_spec.sigma}
        elif kernel_name == "threshold":
            kernel_params = {"thresh": task_spec.thresh, "maxval": task_spec.maxval}

        # 1. Check cache first
        if not force_search:
            cached = self.registry.get_latest_active(
                kernel_name=kernel_name,
                size_str=size_str,
                hardware_fingerprint=self.env_fingerprint,
                dtype_str=dtype_str,
                kernel_params=kernel_params,
                check_sha=True,
            )
            if cached is not None:
                return OrchestratorResult(
                    kernel_name=kernel_name,
                    size_str=size_str,
                    cache_hit=True,
                    selected_candidate_id=cached.candidate_id,
                    selected_backend=cached.backend,
                    source_code=cached.source_code,
                    baseline_median_ms=cached.median_time_ms * cached.speedup_ratio,
                    selected_median_ms=cached.median_time_ms,
                    speedup_ratio=cached.speedup_ratio,
                    search_duration_seconds=0.0,
                    candidates_evaluated=0,
                    promoted_to_cache=False,
                    budget_exhausted=False,
                    early_stop_reason=None,
                    candidate_logs=[],
                )

        # 2. Cache miss -> initiate search
        t_search_start = time.perf_counter()
        deadline = t_search_start + budget.max_seconds
        logs: List[ExecutionLogEntry] = []

        # Benchmark baseline (OpenCV)
        cv_cand = self.candidate_generator.generate_candidate(kernel_name, KernelConfig(backend="opencv"))
        cv_verif = self.verifier.verify_candidate(cv_cand.candidate_id, cv_cand.source_code, task_spec)
        if not cv_verif.all_passed:
            raise RuntimeError(f"Baseline OpenCV implementation failed verification: {cv_verif.rejection_reason}")

        cv_bench = self.benchmarker.benchmark_candidate(cv_cand.candidate_id, cv_cand.source_code, task_spec)
        baseline_ms = cv_bench.median_time_ms

        logs.append(ExecutionLogEntry(
            candidate_id=cv_cand.candidate_id,
            backend="opencv",
            verified=True,
            rejection_reason=None,
            median_time_ms=baseline_ms,
            speedup_ratio=1.0,
            warmup_compile_ms=cv_bench.warmup_compile_time_ms,
            evaluation_cost_ms=cv_bench.warmup_compile_time_ms + sum(cv_bench.all_times_ms),
        ))

        best_candidate: Candidate = cv_cand
        best_metrics: BenchmarkMetrics = cv_bench
        best_speedup: float = 1.0

        # Generate candidates based on strategy
        evaluated_count = 1
        early_stop_reason = None
        budget_exhausted = False

        if budget.strategy == "random":
            candidate_pool = self.candidate_generator.generate_random_batch(kernel_name, budget.max_candidates)
            # Remove the first opencv which we already tested
            candidate_pool = [c for c in candidate_pool if c.config.backend != "opencv"]
        elif budget.strategy == "bandit":
            arm_history: Dict[str, List[float]] = {}
            candidate_pool = []
            for _ in range(budget.max_candidates):
                cand = self.candidate_generator.sample_bandit_arm(kernel_name, arm_history)
                candidate_pool.append(cand)
        else: # Evolutionary
            initial_pop = self.candidate_generator.generate_random_batch(kernel_name, min(budget.max_candidates, 6))
            candidate_pool = initial_pop

        # Search evaluation loop
        pop_for_evolution: List[Candidate] = [cv_cand]
        cv_cand.score = 1000.0 / baseline_ms

        for cand in candidate_pool:
            now = time.perf_counter()
            search_time_spent_ms = (now - t_search_start) * 1000.0

            if now >= deadline:
                budget_exhausted = True
                early_stop_reason = f"Time budget exceeded ({budget.max_seconds:.1f}s)"
                break

            if evaluated_count >= budget.max_candidates:
                budget_exhausted = True
                early_stop_reason = f"Candidate count budget reached ({budget.max_candidates})"
                break

            # ROI Stopping Rule:
            # Stop search when search cost exceeds total projected lifetime savings:
            # search_cost_ms > (baseline_ms - best_ms) * expected_lifetime_invocations
            # Rule: Never stop before at least one novel candidate has been evaluated and compared against baseline.
            # If no novel candidate has beaten baseline yet, allow search to continue exploring unless time/candidate budget is exhausted
            # or unless a user disables continued exploration (expected_lifetime_invocations <= 10).
            best_saving_per_call = baseline_ms - best_metrics.median_time_ms
            
            # evaluated_count >= 2 guarantees baseline (cand 1) AND at least one candidate (cand 2) have been measured
            if evaluated_count >= 2:
                if best_saving_per_call > 0.0:
                    projected_lifetime_savings_ms = best_saving_per_call * budget.expected_lifetime_invocations
                    if search_time_spent_ms > projected_lifetime_savings_ms:
                        early_stop_reason = (
                            f"ROI cutoff: Search cost ({search_time_spent_ms:.1f} ms) exceeds "
                            f"projected lifetime savings ({projected_lifetime_savings_ms:.1f} ms = "
                            f"{best_saving_per_call:.3f} ms savings/call * {budget.expected_lifetime_invocations:,} expected invocations)"
                        )
                        break
                elif budget.expected_lifetime_invocations <= 10:
                    # Explicit low amortization scenario: immediately stop if no candidate produced any savings
                    early_stop_reason = (
                        f"ROI cutoff: Search cost ({search_time_spent_ms:.1f} ms) exceeds zero savings "
                        f"with limited expected invocations ({budget.expected_lifetime_invocations:,})"
                    )
                    break

            t_eval_start = time.perf_counter()

            # Step 1: Sandbox Verification
            is_quant = cand.config.quantized
            verif_report = self.verifier.verify_candidate(
                cand.candidate_id,
                cand.source_code,
                task_spec,
                is_quantized=is_quant
            )

            if not verif_report.all_passed:
                eval_cost = (time.perf_counter() - t_eval_start) * 1000.0
                logs.append(ExecutionLogEntry(
                    candidate_id=cand.candidate_id,
                    backend=cand.config.backend,
                    verified=False,
                    rejection_reason=verif_report.rejection_reason,
                    median_time_ms=None,
                    speedup_ratio=None,
                    warmup_compile_ms=0.0,
                    evaluation_cost_ms=eval_cost,
                ))
                cand.score = 0.0
                evaluated_count += 1
                continue

            # Step 2: Sandbox Benchmarking
            try:
                bench_metrics = self.benchmarker.benchmark_candidate(
                    cand.candidate_id,
                    cand.source_code,
                    task_spec,
                    is_pure_python=(cand.config.backend == "python_loops")
                )
            except Exception as e:
                eval_cost = (time.perf_counter() - t_eval_start) * 1000.0
                logs.append(ExecutionLogEntry(
                    candidate_id=cand.candidate_id,
                    backend=cand.config.backend,
                    verified=False,
                    rejection_reason=f"Benchmarking runtime exception: {e}",
                    median_time_ms=None,
                    speedup_ratio=None,
                    warmup_compile_ms=0.0,
                    evaluation_cost_ms=eval_cost,
                ))
                cand.score = 0.0
                evaluated_count += 1
                continue

            eval_cost = (time.perf_counter() - t_eval_start) * 1000.0
            cand_speedup = baseline_ms / bench_metrics.median_time_ms if bench_metrics.median_time_ms > 0 else 1.0
            cand.score = 1000.0 / bench_metrics.median_time_ms

            logs.append(ExecutionLogEntry(
                candidate_id=cand.candidate_id,
                backend=cand.config.backend,
                verified=True,
                rejection_reason=None,
                median_time_ms=bench_metrics.median_time_ms,
                speedup_ratio=cand_speedup,
                warmup_compile_ms=bench_metrics.warmup_compile_time_ms,
                evaluation_cost_ms=eval_cost,
            ))
            pop_for_evolution.append(cand)
            evaluated_count += 1

            # Update best candidate
            if bench_metrics.median_time_ms < best_metrics.median_time_ms:
                best_candidate = cand
                best_metrics = bench_metrics
                best_speedup = cand_speedup

        if evaluated_count >= budget.max_candidates:
            budget_exhausted = True
            if early_stop_reason is None:
                early_stop_reason = f"Candidate count budget reached ({budget.max_candidates})"

        search_duration = time.perf_counter() - t_search_start

        # 3. Final Interleaved Verification & Strict Selection Rule (>=30 interleaved runs)
        # Selection rule: promote only if candidate beats baseline by >10% AND outside run-to-run std.
        # Never select a slower candidate.
        promoted = False
        is_novel_best = False
        break_even_invocations = None

        if best_candidate.candidate_id != cv_cand.candidate_id and best_metrics.median_time_ms < baseline_ms:
            interleaved_res = self.benchmarker.benchmark_interleaved_comparison(
                cv_cand.source_code,
                best_candidate.source_code,
                task_spec,
                runs=30
            )

            if interleaved_res.promotable:
                is_novel_best = True
                baseline_ms = interleaved_res.cand1_mean_ms
                selected_ms = interleaved_res.cand2_mean_ms
                best_speedup = baseline_ms / selected_ms
                total_overhead_ms = (search_duration * 1000.0) + best_metrics.warmup_compile_time_ms
                per_call_saving_ms = baseline_ms - selected_ms
                break_even_invocations = total_overhead_ms / per_call_saving_ms if per_call_saving_ms > 0 else None
            else:
                best_candidate = cv_cand
                best_metrics = cv_bench
                best_speedup = 1.00
                if early_stop_reason is None:
                    early_stop_reason = f"Candidate rejected by selection rule: {interleaved_res.rejection_reason}"
        else:
            best_candidate = cv_cand
            best_metrics = cv_bench
            best_speedup = 1.00

        is_initial_baseline = (best_candidate.candidate_id == cv_cand.candidate_id)

        roi_warning = None
        if is_novel_best and break_even_invocations is not None and break_even_invocations > budget.expected_lifetime_invocations:
            roi_warning = (
                f"Negative ROI Amortization: Break-even volume is {int(break_even_invocations):,} calls, "
                f"which exceeds expected invocations ({budget.expected_lifetime_invocations:,} calls). "
                f"Search & compile overhead ({total_overhead_ms:.1f} ms) exceeds total lifetime savings "
                f"({per_call_saving_ms * budget.expected_lifetime_invocations:.1f} ms)."
            )

        if is_novel_best or is_initial_baseline:
            # Diff against baseline
            diff_str = self.registry.compute_diff(
                cv_cand.source_code,
                best_candidate.source_code,
                old_label="baseline_opencv",
                new_label=best_candidate.candidate_id
            )

            decision = self.approval_gate.request_approval(
                candidate_id=best_candidate.candidate_id,
                kernel_name=kernel_name,
                size_str=size_str,
                speedup_ratio=best_speedup,
                baseline_ms=baseline_ms,
                new_ms=best_metrics.median_time_ms,
                diff_str=diff_str,
                break_even_invocations=break_even_invocations,
                expected_invocations=budget.expected_lifetime_invocations,
                roi_warning=roi_warning,
            )

            if decision.approved:
                try:
                    self.registry.register_candidate(
                        kernel_name=kernel_name,
                        size_str=size_str,
                        hardware_fingerprint=self.env_fingerprint,
                        candidate_id=best_candidate.candidate_id,
                        backend=best_candidate.config.backend,
                        source_code=best_candidate.source_code,
                        median_time_ms=best_metrics.median_time_ms,
                        speedup_ratio=best_speedup,
                        peak_memory_mb=best_metrics.peak_memory_mb,
                        verification_passed=True,
                        dtype_str=dtype_str,
                        kernel_params=kernel_params,
                        metadata={
                            "search_duration_sec": search_duration,
                            "candidates_evaluated": evaluated_count,
                            "approved_by": decision.reviewer,
                            "approval_reason": decision.reason,
                        }
                    )
                    promoted = True
                except ValueError as err:
                    # E.g. Candidate matches a previously ROLLED_BACK version and cannot be re-promoted
                    if "ROLLED_BACK" in str(err):
                        early_stop_reason = f"Candidate won search but cannot be promoted: {err}"
                        promoted = False
                    else:
                        raise

        return OrchestratorResult(
            kernel_name=kernel_name,
            size_str=size_str,
            cache_hit=False,
            selected_candidate_id=best_candidate.candidate_id,
            selected_backend=best_candidate.config.backend,
            source_code=best_candidate.source_code,
            baseline_median_ms=baseline_ms,
            selected_median_ms=best_metrics.median_time_ms,
            speedup_ratio=best_speedup,
            search_duration_seconds=search_duration,
            candidates_evaluated=evaluated_count,
            promoted_to_cache=promoted,
            budget_exhausted=budget_exhausted,
            early_stop_reason=early_stop_reason,
            candidate_logs=[asdict(l) for l in logs],
        )
