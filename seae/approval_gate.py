"""Human-in-the-loop approval gate for SEAE candidate promotion.

Promoted candidates must pass:
1. Strict numerical verification.
2. Verified benchmark speedup over active baseline.
3. Explicit human review displaying the unified diff and speedup metrics,
   unless bypassed with an explicit --auto-approve flag (used for automated testing/CI).
"""

import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ApprovalDecision:
    """Outcome of human approval evaluation."""
    approved: bool
    reviewer: str
    reason: str


class ApprovalGate:
    """Manages diff review and human approval before kernel promotion."""

    def __init__(self, auto_approve: bool = False) -> None:
        self.auto_approve = auto_approve

    def request_approval(
        self,
        candidate_id: str,
        kernel_name: str,
        size_str: str,
        speedup_ratio: float,
        baseline_ms: float,
        new_ms: float,
        diff_str: str,
        break_even_invocations: Optional[float] = None,
        expected_invocations: Optional[int] = None,
        roi_warning: Optional[str] = None,
    ) -> ApprovalDecision:
        """Prompt user or evaluate auto-approve flag."""
        if self.auto_approve:
            reason = f"Auto-approved via --auto-approve (Speedup: {speedup_ratio:.2f}x: {baseline_ms:.3f}ms -> {new_ms:.3f}ms)"
            if roi_warning:
                reason += f" [WARNING: {roi_warning}]"
            return ApprovalDecision(
                approved=True,
                reviewer="auto_approve_flag",
                reason=reason,
            )

        # Interactive CLI review
        print("\n" + "=" * 70)
        print("  SEAE KERNEL PROMOTION APPROVAL GATE")
        print("=" * 70)
        print(f"Candidate ID : {candidate_id}")
        print(f"Task         : {kernel_name} [{size_str}]")
        print(f"Speedup      : {speedup_ratio:.2f}x ({baseline_ms:.3f} ms -> {new_ms:.3f} ms)")
        if break_even_invocations is not None and expected_invocations is not None:
            if break_even_invocations == float("inf"):
                print(f"Break-Even   : N/A (Baseline selected or no savings)")
            else:
                print(f"Break-Even   : {int(break_even_invocations):,} invocations (Expected: {expected_invocations:,})")
        else:
            print("Break-Even   : N/A")
        if roi_warning:
            print(f"\n[!WARNING] {roi_warning}")
        print("-" * 70)
        print("PROPOSED CODE DIFF:")
        print(diff_str if diff_str.strip() else "(Initial baseline implementation)")
        print("-" * 70)

        # Prompt user via stdin (interactive terminal or piped input)
        if sys.stdin:
            try:
                resp = input("Promote this candidate to active registry cache? [y/N]: ").strip().lower()
                if resp in ("y", "yes"):
                    return ApprovalDecision(approved=True, reviewer="human_cli", reason="Accepted by human reviewer.")
                else:
                    return ApprovalDecision(approved=False, reviewer="human_cli", reason="Rejected by human reviewer.")
            except (EOFError, KeyboardInterrupt, OSError):
                return ApprovalDecision(approved=False, reviewer="human_cli", reason="Rejected (aborted / non-interactive environment).")

        # If no stdin is available: safe rejection
        return ApprovalDecision(
            approved=False,
            reviewer="approval_gate",
            reason="No stdin stream available and --auto-approve flag omitted. Promotion blocked."
        )
