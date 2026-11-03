"""CLI interface for SEAE (Self-Evolving Algorithm Engine).

Commands:
  run       : Optimize/run image processing kernel on local machine.
  history   : Inspect version history and audit log for a kernel.
  rollback  : Roll back active cached kernel to last-known-good version.
  export    : Export registry database to JSON.
"""

import sys
import argparse
import json
from typing import List, Optional

from seae.orchestrator import Orchestrator, SearchBudget
from seae.approval_gate import ApprovalGate
from seae.registry import KernelRegistry, make_environment_fingerprint
from seae.hardware_manager import HardwareManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seae",
        description="SEAE: Self-Evolving Algorithm Engine - Adaptive Kernel Optimizer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. run command
    run_parser = subparsers.add_parser("run", help="Run or optimize an image processing kernel.")
    run_parser.add_argument("--kernel", type=str, required=True,
                            choices=["gaussian_blur", "threshold", "sobel", "conv2d", "fused_contrast_blur"],
                            help="Kernel type to execute/optimize.")
    run_parser.add_argument("--size", type=str, default="1920x1080",
                            help="Input resolution in WxH format (e.g. 1920x1080).")
    run_parser.add_argument("--dtype", type=str, default="uint8",
                            help="Image array data type (e.g. uint8, float32).")
    run_parser.add_argument("--max-candidates", type=int, default=10,
                            help="Maximum candidate budget to evaluate.")
    run_parser.add_argument("--max-seconds", type=float, default=25.0,
                            help="Maximum wall-clock search budget in seconds.")
    run_parser.add_argument("--expected-invocations", type=int, default=10000,
                            help="Projected lifetime invocations to calculate search ROI and early-stopping.")
    run_parser.add_argument("--strategy", type=str, default="evolutionary",
                            choices=["evolutionary", "bandit", "random"],
                            help="Search strategy.")
    run_parser.add_argument("--auto-approve", action="store_true",
                            help="Bypass interactive approval gate prompt.")
    run_parser.add_argument("--force-search", action="store_true",
                            help="Ignore cached entry and trigger new optimization search.")

    # Kernel specific parameters
    run_parser.add_argument("--ksize", type=int, default=5, help="Kernel window size (e.g. 3, 5).")
    run_parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma.")
    run_parser.add_argument("--thresh", type=float, default=128.0, help="Binary threshold value.")
    run_parser.add_argument("--maxval", type=float, default=255.0, help="Binary threshold maxval.")

    # 2. history command
    hist_parser = subparsers.add_parser("history", help="Show version history of a kernel.")
    hist_parser.add_argument("--kernel", type=str, required=True, help="Kernel name.")
    hist_parser.add_argument("--size", type=str, default="1920x1080", help="Resolution.")
    hist_parser.add_argument("--dtype", type=str, default="uint8", help="Dtype.")

    # 3. rollback command
    rb_parser = subparsers.add_parser("rollback", help="Roll back active kernel to previous version.")
    rb_parser.add_argument("--kernel", type=str, required=True, help="Kernel name.")
    rb_parser.add_argument("--size", type=str, default="1920x1080", help="Resolution.")
    rb_parser.add_argument("--dtype", type=str, default="uint8", help="Dtype.")

    # 4. export command
    exp_parser = subparsers.add_parser("export", help="Export registry to JSON file.")
    exp_parser.add_argument("--output", type=str, default="seae_registry_export.json", help="Output JSON path.")

    return parser


def handle_run(args: argparse.Namespace) -> int:
    approval = ApprovalGate(auto_approve=args.auto_approve)
    orchestrator = Orchestrator(approval_gate=approval)

    budget = SearchBudget(
        max_candidates=args.max_candidates,
        max_seconds=args.max_seconds,
        strategy=args.strategy,
        expected_lifetime_invocations=args.expected_invocations,
    )

    print("=" * 65)
    print("                SEAE - KERNEL EXECUTION ENGINE")
    print("=" * 65)
    print(f"Kernel Target       : {args.kernel}")
    print(f"Resolution          : {args.size} ({args.dtype})")
    print(f"Hardware Arch       : {orchestrator.hw_profile.fingerprint}")
    print(f"Auto-Approve        : {args.auto_approve}")
    print(f"Expected Invocations: {args.expected_invocations:,}")
    print(f"Search Strategy     : {args.strategy} (Budget: {args.max_candidates} cands / {args.max_seconds}s)")
    print("-" * 65)

    res = orchestrator.run(
        kernel_name=args.kernel,
        size_str=args.size,
        dtype_str=args.dtype,
        budget=budget,
        force_search=args.force_search,
        ksize=args.ksize,
        sigma=args.sigma,
        thresh=args.thresh,
        maxval=args.maxval,
    )

    print("\nEXECUTION SUMMARY:")
    print(f"Cache Status       : {'[CACHE HIT]' if res.cache_hit else '[SEARCH EXECUTED]'}")
    print(f"Selected Backend   : {res.selected_backend} ({res.selected_candidate_id})")
    print(f"Baseline Latency   : {res.baseline_median_ms:.3f} ms")
    print(f"Selected Latency   : {res.selected_median_ms:.3f} ms")
    print(f"Speedup Ratio      : {res.speedup_ratio:.2f}x")
    if not res.cache_hit:
        print(f"Candidates Tested  : {res.candidates_evaluated}")
        print(f"Search Duration    : {res.search_duration_seconds:.2f} s")
        print(f"Promoted to Cache  : {res.promoted_to_cache}")
        if res.early_stop_reason:
            print(f"Stopping Condition : {res.early_stop_reason}")

    print("=" * 65)
    return 0


def handle_history(args: argparse.Namespace) -> int:
    registry = KernelRegistry()
    hw = HardwareManager.get_hardware_profile()
    env_fp = make_environment_fingerprint(hw.fingerprint)

    kernel_params = {}
    if args.kernel == "threshold":
        kernel_params = {"thresh": 128.0, "maxval": 255.0}
    elif args.kernel == "gaussian_blur":
        kernel_params = {"ksize": 5, "sigma": 1.0}

    history = registry.get_history(
        kernel_name=args.kernel,
        size_str=args.size,
        hardware_fingerprint=env_fp,
        dtype_str=args.dtype,
        kernel_params=kernel_params,
    )

    print(f"Version History for {args.kernel} [{args.size}] on {env_fp}:")
    if not history:
        print("  No records found in registry cache.")
        return 0

    for e in history:
        print(f"  v{e.version:<2} | Status: {e.status:<10} | Backend: {e.backend:<12} | "
              f"Time: {e.median_time_ms:.3f} ms ({e.speedup_ratio:.2f}x) | SHA: {e.code_sha256[:8]}")
    return 0


def handle_rollback(args: argparse.Namespace) -> int:
    registry = KernelRegistry()
    hw = HardwareManager.get_hardware_profile()
    env_fp = make_environment_fingerprint(hw.fingerprint)

    kernel_params = {}
    if args.kernel == "threshold":
        kernel_params = {"thresh": 128.0, "maxval": 255.0}
    elif args.kernel == "gaussian_blur":
        kernel_params = {"ksize": 5, "sigma": 1.0}

    entry, status = registry.rollback(
        kernel_name=args.kernel,
        size_str=args.size,
        hardware_fingerprint=env_fp,
        dtype_str=args.dtype,
        kernel_params=kernel_params,
    )
    if status == "RESTORED_PREVIOUS" and entry is not None:
        print(f"Rollback successful: Reactivated v{entry.version} ({entry.candidate_id}, backend: {entry.backend}).")
    elif status == "FALLBACK_TO_BASELINE":
        print(f"Rollback successful: Active entry marked ROLLED_BACK. Cache falls back to baseline ground truth.")
    else:
        print(f"Rollback aborted: {status}")
    return 0


def handle_export(args: argparse.Namespace) -> int:
    registry = KernelRegistry()
    registry.export_to_json(args.output)
    print(f"Exported registry cache to {args.output}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return handle_run(args)
    elif args.command == "history":
        return handle_history(args)
    elif args.command == "rollback":
        return handle_rollback(args)
    elif args.command == "export":
        return handle_export(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
