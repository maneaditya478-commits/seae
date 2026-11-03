"""Isolated process execution sandbox for running untrusted kernel candidates.

LIMITATIONS & SECURITY SCOPE:
This sandbox provides process, resource, and execution isolation for prototype
algorithm discovery (preventing crashes, memory leaks, hangs, and accidental network
connections from disturbing the main process). It is an engineering sandbox,
NOT an adversarial cryptographic security boundary. It should not be used to run
hostile multi-tenant untrusted user-uploaded payloads without OS-level containerization
(e.g., Docker, seccomp, gVisor, or Windows AppContainers).

Key properties:
- Subprocess isolation (never executes candidate code in the host process).
- Timeout enforcement via subprocess termination / kill tree.
- Memory limit tracking and enforcement (via psutil process tree monitoring and peak RSS tracking).
- Network access blocked: overrides socket creation inside worker process.
- Error handling: catches syntax errors, compilation bugs, exceptions, and hangs.
"""

import os
import sys
import json
import base64
import tempfile
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List
import numpy as np


@dataclass
class SandboxResult:
    """Result returned by executing code inside the sandbox."""
    success: bool
    stdout: str
    stderr: str
    error_message: Optional[str] = None
    timed_out: bool = False
    oom_killed: bool = False
    execution_time_ms: float = 0.0
    peak_rss_mb: float = 0.0
    output_array: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.output_array is not None:
            d["output_shape"] = self.output_array.shape
            d["output_dtype"] = str(self.output_array.dtype)
            del d["output_array"]
        return d


@dataclass
class BatterySandboxResult:
    """Result returned by executing an entire test battery in a single sandbox subprocess."""
    success: bool
    stdout: str
    stderr: str
    error_message: Optional[str] = None
    timed_out: bool = False
    oom_killed: bool = False
    execution_time_ms: float = 0.0
    peak_rss_mb: float = 0.0
    output_arrays: Optional[Dict[str, np.ndarray]] = None


RUNNER_BOOTSTRAP_TEMPLATE = r'''
import sys
import os
import json
import socket
import traceback
import numpy as np

# 1. Network isolation inside the runner subprocess
def _block_network(*args, **kwargs):
    raise PermissionError("Network access is strictly prohibited inside the SEAE sandbox.")

socket.socket = _block_network
socket.create_connection = _block_network
socket.getaddrinfo = _block_network

def main():
    if len(sys.argv) < 3:
        print("Usage: runner.py <candidate_py_path> <io_dir>", file=sys.stderr)
        sys.exit(1)

    candidate_file = sys.argv[1]
    io_dir = sys.argv[2]
    
    input_file = os.path.join(io_dir, "input.npz")
    output_file = os.path.join(io_dir, "output.npz")
    status_file = os.path.join(io_dir, "status.json")

    try:
        # Load inputs
        npz = np.load(input_file)
        img = npz["img"]
        kwargs = {}
        for key in npz.files:
            if key != "img":
                val = npz[key]
                if val.ndim == 0:
                    kwargs[key] = val.item()
                else:
                    kwargs[key] = val

        # Load candidate module from disk (ensures Numba cache=True locator is available)
        import importlib.util
        spec_mod = importlib.util.spec_from_file_location("candidate_module", candidate_file)
        cand_mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(cand_mod)

        if not hasattr(cand_mod, "kernel_entry") or not callable(getattr(cand_mod, "kernel_entry")):
            raise ValueError("Candidate does not define callable 'kernel_entry(img, **kwargs)'.")

        kernel_func = cand_mod.kernel_entry

        # Run kernel execution
        out = kernel_func(img, **kwargs)
        if not isinstance(out, np.ndarray):
            out = np.asarray(out)

        # Save output
        np.savez_compressed(output_file, out=out)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"status": "SUCCESS", "error": None}, f)

    except Exception as exc:
        err_msg = traceback.format_exc()
        try:
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({"status": "ERROR", "error": str(exc), "traceback": err_msg}, f)
        except Exception:
            pass
        print(err_msg, file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
'''


BATTERY_BOOTSTRAP_TEMPLATE = r'''
import sys
import os
import json
import socket
import traceback
import numpy as np

def _block_network(*args, **kwargs):
    raise PermissionError("Network access is strictly prohibited inside the SEAE sandbox.")

socket.socket = _block_network
socket.create_connection = _block_network
socket.getaddrinfo = _block_network

def main():
    if len(sys.argv) < 3:
        print("Usage: runner.py <candidate_py_path> <io_dir>", file=sys.stderr)
        sys.exit(1)

    candidate_file = sys.argv[1]
    io_dir = sys.argv[2]

    input_file = os.path.join(io_dir, "battery_inputs.npz")
    output_file = os.path.join(io_dir, "battery_outputs.npz")
    status_file = os.path.join(io_dir, "battery_status.json")

    try:
        npz = np.load(input_file)
        kwargs = {}
        for key in npz.files:
            if key.startswith("kwarg_"):
                k_name = key[len("kwarg_"):]
                val = npz[key]
                kwargs[k_name] = val.item() if val.ndim == 0 else val

        import importlib.util
        spec_mod = importlib.util.spec_from_file_location("candidate_module", candidate_file)
        cand_mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(cand_mod)

        if not hasattr(cand_mod, "kernel_entry") or not callable(getattr(cand_mod, "kernel_entry")):
            raise ValueError("Candidate does not define callable 'kernel_entry(img, **kwargs)'.")

        kernel_func = cand_mod.kernel_entry

        outputs = {}
        for key in npz.files:
            if key.startswith("test_"):
                t_name = key[len("test_"):]
                inp = npz[key]
                out = kernel_func(inp, **kwargs)
                if not isinstance(out, np.ndarray):
                    out = np.asarray(out)
                outputs[t_name] = out

        np.savez_compressed(output_file, **outputs)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"status": "SUCCESS", "error": None}, f)

    except Exception as exc:
        err_msg = traceback.format_exc()
        try:
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({"status": "ERROR", "error": str(exc), "traceback": err_msg}, f)
        except Exception:
            pass
        print(err_msg, file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
'''


def get_default_python_executable() -> str:
    """Find the best Python executable across platforms.
    
    Checks for active virtual environments, local .venv, or falls back to sys.executable.
    """
    if "VIRTUAL_ENV" in os.environ:
        venv_dir = os.environ["VIRTUAL_ENV"]
        if sys.platform == "win32":
            cand = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            cand = os.path.join(venv_dir, "bin", "python")
        if os.path.exists(cand):
            return cand

    # Check local workspace .venv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sys.platform == "win32":
        local_cand = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    else:
        local_cand = os.path.join(base_dir, ".venv", "bin", "python")

    if os.path.exists(local_cand):
        return local_cand

    return sys.executable


class Sandbox:
    """Manages secure out-of-process candidate execution."""

    def __init__(
        self,
        timeout_seconds: float = 3.0,
        memory_limit_mb: float = 1024.0,
        python_executable: Optional[str] = None
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.python_executable = python_executable or get_default_python_executable()

    def run_candidate(
        self,
        code_str: str,
        input_image: np.ndarray,
        **kwargs: Any
    ) -> SandboxResult:
        """Run candidate source code in an isolated subprocess.
        
        Args:
            code_str: Python source string defining `def kernel_entry(img, **kwargs)`.
            input_image: Numpy array passed as first argument.
            kwargs: Extra parameters (e.g. ksize, sigma, thresh).
            
        Returns:
            SandboxResult with execution statistics and output array.
        """
        import psutil

        with tempfile.TemporaryDirectory(prefix="seae_sandbox_") as tmpdir:
            candidate_path = os.path.join(tmpdir, "candidate.py")
            bootstrap_path = os.path.join(tmpdir, "runner.py")
            input_path = os.path.join(tmpdir, "input.npz")
            output_path = os.path.join(tmpdir, "output.npz")
            status_path = os.path.join(tmpdir, "status.json")

            with open(candidate_path, "w", encoding="utf-8") as f:
                f.write(code_str)

            with open(bootstrap_path, "w", encoding="utf-8") as f:
                f.write(RUNNER_BOOTSTRAP_TEMPLATE)

            # Package input arguments
            save_dict = {"img": input_image}
            for k, v in kwargs.items():
                save_dict[k] = np.array(v)
            np.savez_compressed(input_path, **save_dict)

            cmd = [self.python_executable, bootstrap_path, candidate_path, tmpdir]
            
            # Windows creation flags for clean process isolation
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW

            t_start = time.perf_counter()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags
            )

            timed_out = False
            oom_killed = False
            peak_rss_mb = 0.0

            try:
                ps_proc = psutil.Process(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                ps_proc = None

            poll_interval = 0.02
            deadline = t_start + self.timeout_seconds

            while True:
                ret = proc.poll()
                if ret is not None:
                    break

                curr_time = time.perf_counter()
                if curr_time > deadline:
                    timed_out = True
                    self._kill_process_tree(proc.pid)
                    break

                # Memory monitoring
                if ps_proc is not None:
                    try:
                        mem_info = ps_proc.memory_info()
                        current_mb = mem_info.rss / (1024 * 1024)
                        if current_mb > peak_rss_mb:
                            peak_rss_mb = current_mb

                        if current_mb > self.memory_limit_mb:
                            oom_killed = True
                            self._kill_process_tree(proc.pid)
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                time.sleep(poll_interval)

            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000.0

            stdout, stderr = "", ""
            try:
                stdout, stderr = proc.communicate(timeout=0.5)
            except Exception:
                pass

            if timed_out:
                return SandboxResult(
                    success=False,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=f"Process timed out after {self.timeout_seconds:.2f}s",
                    timed_out=True,
                    execution_time_ms=elapsed_ms,
                    peak_rss_mb=peak_rss_mb,
                )

            if oom_killed:
                return SandboxResult(
                    success=False,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=f"Memory limit exceeded ({peak_rss_mb:.1f}MB > {self.memory_limit_mb:.1f}MB)",
                    oom_killed=True,
                    execution_time_ms=elapsed_ms,
                    peak_rss_mb=peak_rss_mb,
                )

            # Read status and output
            status_data = {}
            if os.path.exists(status_path):
                try:
                    with open(status_path, "r", encoding="utf-8") as f:
                        status_data = json.load(f)
                except Exception as exc:
                    status_data = {"status": "ERROR", "error": f"Failed to parse status: {exc}"}

            if status_data.get("status") == "SUCCESS" and os.path.exists(output_path):
                try:
                    with np.load(output_path) as out_npz:
                        out_arr = np.array(out_npz["out"], copy=True)
                    return SandboxResult(
                        success=True,
                        stdout=stdout,
                        stderr=stderr,
                        error_message=None,
                        execution_time_ms=elapsed_ms,
                        peak_rss_mb=peak_rss_mb,
                        output_array=out_arr,
                    )
                except Exception as exc:
                    return SandboxResult(
                        success=False,
                        stdout=stdout,
                        stderr=stderr,
                        error_message=f"Failed loading output array: {exc}",
                        execution_time_ms=elapsed_ms,
                        peak_rss_mb=peak_rss_mb,
                    )

            err_msg = status_data.get("error") or stderr or f"Process exited with code {proc.returncode}"
            return SandboxResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                error_message=err_msg,
                execution_time_ms=elapsed_ms,
                peak_rss_mb=peak_rss_mb,
            )

    def run_candidate_battery(
        self,
        code_str: str,
        test_inputs: List[Tuple[str, np.ndarray]],
        **kwargs: Any
    ) -> BatterySandboxResult:
        """Execute an entire test battery inside a SINGLE subprocess for fast verification.

        Avoids spawning 6 individual Python processes per candidate, reducing process
        spawn overhead by over 80%.
        """
        import psutil

        with tempfile.TemporaryDirectory(prefix="seae_battery_") as tmpdir:
            candidate_path = os.path.join(tmpdir, "candidate.py")
            bootstrap_path = os.path.join(tmpdir, "battery_runner.py")
            input_path = os.path.join(tmpdir, "battery_inputs.npz")
            output_path = os.path.join(tmpdir, "battery_outputs.npz")
            status_path = os.path.join(tmpdir, "battery_status.json")

            with open(candidate_path, "w", encoding="utf-8") as f:
                f.write(code_str)

            with open(bootstrap_path, "w", encoding="utf-8") as f:
                f.write(BATTERY_BOOTSTRAP_TEMPLATE)

            # Package all inputs into single npz
            save_dict = {}
            for t_name, inp_arr in test_inputs:
                save_dict[f"test_{t_name}"] = inp_arr
            for k, v in kwargs.items():
                save_dict[f"kwarg_{k}"] = np.array(v)
            np.savez_compressed(input_path, **save_dict)

            cmd = [self.python_executable, bootstrap_path, candidate_path, tmpdir]
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

            t_start = time.perf_counter()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags
            )

            timed_out = False
            oom_killed = False
            peak_rss_mb = 0.0

            try:
                ps_proc = psutil.Process(proc.pid)
            except Exception:
                ps_proc = None

            poll_interval = 0.02
            deadline = t_start + max(self.timeout_seconds, 6.0)

            while True:
                ret = proc.poll()
                if ret is not None:
                    break

                curr_time = time.perf_counter()
                if curr_time > deadline:
                    timed_out = True
                    self._kill_process_tree(proc.pid)
                    break

                if ps_proc is not None:
                    try:
                        mem_info = ps_proc.memory_info()
                        current_mb = mem_info.rss / (1024 * 1024)
                        if current_mb > peak_rss_mb:
                            peak_rss_mb = current_mb
                        if current_mb > self.memory_limit_mb:
                            oom_killed = True
                            self._kill_process_tree(proc.pid)
                            break
                    except Exception:
                        pass
                time.sleep(poll_interval)

            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000.0

            stdout, stderr = "", ""
            try:
                stdout, stderr = proc.communicate(timeout=0.5)
            except Exception:
                pass

            if timed_out:
                return BatterySandboxResult(
                    success=False,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=f"Process timed out after {self.timeout_seconds:.2f}s",
                    timed_out=True,
                    execution_time_ms=elapsed_ms,
                    peak_rss_mb=peak_rss_mb,
                )

            if oom_killed:
                return BatterySandboxResult(
                    success=False,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=f"Memory limit exceeded ({self.memory_limit_mb:.1f} MB)",
                    oom_killed=True,
                    execution_time_ms=elapsed_ms,
                    peak_rss_mb=peak_rss_mb,
                )

            status_data = {}
            if os.path.exists(status_path):
                try:
                    with open(status_path, "r", encoding="utf-8") as f:
                        status_data = json.load(f)
                except Exception as exc:
                    status_data = {"status": "ERROR", "error": f"Failed to parse status: {exc}"}

            if status_data.get("status") == "SUCCESS" and os.path.exists(output_path):
                try:
                    outputs = {}
                    with np.load(output_path) as out_npz:
                        for k in out_npz.files:
                            outputs[k] = np.array(out_npz[k], copy=True)
                    return BatterySandboxResult(
                        success=True,
                        stdout=stdout,
                        stderr=stderr,
                        error_message=None,
                        execution_time_ms=elapsed_ms,
                        peak_rss_mb=peak_rss_mb,
                        output_arrays=outputs,
                    )
                except Exception as exc:
                    return BatterySandboxResult(
                        success=False,
                        stdout=stdout,
                        stderr=stderr,
                        error_message=f"Failed loading output arrays: {exc}",
                        execution_time_ms=elapsed_ms,
                        peak_rss_mb=peak_rss_mb,
                    )

            err_msg = status_data.get("error") or stderr or f"Process exited with code {proc.returncode}"
            return BatterySandboxResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                error_message=err_msg,
                execution_time_ms=elapsed_ms,
                peak_rss_mb=peak_rss_mb,
            )

    @staticmethod
    def _kill_process_tree(pid: int) -> None:
        """Terminate a process and all its child processes."""
        import psutil
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except Exception:
                    pass
            parent.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
