"""Hardware resource and capability manager for SEAE.

Detects host CPU architecture, physical/logical core counts, SIMD instructions,
RAM capacity, and GPU availability. Provides an abstraction interface for
future hardware accelerators (such as FPGA / TPU / NPU backends).
"""

import os
import platform
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple


@dataclass(frozen=True)
class HardwareProfile:
    """Snapshot of system hardware capabilities."""
    fingerprint: str
    arch: str
    machine: str
    cpu_model: str
    logical_cores: int
    physical_cores: int
    ram_gb: float
    simd_features: List[str]
    has_gpu: bool
    gpu_devices: List[str]
    cpu_freq_mhz: Optional[float] = None
    power_plan: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AcceleratorBackendInterface:
    """Abstract interface for future accelerator execution engines (FPGA/NPU)."""

    def is_available(self) -> bool:
        """Check if accelerator hardware runtime is present."""
        raise NotImplementedError

    def get_device_info(self) -> Dict[str, Any]:
        """Return device identifier and capabilities."""
        raise NotImplementedError

    def compile_kernel(self, source_code: str, options: Optional[Dict[str, Any]] = None) -> Any:
        """Synthesize/compile kernel targeting accelerator bitstream or binary."""
        raise NotImplementedError

    def execute(self, compiled_binary: Any, input_data: Any) -> Any:
        """Dispatch workload to accelerator."""
        raise NotImplementedError


class FPGAPlaceholderBackend(AcceleratorBackendInterface):
    """Placeholder backend for future FPGA support (e.g. Xilinx / Intel OpenCL/HLS)."""

    def is_available(self) -> bool:
        return False

    def get_device_info(self) -> Dict[str, Any]:
        return {"status": "unsupported", "backend": "FPGA_OpenCL_Stub"}

    def compile_kernel(self, source_code: str, options: Optional[Dict[str, Any]] = None) -> Any:
        raise RuntimeError("FPGA backend not configured in current environment.")

    def execute(self, compiled_binary: Any, input_data: Any) -> Any:
        raise RuntimeError("FPGA backend not configured.")


class HardwareManager:
    """Hardware detector and profiler."""

    _cached_profile: Optional[HardwareProfile] = None

    @classmethod
    def detect_simd(cls) -> List[str]:
        """Detect SIMD capabilities of the host CPU (including AVX, AVX2, SSE tiers, NEON).
        
        Uses direct hardware CPUID reports via NumPy and validates OpenCV runtime feature
        detection, explicitly discarding compiled-in but unsupported features (which OpenCV
        flags with a trailing '?').
        """
        features: List[str] = []
        machine = platform.machine().lower()

        # 1. Primary: Direct CPUID flags from NumPy umath hardware detection
        np_features_found = False
        try:
            import numpy._core._multiarray_umath as mumat
            if hasattr(mumat, "__cpu_features__"):
                for feat, enabled in mumat.__cpu_features__.items():
                    if enabled:
                        features.append(feat)
                np_features_found = True
        except Exception:
            try:
                import numpy.core._multiarray_umath as mumat_legacy
                if hasattr(mumat_legacy, "__cpu_features__"):
                    for feat, enabled in mumat_legacy.__cpu_features__.items():
                        if enabled:
                            features.append(feat)
                    np_features_found = True
            except Exception:
                pass

        # 2. Secondary: OpenCV runtime feature check
        # NOTE: cv2.getCPUFeaturesLine() flags unsupported build features with '?' (e.g. '*AVX512-SKX?').
        # Only tokens without trailing '?' are supported by the host CPU!
        try:
            import cv2
            if hasattr(cv2, "getCPUFeaturesLine"):
                line = cv2.getCPUFeaturesLine()
                for token in line.split():
                    # If token ends with '?', the CPU does NOT support it at runtime
                    if token.endswith("?"):
                        continue
                    clean = token.lstrip("*").upper()
                    if clean and clean not in features:
                        # Avoid adding variants if normalized exists
                        if clean not in [f.replace(".", "") for f in features]:
                            features.append(clean)
        except Exception:
            pass

        # 3. Deduplicate and normalize feature naming (e.g. SSE41 -> SSE4.1)
        normalized: List[str] = []
        for f in features:
            f_norm = f.upper().strip()
            if f_norm == "SSE41":
                f_norm = "SSE4.1"
            elif f_norm == "SSE42":
                f_norm = "SSE4.2"
            if f_norm not in normalized:
                normalized.append(f_norm)

        # Baseline architectural guarantees
        if not np_features_found:
            if "64" in machine or "x86" in machine or "amd64" in machine:
                for base in ["SSE", "SSE2", "SSE3", "SSSE3", "SSE4.1", "SSE4.2"]:
                    if base not in normalized:
                        normalized.append(base)
            elif "arm" in machine or "aarch64" in machine:
                if "NEON" not in normalized:
                    normalized.append("NEON")

        return sorted(list(set(normalized)))

    @classmethod
    def detect_gpu(cls) -> Tuple[bool, List[str]]:
        """Check for GPU presence (CUDA or DirectML/OpenCL)."""
        has_gpu = False
        gpu_devices: List[str] = []
        try:
            import torch
            if torch.cuda.is_available():
                has_gpu = True
                count = torch.cuda.device_count()
                for i in range(count):
                    gpu_devices.append(torch.cuda.get_device_name(i))
        except Exception:
            pass

        return has_gpu, gpu_devices

    @classmethod
    def get_hardware_profile(cls, refresh: bool = False) -> HardwareProfile:
        """Generate and cache hardware profile."""
        if cls._cached_profile is not None and not refresh:
            return cls._cached_profile

        import psutil

        logical_cores = os.cpu_count() or 1
        physical_cores = psutil.cpu_count(logical=False) or logical_cores
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        arch = platform.architecture()[0]
        machine = platform.machine()
        cpu_model = platform.processor() or "Unknown CPU"

        simd_features = cls.detect_simd()
        has_gpu, gpu_devices = cls.detect_gpu()

        # Build consistent fingerprint using CPU model, core counts, and top SIMD capability
        top_simd = "std"
        for check in ["AVX512F", "AVX2", "AVX", "SSE4.2", "SSE4.1", "SSSE3", "SSE3", "SSE2", "NEON"]:
            if check in simd_features or check.replace(".", "") in [s.replace(".", "") for s in simd_features]:
                top_simd = check.replace(".", "_")
                break

        # Sanitize CPU model name for fingerprint
        import re
        clean_cpu = re.sub(r"[^a-zA-Z0-9]+", "-", cpu_model).strip("-")
        fingerprint = f"{clean_cpu}-{physical_cores}p{logical_cores}c-{top_simd}"

        # Probe CPU frequency and active power plan
        cpu_freq_mhz = None
        try:
            cfreq = psutil.cpu_freq()
            if cfreq:
                cpu_freq_mhz = round(cfreq.current, 1)
        except Exception:
            pass

        power_plan = None
        try:
            p_out = subprocess.check_output(["powercfg", "/getactivescheme"], text=True)
            if "(" in p_out and ")" in p_out:
                power_plan = p_out.split("(")[-1].split(")")[0].strip()
            else:
                power_plan = p_out.strip()
        except Exception:
            pass

        profile = HardwareProfile(
            fingerprint=fingerprint,
            arch=arch,
            machine=machine,
            cpu_model=cpu_model,
            logical_cores=logical_cores,
            physical_cores=physical_cores,
            ram_gb=ram_gb,
            simd_features=simd_features,
            has_gpu=has_gpu,
            gpu_devices=gpu_devices,
            cpu_freq_mhz=cpu_freq_mhz,
            power_plan=power_plan,
        )
        cls._cached_profile = profile
        return profile
