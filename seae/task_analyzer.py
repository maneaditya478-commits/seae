"""Task Analyzer for SEAE.

Extracts features and specifications for incoming kernel optimization tasks.
Features include image resolution, aspect ratio, memory footprint,
data type range, and kernel metadata.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional
import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    """Specification of an image processing kernel task."""
    kernel_name: str
    height: int
    width: int
    channels: int
    dtype_str: str
    ksize: int = 5
    sigma: float = 1.0
    thresh: float = 128.0
    maxval: float = 255.0
    alpha: float = 1.5
    beta: float = 10.0
    extra_params: Dict[str, Any] = None  # type: ignore

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.channels == 1:
            return (self.height, self.width)
        return (self.height, self.width, self.channels)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.dtype_str)

    @property
    def total_pixels(self) -> int:
        return self.height * self.width

    @property
    def memory_bytes(self) -> int:
        itemsize = np.dtype(self.dtype_str).itemsize
        return self.total_pixels * self.channels * itemsize

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_resolution(size_str: str) -> Tuple[int, int]:
    """Parse resolution string such as '1920x1080' or '512x512' into (height, width).
    
    Standard convention for 'WxH' (e.g. 1920x1080 -> width 1920, height 1080).
    """
    cleaned = size_str.lower().strip()
    if "x" not in cleaned:
        raise ValueError(f"Invalid size format: '{size_str}'. Expected format like '1920x1080'.")
    parts = cleaned.split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid size format: '{size_str}'. Expected format like '1920x1080'.")
    w = int(parts[0].strip())
    h = int(parts[1].strip())
    return h, w


class TaskAnalyzer:
    """Extracts features and constraints from an image processing task."""

    @staticmethod
    def analyze(
        kernel_name: str,
        size_str: str = "1920x1080",
        channels: int = 1,
        dtype_str: str = "uint8",
        ksize: int = 5,
        sigma: float = 1.0,
        **kwargs: Any
    ) -> TaskSpec:
        """Create a TaskSpec from user parameters."""
        h, w = parse_resolution(size_str)
        return TaskSpec(
            kernel_name=kernel_name,
            height=h,
            width=w,
            channels=channels,
            dtype_str=dtype_str,
            ksize=ksize,
            sigma=sigma,
            **kwargs
        )

    @staticmethod
    def analyze_array(
        kernel_name: str,
        img: np.ndarray,
        ksize: int = 5,
        sigma: float = 1.0,
        **kwargs: Any
    ) -> TaskSpec:
        """Extract TaskSpec directly from a sample numpy array."""
        if img.ndim == 2:
            h, w = img.shape
            c = 1
        elif img.ndim == 3:
            h, w, c = img.shape
        else:
            raise ValueError(f"Input image must be 2D or 3D, got ndim={img.ndim}")

        return TaskSpec(
            kernel_name=kernel_name,
            height=h,
            width=w,
            channels=c,
            dtype_str=str(img.dtype),
            ksize=ksize,
            sigma=sigma,
            **kwargs
        )
