"""
Viral Trend Detector — Change Point Detection (pure Python/NumPy)
Implements PELT-like cumulative sum (CUSUM) algorithm without C extensions.
"""
import numpy as np
from typing import List


def _cusum_changepoints(signal: np.ndarray, penalty: float = 3.0) -> List[int]:
    """
    Binary segmentation change-point detection using cumulative sums.
    Returns list of change-point indices.
    """
    n = len(signal)
    if n < 4:
        return []

    def _cost(start: int, end: int) -> float:
        segment = signal[start:end]
        if len(segment) < 2:
            return 0.0
        return float(np.sum((segment - np.mean(segment)) ** 2))

    def _binary_seg(start: int, end: int, min_size: int = 3) -> List[int]:
        if end - start < 2 * min_size:
            return []
        best_gain = 0.0
        best_t = -1
        base_cost = _cost(start, end)
        for t in range(start + min_size, end - min_size + 1):
            gain = base_cost - (_cost(start, t) + _cost(t, end))
            if gain > best_gain:
                best_gain = gain
                best_t = t
        threshold = penalty * np.std(signal) ** 2
        if best_gain > threshold and best_t != -1:
            left = _binary_seg(start, best_t, min_size)
            right = _binary_seg(best_t, end, min_size)
            return left + [best_t] + right
        return []

    pts = _binary_seg(0, n)
    return sorted(set(pts))


def detect_changepoints(series: list, penalty: float = 3.0) -> List[int]:
    """
    Public API — takes a plain list and returns change-point indices.
    """
    arr = np.array(series, dtype=float)
    if np.std(arr) == 0:
        return []
    return _cusum_changepoints(arr, penalty=penalty)
