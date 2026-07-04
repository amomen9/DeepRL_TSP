"""
Library_dp.py - Held-Karp DP memory estimation and OS-level memory checks.

Contents
--------
dp_memory_estimation
get_free_os_memory_bytes
should_skip_dp_due_to_memory
"""


@staticmethod
def dp_memory_estimation(n, method="bottom_up"):
    """Estimate the RAM needed for the Held-Karp DP tables.

    Parameters
    ----------
    n : int
        Number of cities.
    method : str
        '"bottom_up"' or '"top_down"'.

    Returns
    -------
    dict
        '{"bytes": int, "MB": float, "GB": float}'
    """
    if n <= 2:
        total = 0
    elif method == "bottom_up":
        # Compact Held-Karp cost table (NumPy float64), indexed over the n-1
        # non-depot cities: 2^{n-1} rows x (n-1) cols x 8 bytes. One buffer is
        # reused across the best/worst/expected solves, so this single array is
        # the whole DP-path peak. (Was a Python list-of-lists at ~250 B/cell,
        # plus an unmodeled ~250 B/cell environment state table that dominated
        # RAM; the state table is no longer built during the DP solve.)
        total = 2**(n - 1) * (n - 1) * 8 + 128
    elif method == "top_down":
        # Number of DP states:
        #   i=0: 2^{n-1} subsets; i=j (j>=1): 2^{n-2} subsets each
        #   total = 2^{n-1} + (n-1)*2^{n-2} = (n+1)*2^{n-2}
        num_states = (n + 1) * 2**(n - 2)
        # memo dict – key: (int,int) tuple ~72 B, value: float 28 B,
        #              dict-entry overhead ~52 B
        memo_per = 72 + 28 + 52
        # choices dict – key: same tuple, value: small list ~64 B,
        #                dict-entry overhead ~52 B
        choices_per = 72 + 64 + 52
        total = num_states * (memo_per + choices_per)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "bytes": total,
        "MB": round(total / (1024**2), 2),
        "GB": round(total / (1024**3), 4),
    }


@staticmethod
def get_free_os_memory_bytes() -> int:
    """Best-effort free available RAM in bytes.

    Prefer psutil.virtual_memory().available when available.
    Fallback to Windows GlobalMemoryStatusEx via ctypes.
    """
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    # Windows fallback
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return int(stat.ullAvailPhys)
    except Exception:
        # Unknown free memory
        return 0


@staticmethod
def should_skip_dp_due_to_memory(n: int, total_dp_runs: int, method: str = "bottom_up") -> tuple[bool, int, int]:
    """Return (skip, needed_bytes, free_bytes)."""
    free_bytes = get_free_os_memory_bytes()
    per_run = dp_memory_estimation(n, method=method).get("bytes", 0)
    needed_bytes = int(per_run) * int(max(1, total_dp_runs))
    skip = free_bytes > 0 and needed_bytes > free_bytes
    return skip, needed_bytes, free_bytes
