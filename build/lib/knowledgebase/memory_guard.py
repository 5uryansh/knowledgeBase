"""Stop cleanly before the OS OOM-killer does (Linux/WSL; a no-op elsewhere)."""
from __future__ import annotations

import os
import sys


def _read_value_kb(path: str, key: str) -> int:
    with open(path, "r") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1])
    raise ValueError(f"{key} not found in {path}")


def current_memory_percent() -> float | None:
    try:
        return _read_value_kb("/proc/self/status", "VmRSS:") / _read_value_kb("/proc/meminfo", "MemTotal:") * 100
    except (OSError, ValueError):
        return None


def check_memory_or_exit(max_percent: float | None = None, context: str = "") -> None:
    limit = max_percent if max_percent is not None else float(os.environ.get("KB_MAX_MEMORY_PERCENT", 60))
    used = current_memory_percent()
    if used is None or used < limit:
        return
    label = f" ({context})" if context else ""
    print(
        f"Memory guard{label}: process is using {used:.1f}% of total RAM (limit {limit}%). "
        "Stopping before the OS OOM-killer does. Re-run `kb build` to continue — concept extraction is cached, "
        "so finished conversations are not redone.",
        file=sys.stderr,
    )
    sys.exit(1)
