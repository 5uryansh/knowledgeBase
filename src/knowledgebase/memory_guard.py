import sys


def _read_value_kb(path, key):
    with open(path, "r") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1])
    raise ValueError(f"{key} not found in {path}")


def current_memory_percent():
    rss_kb = _read_value_kb("/proc/self/status", "VmRSS:")
    total_kb = _read_value_kb("/proc/meminfo", "MemTotal:")
    return (rss_kb / total_kb) * 100


def check_memory_or_exit(max_percent=60, context=""):
    percent_used = current_memory_percent()

    if percent_used >= max_percent:
        label = f" ({context})" if context else ""
        print(
            f"Memory guard{label}: process is using {percent_used:.1f}% of total RAM "
            f"(limit {max_percent}%). Stopping now before the OS OOM-killer does. "
            f"Rerun main.py to continue; the crashed step is where the next optimization is needed.",
            file=sys.stderr,
        )
        sys.exit(1)
