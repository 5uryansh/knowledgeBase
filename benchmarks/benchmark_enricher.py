"""Benchmark the enricher's extraction stages (topic + entity extraction).

Read-only: it never modifies files, so it's safe to run repeatedly against the
real vault. Measures per-document timing, peak RSS (via a background sampler so
intra-call spikes are captured), and result counts — so optimizations (caching,
ONNX/quantization, model swaps) can be compared against a stable baseline.

Usage:
    python benchmark_enricher.py [DIR] [--limit N]

DIR defaults to the Claude Code conversations folder.
"""
import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.topic_extractor import TopicExtractor
from src.utils.entity_extractor import EntityExtractor

DEFAULT_DIR = "/mnt/c/Users/Suryansh/Documents/KnowledgeBase/conversations/claude-code"


def _rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


class PeakRSSSampler:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak_mb = 0.0
        self._running = False
        self._thread = None

    def _sample(self):
        while self._running:
            self.peak_mb = max(self.peak_mb, _rss_mb())
            time.sleep(self.interval)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        self.peak_mb = max(self.peak_mb, _rss_mb())
        return self.peak_mb


def main():
    parser = argparse.ArgumentParser(description="Benchmark enricher extraction stages")
    parser.add_argument("directory", nargs="?", default=DEFAULT_DIR,
                        help="Directory of markdown files to benchmark")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N files (for quick iteration)")
    args = parser.parse_args()

    files = sorted(Path(args.directory).rglob("*.md"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"No markdown files found in {args.directory}")
        return

    print(f"Benchmarking {len(files)} files from {args.directory}\n")

    sampler = PeakRSSSampler()
    sampler.start()

    load_start = time.time()
    topic_extractor = TopicExtractor()
    entity_extractor = EntityExtractor()
    load_time = time.time() - load_start
    print(f"Model load time: {load_time:.1f}s | RSS after load: {_rss_mb():.0f} MB\n")

    header = f"{'#':>3} {'file':<42} {'chars':>8} {'topic_s':>8} {'entity_s':>9} {'total_s':>8} {'tops':>5} {'ents':>5} {'rss_mb':>7}"
    print(header)
    print("-" * len(header))

    rows = []
    for i, f in enumerate(files, 1):
        text = f.read_text(encoding="utf-8")

        t0 = time.time()
        topics = topic_extractor.extract_topics([text])
        t_topic = time.time() - t0

        t1 = time.time()
        entities = entity_extractor.extract_entities(text)
        t_entity = time.time() - t1

        total = t_topic + t_entity
        rss = _rss_mb()
        rows.append({
            "chars": len(text), "topic_s": t_topic, "entity_s": t_entity,
            "total_s": total, "topics": len(topics), "entities": len(entities),
        })
        print(f"{i:>3} {f.name[:42]:<42} {len(text):>8} {t_topic:>8.2f} {t_entity:>9.2f} {total:>8.2f} {len(topics):>5} {len(entities):>5} {rss:>7.0f}", flush=True)

    peak_rss = sampler.stop()

    total_chars = sum(r["chars"] for r in rows)
    total_topic = sum(r["topic_s"] for r in rows)
    total_entity = sum(r["entity_s"] for r in rows)
    total_time = total_topic + total_entity
    per_file = [r["total_s"] for r in rows]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed:        {len(rows)}")
    print(f"Total chars:            {total_chars:,}")
    print(f"Total extraction time:  {total_time:.1f}s  (topics {total_topic:.1f}s + entities {total_entity:.1f}s)")
    if total_time > 0:
        print(f"  entities are {total_entity / total_time * 100:.0f}% of extraction time")
        print(f"Throughput:             {total_chars / total_time:,.0f} chars/s")
    print(f"Per-file mean:          {statistics.mean(per_file):.2f}s")
    print(f"Per-file median:        {statistics.median(per_file):.2f}s")
    print(f"Per-file max:           {max(per_file):.2f}s")
    print(f"Peak RSS:               {peak_rss:.0f} MB")
    print(f"Model load (one-time):  {load_time:.1f}s")


if __name__ == "__main__":
    main()
