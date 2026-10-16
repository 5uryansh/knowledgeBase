from datetime import date

import pytest

from knowledgebase.retrieval.timefilter import parse_time_window

TODAY = date(2026, 10, 17)


@pytest.mark.parametrize("query,inside,outside", [
    ("When did Melanie go camping in June?", ["2023-06-12", "2026-06-01"], ["2023-07-01"]),
    ("what did I decide about vLLM last spring", ["2025-04-02"], ["2026-04-02", "2025-07-01"]),
    ("vllm in March 2026", ["2026-03-05"], ["2025-03-05"]),
    ("notes from last month", ["2026-09-10"], ["2026-10-01"]),
    ("the ESP32 thing in 2025", ["2025-12-31"], ["2026-01-01"]),
    ("winter 2025 plans", ["2025-12-20", "2026-01-15"], ["2025-03-01"]),
    ("what was that two months ago", ["2026-08-17"], ["2026-03-01"]),
])
def test_windows(query, inside, outside):
    w = parse_time_window(query, TODAY)
    assert w is not None
    assert all(w.matches(d) for d in inside)
    assert not any(w.matches(d) for d in outside)


@pytest.mark.parametrize("query", ["you may want this", "how does PPO work", "fix the march of progress", ""])
def test_no_false_positives(query):
    assert parse_time_window(query, TODAY) is None


def test_search_reports_window(index):
    from knowledgebase.server.service import KnowledgeService

    svc = KnowledgeService.__new__(KnowledgeService)
    svc._index, svc._summaries = index, {}
    data = svc.search("what did I learn about quantization in January 2026", k=3)
    assert data["time_window"] == "January 2026"
    assert data["results"][0]["date"].startswith("2026-01")
