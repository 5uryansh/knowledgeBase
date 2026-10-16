import json

from benchmarks.judged import graded_metrics, parse_json, run


class FakeJudge:
    """Grades a passage 3 if it mentions the question's key word, else 0; prefers set A on ties."""

    name = "fake"

    def complete_json(self, prompt: str) -> str:
        if "SET A" in prompt:
            return json.dumps({"verdict": "tie"})
        question = prompt.split("QUESTION:\n", 1)[1].split("\n", 1)[0].lower()
        passages = prompt.split("PASSAGES:\n", 1)[1].split("\n\nRespond", 1)[0].splitlines()
        key = "vllm" if "vllm" in question else "esp32"
        return json.dumps({"grades": {str(i): (3 if key in p.lower() else 0) for i, p in enumerate(passages, 1)}})


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"grades": {"1": 2}}\n```') == {"grades": {"1": 2}}
    assert parse_json("no json here") == {}


def test_graded_metrics():
    m = graded_metrics([0, 3, 2], [3, 3, 2, 0], k=3)
    assert m["answerable@3"] == 1.0
    assert abs(m["precision@3"] - 2 / 3) < 1e-9
    assert 0 < m["ndcg@3"] < 1
    assert m["first_relevant_rr"] == 0.5


def test_judged_run_with_fake_judge(index, tmp_path):
    questions = [{"id": 1, "question": "vLLM out of memory at startup", "category": "A"},
                 {"id": 2, "question": "ESP32 battery drains fast", "category": "H"}]
    result = run(index, questions, FakeJudge(), ["graph_hybrid", "bm25"], k=5, out_dir=tmp_path,
                 pairwise=("graph_hybrid", "bm25"))
    assert result["summary"]["graph_hybrid"]["answerable@5"] == 1.0
    assert result["pairwise"]["ties"] == 2
    # Grades were cached: a second run makes no new grading calls for the same passages.
    lines = (tmp_path / "judge_cache.jsonl").read_text().splitlines()
    run(index, questions, FakeJudge(), ["graph_hybrid", "bm25"], k=5, out_dir=tmp_path)
    assert len((tmp_path / "judge_cache.jsonl").read_text().splitlines()) == len(lines)
