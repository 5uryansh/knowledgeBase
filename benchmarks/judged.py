"""LLM-judged benchmark on *your own* history — no relevance labels needed.

    export GEMINI_API_KEY=...                       # or use any OpenAI-compatible endpoint:
    export KB_JUDGE_BASE_URL=http://localhost:8000/v1 KB_JUDGE_MODEL=qwen2.5-7b KB_JUDGE_API_KEY=...
    python -m benchmarks.judged --vault ~/Documents/KnowledgeBase --questions benchmarks/questions.json

For each question and retrieval mode, the top-k passages are graded 0–3 by an LLM
(3 = fully answers, 0 = irrelevant). From the grades we compute graded nDCG@k,
precision@k (grade ≥ 2) and answerable@k (some passage graded 3). Optionally
(`--pairwise A,B`) the judge also compares two modes' passage sets head to head,
twice with the order swapped to cancel position bias.

Grades are cached per (question, chunk) in ``<results>/judge_cache.jsonl``, so
modes that retrieve the same passage never pay for it twice, and an interrupted
run resumes where it stopped. Ported from the Gemini judge in PR #9.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from knowledgebase.retrieval.index import KnowledgeIndex
from knowledgebase.retrieval.retriever import MODES

ROOT = Path(__file__).resolve().parent

GRADE_PROMPT = """Grade how well each passage answers the QUESTION, on a 0-3 scale:
3 = directly and fully answers the question
2 = clearly relevant, partially answers it
1 = marginally related
0 = irrelevant

Judge each passage on its own merit. Do not reward a passage for being long.

QUESTION:
{question}

PASSAGES:
{passages}

Respond with ONLY a JSON object mapping passage number to grade, e.g.:
{{"grades": {{"1": 3, "2": 0, "3": 2}}}} - one integer per passage, in order."""

PAIRWISE_PROMPT = """You are evaluating two retrieval systems for a personal knowledge base.
Given a QUESTION and two sets of retrieved passages (SET A and SET B), decide which set would better help
someone answer the question. Judge relevance, coverage (including information connected across different
conversations), answerability and signal-to-noise. Do NOT prefer a set because it is longer. Ignore order.

QUESTION:
{question}

SET A:
{set_a}

SET B:
{set_b}

Respond with ONLY a JSON object:
{{"verdict": "A_much_better" | "A_slightly_better" | "tie" | "B_slightly_better" | "B_much_better", "reason": "<one sentence>"}}"""

VERDICT_SCORES = {"A_much_better": 2, "A_slightly_better": 1, "tie": 0, "B_slightly_better": -1, "B_much_better": -2}
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3-flash", "gemini-2.5-flash"]


class LLM(Protocol):
    name: str

    def complete_json(self, prompt: str) -> str:  # pragma: no cover
        ...


class GeminiLLM:
    """Rotates through a model pool so one model's daily quota doesn't stall the run."""

    def __init__(self, api_key: str, models: list[str] | None = None, retries: int = 2):
        from google import genai  # pip install google-genai

        self.client = genai.Client(api_key=api_key)
        self.models = models or GEMINI_MODELS
        self.idx, self.retries = 0, retries
        self.name = f"gemini:{self.models[0]}"

    def complete_json(self, prompt: str) -> str:
        last = None
        for _ in range(len(self.models)):
            model = self.models[self.idx]
            for attempt in range(self.retries):
                try:
                    r = self.client.models.generate_content(
                        model=model, contents=prompt,
                        config={"response_mime_type": "application/json", "temperature": 0})
                    return r.text
                except Exception as exc:  # noqa: BLE001 - quota / transient
                    last = exc
                    time.sleep(min(2 ** attempt, 8))
            self.idx = (self.idx + 1) % len(self.models)
            print(f"  judge: rotating to {self.models[self.idx]} after: {str(last)[:80]}", flush=True)
        raise RuntimeError(f"all judge models exhausted: {last}")


class OpenAICompatibleLLM:
    """Any /v1/chat/completions endpoint: OpenAI, vLLM, Ollama, llama.cpp, a Hugging Face Space, ..."""

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key = model, api_key
        self.name = f"openai-compatible:{model}"

    def complete_json(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310 - user-configured URL
                    return json.loads(resp.read())["choices"][0]["message"]["content"]
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return ""


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1]
    start, end = raw.find("{"), raw.rfind("}")
    try:
        return json.loads(raw[start:end + 1]) if start >= 0 else {}
    except json.JSONDecodeError:
        return {}


def format_passages(chunks, chars: int = 700) -> str:
    if not chunks:
        return "(no passages)"
    return "\n".join(f"[{i}] ({c.title}, {c.date}) {c.plain[:chars].replace(chr(10), ' ')}" for i, c in enumerate(chunks, 1))


class GradeCache:
    def __init__(self, path: Path):
        self.path = path
        self.grades: dict[tuple[str, str], int] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.grades[(row["q"], row["chunk"])] = row["grade"]

    def get(self, q: str, chunk: str) -> int | None:
        return self.grades.get((q, chunk))

    def put(self, q: str, chunk: str, grade: int) -> None:
        self.grades[(q, chunk)] = grade
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"q": q, "chunk": chunk, "grade": grade}) + "\n")


def grade(llm: LLM, cache: GradeCache, qid: str, question: str, chunks, delay: float) -> list[int]:
    missing = [c for c in chunks if cache.get(qid, c.id) is None]
    for start in range(0, len(missing), 10):
        batch = missing[start:start + 10]
        parsed = parse_json(llm.complete_json(GRADE_PROMPT.format(question=question, passages=format_passages(batch))))
        grades = parsed.get("grades", {}) if isinstance(parsed, dict) else {}
        for i, c in enumerate(batch, 1):
            try:
                g = max(0, min(3, int(grades.get(str(i), 0))))
            except (TypeError, ValueError):
                g = 0
            cache.put(qid, c.id, g)
        time.sleep(delay)
    return [cache.get(qid, c.id) or 0 for c in chunks]


def graded_metrics(grades: list[int], all_known: list[int], k: int) -> dict[str, float]:
    top = grades[:k]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(top))
    ideal = sorted(all_known, reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return {
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        f"precision@{k}": sum(1 for g in top if g >= 2) / max(len(top), 1),
        f"answerable@{k}": 1.0 if any(g == 3 for g in top) else 0.0,
        "first_relevant_rr": next((1 / (i + 1) for i, g in enumerate(top) if g >= 2), 0.0),
    }


def build_llm() -> LLM:
    if os.environ.get("KB_JUDGE_BASE_URL"):
        return OpenAICompatibleLLM(os.environ["KB_JUDGE_BASE_URL"], os.environ.get("KB_JUDGE_MODEL", "default"),
                                   os.environ.get("KB_JUDGE_API_KEY", ""))
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiLLM(os.environ["GEMINI_API_KEY"])
    raise SystemExit("Set GEMINI_API_KEY, or KB_JUDGE_BASE_URL (+ KB_JUDGE_MODEL, KB_JUDGE_API_KEY) for an "
                     "OpenAI-compatible judge.")


def run(index: KnowledgeIndex, questions: list[dict], llm: LLM, modes: list[str], k: int, out_dir: Path,
        pairwise: tuple[str, str] | None = None, delay: float = 0.0) -> dict:
    cache = GradeCache(out_dir / "judge_cache.jsonl")
    per_mode: dict[str, list[dict]] = defaultdict(list)
    pair_scores: list[float] = []
    for n, q in enumerate(questions, 1):
        qid, text = str(q["id"]), q["question"]
        retrieved = {m: [r.chunk for r in index.search(text, k=k, mode=m)] for m in modes}
        graded = {m: grade(llm, cache, qid, text, chunks, delay) for m, chunks in retrieved.items()}
        pool = {c.id: cache.get(qid, c.id) or 0 for chunks in retrieved.values() for c in chunks}
        for m in modes:
            per_mode[m].append({**graded_metrics(graded[m], list(pool.values()), k), "category": q.get("category", "")})
        if pairwise:
            a, b = pairwise
            s1 = VERDICT_SCORES.get(parse_json(llm.complete_json(PAIRWISE_PROMPT.format(
                question=text, set_a=format_passages(retrieved[a]), set_b=format_passages(retrieved[b])))).get("verdict"), 0)
            s2 = VERDICT_SCORES.get(parse_json(llm.complete_json(PAIRWISE_PROMPT.format(
                question=text, set_a=format_passages(retrieved[b]), set_b=format_passages(retrieved[a])))).get("verdict"), 0)
            pair_scores.append((s1 - s2) / 2)  # positive = first mode better
        print(f"  [{n}/{len(questions)}] {text[:70]}", flush=True)

    summary = {m: {key: statistics.fmean(r[key] for r in rows) for key in rows[0] if key != "category"}
               for m, rows in per_mode.items() if rows}
    result = {"judge": llm.name, "k": k, "questions": len(questions), "summary": summary}
    if pairwise and pair_scores:
        result["pairwise"] = {
            "modes": list(pairwise),
            "mean_score": statistics.fmean(pair_scores),
            "wins": sum(1 for s in pair_scores if s > 0), "ties": sum(1 for s in pair_scores if s == 0),
            "losses": sum(1 for s in pair_scores if s < 0),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vault", default="vault")
    ap.add_argument("--questions", default=str(ROOT / "questions.json"))
    ap.add_argument("--modes", default="graph_hybrid,hybrid,bm25,dense")
    ap.add_argument("--pairwise", default="graph_hybrid,hybrid", help="two modes to compare head to head ('' = off)")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between judge calls (free-tier RPM limits)")
    args = ap.parse_args()

    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    if args.limit:
        questions = questions[: args.limit]
    index = KnowledgeIndex.load(Path(args.vault).expanduser() / "indexes")
    modes = [m for m in args.modes.split(",") if m in MODES and (m != "dense" or index.embedder)]
    pairwise = tuple(args.pairwise.split(",")) if args.pairwise else None
    out_dir = ROOT / "results" / "judged"
    result = run(index, questions, build_llm(), modes, args.k, out_dir, pairwise, args.delay)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"judged_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
