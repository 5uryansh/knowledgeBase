"""Gemini-based pairwise judge for two retrieved passage sets.

Returns a graded verdict for SET A vs SET B. The caller runs each question twice
with the systems swapped and averages, to cancel position bias.
"""
from __future__ import annotations
import json
import time

from . import config

# Verdict -> score from SET A's perspective.
VERDICT_SCORES = {
    "A_much_better": 2,
    "A_slightly_better": 1,
    "tie": 0,
    "B_slightly_better": -1,
    "B_much_better": -2,
}

PROMPT_TEMPLATE = """You are evaluating two retrieval systems for a personal knowledge base.
Given a QUESTION and two sets of retrieved passages (SET A and SET B), decide
which set would better help someone answer the question.

Judge on:
1. Relevance - how many passages actually pertain to the question.
2. Coverage - whether the set captures the different relevant threads/aspects,
   including information connected across different conversations or sources.
3. Answerability - which set, used alone, supports a more complete, grounded answer.
4. Signal-to-noise - penalize irrelevant passages.

Important:
- Do NOT prefer a set because it is longer or has more passages. More is not better.
- Do NOT let the order of passages influence you.
- Judge only on the content's usefulness for the question.

QUESTION:
{question}

SET A:
{set_a}

SET B:
{set_b}

Respond with ONLY a JSON object:
{{"verdict": "A_much_better" | "A_slightly_better" | "tie" | "B_slightly_better" | "B_much_better",
 "reason": "<one sentence>"}}"""


def _format_passages(chunks) -> str:
    if not chunks:
        return "(no passages)"
    lines = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.text[:config.JUDGE_PASSAGE_CHARS].replace("\n", " ")
        lines.append(f"[{i}] (source: {chunk.source.name}) {text}")
    return "\n".join(lines)


class Judge:
    def __init__(self) -> None:
        self._client = None
        if config.GEMINI_API_KEY:
            from google import genai
            self._client = genai.Client(api_key=config.GEMINI_API_KEY)

    @property
    def available(self) -> bool:
        return self._client is not None

    def _call(self, prompt: str) -> str:
        last_error = None
        for attempt in range(config.JUDGE_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                    config={"response_mime_type": "application/json", "temperature": 0},
                )
                return response.text
            except Exception as error:  # includes rate-limit / transient API errors
                last_error = error
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Gemini call failed after {config.JUDGE_MAX_RETRIES} retries: {last_error}")

    def judge(self, question: str, set_a_chunks, set_b_chunks) -> dict:
        """Return {'verdict', 'score', 'reason'} for SET A vs SET B."""
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            set_a=_format_passages(set_a_chunks),
            set_b=_format_passages(set_b_chunks),
        )
        raw = self._call(prompt)
        try:
            parsed = json.loads(raw)
            verdict = parsed.get("verdict", "tie")
            reason = parsed.get("reason", "")
        except (json.JSONDecodeError, AttributeError):
            verdict, reason = "tie", f"unparseable judge response: {raw[:120]!r}"
        score = VERDICT_SCORES.get(verdict, 0)
        return {"verdict": verdict, "score": score, "reason": reason}
