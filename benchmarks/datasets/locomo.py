"""LoCoMo → knowledgeBase.

LoCoMo (Maharana et al., ACL 2024, "Evaluating Very Long-Term Conversational
Memory of LLM Agents", https://github.com/snap-research/locomo) contains 10 very
long chat histories (19–32 sessions each, ~600 turns) with ~1,500 answerable
questions whose *evidence turns* are annotated. That is exactly the
knowledgeBase use case: "find where in my chat history X was discussed",
including questions whose evidence is spread across several conversations.

Each LoCoMo session becomes one conversation in a **Claude.ai export file**
(``conversations.json``), so the benchmark runs the real parser → enrichment →
graph → index pipeline end to end.

Categories (as used by the LoCoMo authors and follow-up work such as Mem0):
1 multi-hop · 2 temporal · 3 open-domain · 4 single-hop · 5 adversarial (no answer; skipped)
"""
from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path

from benchmarks.datasets import Corpus, Query

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
_DIA = re.compile(r"D(\d+):(\d+)")


def download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading LoCoMo from {URL} ...")
        urllib.request.urlretrieve(URL, dest)  # noqa: S310 - fixed https URL
    return dest


def _parse_session_time(value: str) -> str:
    # e.g. "1:56 pm on 8 May, 2023"
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(value.strip(), fmt).isoformat() + "Z"
        except ValueError:
            continue
    return ""


def _turn_text(turn: dict) -> str:
    text = turn.get("text", "").strip()
    caption = turn.get("blip_caption")
    if caption:
        text = f"{text} [shares a photo: {caption}]".strip()
    return f"{turn['speaker']}: {text}"


def convert(sample: dict, out_dir: Path) -> Corpus:
    """Write one LoCoMo sample as a Claude export; return the corpus + queries."""
    conv = sample["conversation"]
    speaker_a = conv["speaker_a"]
    sessions = sorted(
        (int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k) and conv[k]),
    )
    export = []
    locators: dict[str, tuple[str, int]] = {}
    for s in sessions:
        turns = conv[f"session_{s}"]
        when = _parse_session_time(conv.get(f"session_{s}_date_time", ""))
        other = next((t["speaker"] for t in turns if t["speaker"] != speaker_a), conv["speaker_b"])
        native_id = f"{sample['sample_id']}-session-{s}"
        messages = []
        for idx, turn in enumerate(turns):
            messages.append({
                "uuid": f"{native_id}-{idx}",
                "sender": "human" if turn["speaker"] == speaker_a else "assistant",
                "created_at": when,
                "content": [{"type": "text", "text": _turn_text(turn)}],
            })
            locators[turn["dia_id"]] = (native_id, idx)
        date_label = conv.get(f"session_{s}_date_time", "").split(" on ")[-1]
        export.append({
            "uuid": native_id,
            "name": f"{speaker_a} & {other} — {date_label}",
            "created_at": when,
            "chat_messages": messages,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = out_dir / "conversations.json"
    export_path.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")

    corpus = Corpus(id=sample["sample_id"], export_path=str(export_path), platform="claude",
                    evidence_locators=locators)
    for n, qa in enumerate(sample["qa"]):
        category = CATEGORIES.get(qa.get("category"))
        if category is None:
            continue
        evidence_ids = []
        for ev in qa.get("evidence", []):
            evidence_ids.extend(f"D{a}:{b}" for a, b in _DIA.findall(ev.replace(" ", "")))
        evidence_ids = [e for e in dict.fromkeys(evidence_ids) if e in locators]
        if not evidence_ids:
            continue
        q = Query(id=f"{sample['sample_id']}-q{n}", text=qa["question"], category=category,
                  answer=str(qa.get("answer", "")))
        q.evidence = [{e} for e in evidence_ids]  # resolved to chunk ids later
        q.evidence_sessions = {locators[e][0] for e in evidence_ids}
        corpus.queries.append(q)
    return corpus


def load(data_file: Path, work_dir: Path, samples: list[int] | None = None) -> list[Corpus]:
    data = json.loads(download(data_file).read_text(encoding="utf-8"))
    chosen = range(len(data)) if samples is None else samples
    return [convert(data[i], work_dir / data[i]["sample_id"]) for i in chosen]
