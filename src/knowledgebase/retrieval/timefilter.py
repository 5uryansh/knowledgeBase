"""Time hints in queries: "in March", "last spring", "2025", "two months ago", "recently".

Personal recall is full of them ("what did I decide about vLLM last spring?"). A
detected window softly boosts chunks from conversations inside it — it never
filters, because people misremember dates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"], start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
SEASONS = {"spring": (3, 5), "summer": (6, 8), "autumn": (9, 11), "fall": (9, 11), "winter": (12, 2)}
NUMBER_WORDS = {"one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "few": 3, "couple": 2}
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))


@dataclass
class TimeWindow:
    label: str
    matches: Callable[[str], bool]


def _month_range(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    end = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return start, end


def _range(start: date, end: date, label: str) -> TimeWindow:
    s, e = start.isoformat(), end.isoformat()
    return TimeWindow(label, lambda d: bool(d) and s <= d[:10] <= e)


def parse_time_window(query: str, today: date | None = None) -> TimeWindow | None:
    q = query.lower()
    today = today or date.today()

    # "in March 2026", "march 2025"
    m = re.search(rf"\b({_MONTH_RE})\.?\s+(20\d\d|19\d\d)\b", q)
    if m:
        start, end = _month_range(int(m.group(2)), MONTHS[m.group(1)])
        return _range(start, end, f"{m.group(1).title()} {m.group(2)}")
    # "last/this spring", "summer 2025"
    m = re.search(r"\b(?:(last|this|past)\s+)?(spring|summer|autumn|fall|winter)(?:\s+(?:of\s+)?(20\d\d))?\b", q)
    if m:
        which, season, year = m.group(1), m.group(2), m.group(3)
        first, last = SEASONS[season]
        if year:
            y = int(year)
        else:
            y = today.year
            season_started = (today.month >= first) if first <= last else (today.month >= first or today.month <= last)
            if which in ("last", "past") or (which is None and not season_started):
                y -= 1
        start = date(y, first, 1)
        end_year = y + (1 if first > last else 0)
        end = _month_range(end_year, last)[1]
        return _range(start, end, f"{season} {y}")
    # "N weeks/months/years ago"
    m = re.search(r"\b(\d+|one|two|three|four|five|six|a|an|few|couple)\s+(?:of\s+)?(day|week|month|year)s?\s+ago\b", q)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1)]
        days = {"day": 1, "week": 7, "month": 30, "year": 365}[m.group(2)] * n
        centre = today - timedelta(days=days)
        slack = max(3, days // 3)
        return _range(centre - timedelta(days=slack), centre + timedelta(days=slack), m.group(0))
    # "last week/month/year", "this month/year", "yesterday", "recently"
    m = re.search(r"\b(last|this|past)\s+(week|month|year)\b", q)
    if m:
        unit = m.group(2)
        if unit == "year":
            y = today.year - (m.group(1) != "this")
            return _range(date(y, 1, 1), date(y, 12, 31), f"{m.group(1)} year")
        if unit == "month":
            if m.group(1) == "this":
                start, end = _month_range(today.year, today.month)
            else:
                prev = today.replace(day=1) - timedelta(days=1)
                start, end = _month_range(prev.year, prev.month)
            return _range(start, end, f"{m.group(1)} month")
        start = today - timedelta(days=7 if m.group(1) == "this" else 14)
        return _range(start, today, f"{m.group(1)} week")
    if re.search(r"\b(recently|lately|latest)\b", q):
        return _range(today - timedelta(days=60), today, "recently")
    # "in March" (any year), "in July"
    # (a preposition is required so "may" the verb and "mar" the word are not read as months)
    m = re.search(rf"\b(?:in|during|since|around|of)\s+({_MONTH_RE})\b(?!\s+\d)", q)
    if m:
        month = MONTHS[m.group(1)]
        return TimeWindow(m.group(1).title(), lambda d, month=month: bool(d) and len(d) >= 7 and d[5:7] == f"{month:02d}")
    # "in 2025"
    m = re.search(r"\b(?:in|during|since|from|of)\s+(20\d\d)\b", q)
    if m:
        y = int(m.group(1))
        return _range(date(y, 1, 1), date(y, 12, 31), m.group(1))
    return None
