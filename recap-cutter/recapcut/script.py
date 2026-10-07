"""Narration script parsing.

Script format (plain .txt, UTF-8):

    # '#' se shuru hone wali line comment hai
    Raj ek chhote se gaon me rehta hai.
    [00:12:30] Ek din usse ek ajeeb chitthi milti hai.
    [01:05:00-01:06:10] Climax me dono aamne saamne aate hain!

Square-bracket timestamps are optional hints: "is line ki clip movie me yahan ke aas paas hai".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .media import parse_ts

_TS = r"\d{1,2}(?::\d{1,2}){1,2}(?:\.\d+)?"
_HINT = re.compile(rf"^\s*\[\s*({_TS})\s*(?:-\s*({_TS}))?\s*\]\s*")
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")


@dataclass
class ScriptLine:
    text: str
    hint: tuple[float, float] | None = None


def parse_script(text: str) -> list[ScriptLine]:
    lines: list[ScriptLine] = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        for sent in _SENT_SPLIT.split(raw):
            sent = sent.strip()
            hint = None
            m = _HINT.match(sent)
            if m:
                a = parse_ts(m.group(1))
                b = parse_ts(m.group(2)) if m.group(2) else a
                hint = (min(a, b), max(a, b))
                sent = sent[m.end():].strip()
            if sent:
                lines.append(ScriptLine(sent, hint))
            elif hint:
                # A bare "[12:30]" line applies to the next sentence.
                lines.append(ScriptLine("", hint))
    # Fold empty hint-only entries into the following sentence.
    merged: list[ScriptLine] = []
    pending: tuple[float, float] | None = None
    for ln in lines:
        if not ln.text:
            pending = ln.hint
            continue
        if pending and ln.hint is None:
            ln.hint = pending
        pending = None
        merged.append(ln)
    return merged


def normalize(text: str) -> str:
    """Lowercase and drop punctuation/whitespace/symbols, keeping letters, marks and digits
    (so Devanagari matras survive)."""
    out = []
    for ch in unicodedata.normalize("NFC", text.lower()):
        cat = unicodedata.category(ch)
        if cat[0] in "LMN":
            out.append(ch)
    return "".join(out)
