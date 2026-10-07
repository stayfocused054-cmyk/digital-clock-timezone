"""Minimal SRT / WebVTT reader and writer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cue:
    start: float
    end: float
    text: str


_TIME = r"(\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{1,3}"
_LINE = re.compile(rf"({_TIME})\s*-->\s*({_TIME})")
_TAGS = re.compile(r"<[^>]+>|\{[^}]*\}")


def _to_seconds(ts: str) -> float:
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 2:
        parts.insert(0, "0")
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue
            return text
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def load_subtitles(path: str | Path) -> list[Cue]:
    text = _read_text(Path(path)).replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        for i, ln in enumerate(lines):
            m = _LINE.search(ln)
            if not m:
                continue
            body = " ".join(lines[i + 1:])
            body = _TAGS.sub("", body).replace("\\N", " ").strip()
            body = re.sub(r"\s+", " ", body)
            if body:
                cues.append(Cue(_to_seconds(m.group(1)), _to_seconds(m.group(3)), body))
            break
    cues.sort(key=lambda c: c.start)
    return cues


def _srt_ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[Cue], path: str | Path) -> None:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(f"{i}\n{_srt_ts(c.start)} --> {_srt_ts(c.end)}\n{c.text}\n")
    Path(path).write_text("\n".join(out), encoding="utf-8")


def text_between(cues: list[Cue], start: float, end: float) -> str:
    return " ".join(c.text for c in cues if c.end >= start and c.start <= end)
