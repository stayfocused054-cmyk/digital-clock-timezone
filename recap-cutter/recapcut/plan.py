"""The edit plan: which movie moment plays under which part of the narration.
Saved as plan.json (full) and plan.csv (easy to fix by hand in Excel / Sheets)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .media import fmt_ts, parse_ts


@dataclass
class Clip:
    narr_start: float
    narr_end: float
    movie_start: float
    text: str = ""
    score: float = 0.0
    why: str = ""


@dataclass
class Plan:
    movie: str
    narration_audio: str
    narration_duration: float
    srt: str = ""
    clips: list[Clip] = field(default_factory=list)


CSV_FIELDS = ["n", "narr_start", "narr_end", "movie_time", "score", "why", "text"]


def save(plan: Plan, json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(asdict(plan), ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for k, c in enumerate(plan.clips, 1):
            w.writerow({
                "n": k,
                "narr_start": f"{c.narr_start:.3f}",
                "narr_end": f"{c.narr_end:.3f}",
                "movie_time": fmt_ts(c.movie_start),
                "score": f"{c.score:.2f}",
                "why": c.why,
                "text": c.text,
            })


def load(path: Path) -> Plan:
    """Loads plan.json, or plan.csv next to it (CSV wins for clip times, so hand edits count)."""
    path = Path(path)
    json_path = path if path.suffix.lower() == ".json" else path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} nahi mila (CSV ke saath wali plan.json chahiye).")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    plan = Plan(
        movie=data["movie"],
        narration_audio=data["narration_audio"],
        narration_duration=float(data["narration_duration"]),
        srt=data.get("srt", ""),
        clips=[Clip(**c) for c in data["clips"]],
    )
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        clips = []
        for r in rows:
            clips.append(Clip(
                narr_start=float(r["narr_start"]),
                narr_end=float(r["narr_end"]),
                movie_start=parse_ts(r["movie_time"]),
                text=r.get("text", ""),
                score=float(r.get("score") or 0),
                why=r.get("why", ""),
            ))
        clips.sort(key=lambda c: c.narr_start)
        plan.clips = clips
    return plan
