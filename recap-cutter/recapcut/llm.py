"""Optional: ask Claude where each narration line happens in the movie, using the
movie's subtitles (and what Claude knows about the film). The answer is used as a
strong hint; CLIP and the order constraint still pick the exact shot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .media import fmt_ts, parse_ts
from .subtitles import Cue


@dataclass
class LlmHint:
    start: float
    end: float
    confidence: float


SYSTEM = """You help a video editor build a movie recap. You get the movie's subtitles with \
timestamps and the editor's narration, split into numbered lines. The narration retells the \
movie, mostly in story order, often in Hindi or Hinglish while the subtitles may be in another \
language.

For every narration line, find the stretch of the movie whose footage best shows what that \
line describes. Use the dialogue, its timing, and what you know about this film. Scenes \
without dialogue sit in the gaps between subtitles, so infer them from context. If a line is \
general commentary rather than a specific event, pick footage that fits its mood near the \
surrounding lines. Keep each range short (roughly 3-20 seconds) and inside the movie's length. \
Set confidence from 0 to 1: high only when the dialogue or plot clearly pins the moment."""


SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "integer"},
                    "start": {"type": "string", "description": "H:MM:SS"},
                    "end": {"type": "string", "description": "H:MM:SS"},
                    "confidence": {"type": "number"},
                },
                "required": ["line", "start", "end", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["matches"],
    "additionalProperties": False,
}


def claude_hints(
    lines: list[str],
    cues: list[Cue],
    movie_duration: float,
    movie_title: str | None,
    model: str,
    log: Callable[[str], None] = print,
) -> list[LlmHint | None]:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("anthropic package install nahi hai (pip install anthropic)") from e

    subs = "\n".join(f"[{fmt_ts(c.start)[:8]}] {c.text}" for c in cues)
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(lines))
    title = movie_title or "unknown"
    prompt = (
        f"Movie: {title}\nMovie length: {fmt_ts(movie_duration)[:8]}\n\n"
        f"<subtitles>\n{subs}\n</subtitles>\n\n"
        f"<narration>\n{numbered}\n</narration>\n\n"
        "Return one match for every narration line number."
    )

    log(f"Claude ({model}) se har line ka movie timestamp puchh raha hoon...")
    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=model,
        max_tokens=32000,
        system=SYSTEM,
        betas=["server-side-fallback-2026-07-01"],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()

    if msg.stop_reason == "refusal":
        log("  Claude ne yeh request mana kar di, AI hints ke bina aage badh raha hoon.")
        return [None] * len(lines)
    if msg.stop_reason == "max_tokens":
        log("  Claude ka jawab adhoora reh gaya, AI hints ke bina aage badh raha hoon.")
        return [None] * len(lines)
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log("  Claude ka jawab samajh nahi aaya, AI hints ke bina aage badh raha hoon.")
        return [None] * len(lines)

    hints: list[LlmHint | None] = [None] * len(lines)
    for m in data.get("matches", []):
        try:
            i = int(m["line"])
            a, b = parse_ts(m["start"]), parse_ts(m["end"])
        except (KeyError, ValueError, TypeError):
            continue
        if 0 <= i < len(lines):
            a, b = (min(max(x, 0.0), movie_duration) for x in sorted((a, b)))
            conf = float(min(max(m.get("confidence", 0.5), 0.0), 1.0))
            hints[i] = LlmHint(a, b, conf)
    found = sum(h is not None for h in hints)
    log(f"  Claude ne {found}/{len(lines)} lines ke timestamps diye.")
    return hints
