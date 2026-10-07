"""Narration timing: where each script sentence sits on the narration audio timeline.

Three ways to get timings, best first:
1. script + audio + faster-whisper : word timestamps, script text aligned onto them
2. script + audio (no whisper)     : sentence breaks snapped to pauses in the audio
3. script only                     : edge-tts speaks each sentence, so timings are exact
And audio only (no script) uses whisper's own sentences.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import media
from .script import ScriptLine, normalize


class MissingDependency(RuntimeError):
    pass


@dataclass
class Segment:
    start: float
    end: float
    text: str
    hint: tuple[float, float] | None = None
    seg_index: int = -1


@dataclass
class Word:
    start: float
    end: float
    text: str


# --------------------------------------------------------------------------- whisper

def transcribe(
    audio: Path,
    cache_dir: Path,
    language: str | None,
    model_size: str,
    initial_prompt: str | None,
    log: Callable[[str], None] = print,
) -> tuple[list[Word], list[Segment]]:
    key = hashlib.sha1(
        f"{audio.resolve()}|{audio.stat().st_size}|{audio.stat().st_mtime}|{language}|"
        f"{model_size}|{initial_prompt}".encode()
    ).hexdigest()[:16]
    cache = cache_dir / f"whisper_{key}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return [Word(**w) for w in data["words"]], [Segment(**s) for s in data["segments"]]
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise MissingDependency("faster-whisper install nahi hai (pip install faster-whisper)") from e

    log(f"Whisper ({model_size}) se narration sun raha hoon...")
    model = WhisperModel(model_size, device="auto", compute_type="default")
    segs, _info = model.transcribe(
        str(audio), language=language, word_timestamps=True, initial_prompt=initial_prompt
    )
    words: list[Word] = []
    segments: list[Segment] = []
    for s in segs:
        if s.text.strip():
            segments.append(Segment(float(s.start), float(s.end), s.text.strip()))
        for w in s.words or []:
            words.append(Word(float(w.start), float(w.end), w.word))
    cache.write_text(
        json.dumps({"words": [asdict(w) for w in words],
                    "segments": [asdict(s) for s in segments]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return words, segments


def align_script_to_words(
    lines: list[ScriptLine], words: list[Word], min_ratio: float = 0.35
) -> list[Segment] | None:
    """Maps every script sentence onto whisper word timings via character alignment.
    Returns None when the transcript does not resemble the script (e.g. different alphabet)."""
    parts = [normalize(ln.text) for ln in lines]
    bounds = np.cumsum([0] + [len(p) for p in parts])
    s_text = "".join(parts)
    w_text_parts: list[str] = []
    char_word: list[int] = []
    for wi, w in enumerate(words):
        n = normalize(w.text)
        w_text_parts.append(n)
        char_word.extend([wi] * len(n))
    w_text = "".join(w_text_parts)
    if not s_text or not w_text:
        return None

    sm = difflib.SequenceMatcher(None, s_text, w_text, autojunk=False)
    mapping = np.full(len(s_text), -1, dtype=np.int64)
    matched = 0
    for a, b, size in sm.get_matching_blocks():
        if size:
            mapping[a:a + size] = np.arange(b, b + size)
            matched += size
    if matched / len(s_text) < min_ratio:
        return None

    times: list[list[float | None]] = []
    for k in range(len(lines)):
        idx = mapping[bounds[k]:bounds[k + 1]]
        idx = idx[idx >= 0]
        if len(idx) == 0:
            times.append([None, None])
        else:
            times.append([words[char_word[idx[0]]].start, words[char_word[idx[-1]]].end])
    end_t = words[-1].end
    _interpolate_missing(times, [len(p) for p in parts], 0.0, end_t)
    return [
        Segment(float(t[0]), float(t[1]), ln.text, ln.hint)
        for t, ln in zip(times, lines)
    ]


def _interpolate_missing(
    times: list[list[float | None]], weights: list[int], t0: float, t1: float
) -> None:
    n = len(times)
    i = 0
    while i < n:
        if times[i][0] is not None:
            i += 1
            continue
        j = i
        while j < n and times[j][0] is None:
            j += 1
        left = times[i - 1][1] if i > 0 else t0
        right = times[j][0] if j < n else t1
        left = float(left)  # type: ignore[arg-type]
        right = max(float(right), left)  # type: ignore[arg-type]
        w = np.array([max(weights[k], 1) for k in range(i, j)], dtype=float)
        edges = left + (right - left) * np.concatenate([[0], np.cumsum(w) / w.sum()])
        for k in range(i, j):
            times[k] = [float(edges[k - i]), float(edges[k - i + 1])]
        i = j


# --------------------------------------------------------------------------- silence based

def align_script_by_silence(
    lines: list[ScriptLine], audio: Path, total: float
) -> list[Segment]:
    """No speech recognition: put sentence breaks on the pauses that best fit the
    sentence lengths (TTS voices pause clearly between sentences)."""
    n = len(lines)
    sil = media.silence_intervals(audio)
    speech_start = sil[0][1] if sil and sil[0][0] <= 0.05 else 0.0
    speech_end = sil[-1][0] if sil and sil[-1][1] >= total - 0.05 else total
    if speech_end <= speech_start:
        speech_start, speech_end = 0.0, total
    inner = [(a, b) for a, b in sil if a > speech_start + 0.05 and b < speech_end - 0.05]

    chars = np.array([max(len(normalize(ln.text)), 1) for ln in lines], dtype=float)
    expected = speech_start + (speech_end - speech_start) * np.cumsum(chars)[:-1] / chars.sum()

    cuts: list[tuple[float, float]]
    if n == 1:
        cuts = []
    elif len(inner) >= n - 1:
        cuts = _pick_pauses(inner, expected, (speech_end - speech_start) / n)
    else:
        cuts = [(float(e), float(e)) for e in expected]

    segs: list[Segment] = []
    prev_end = speech_start
    for k, ln in enumerate(lines):
        end = cuts[k][0] if k < len(cuts) else speech_end
        segs.append(Segment(prev_end, max(end, prev_end), ln.text, ln.hint))
        if k < len(cuts):
            prev_end = cuts[k][1]
    return segs


def _pick_pauses(
    pauses: list[tuple[float, float]], expected: np.ndarray, avg_len: float
) -> list[tuple[float, float]]:
    """Monotonic assignment of len(expected) cuts to pauses (DP), preferring pauses
    near the expected position and longer pauses."""
    mids = np.array([(a + b) / 2 for a, b in pauses])
    lens = np.array([b - a for a, b in pauses])
    k_n, c_n = len(expected), len(pauses)
    cost = np.abs(mids[None, :] - expected[:, None]) / max(avg_len, 1e-3) - 0.8 * np.minimum(lens / 0.5, 1.0)[None, :]
    inf = 1e18
    dp = np.full((k_n, c_n), inf)
    arg = np.zeros((k_n, c_n), dtype=np.int64)
    dp[0] = cost[0]
    for k in range(1, k_n):
        best = inf
        best_i = -1
        for c in range(c_n):
            # best over c' < c
            if c - 1 >= 0 and dp[k - 1, c - 1] < best:
                best, best_i = dp[k - 1, c - 1], c - 1
            if best_i >= 0:
                dp[k, c] = best + cost[k, c]
                arg[k, c] = best_i
    c = int(np.argmin(dp[-1]))
    chosen = [c]
    for k in range(k_n - 1, 0, -1):
        c = int(arg[k, c])
        chosen.append(c)
    chosen.reverse()
    return [pauses[c] for c in chosen]


# --------------------------------------------------------------------------- TTS

def synthesize(
    lines: list[ScriptLine],
    voice: str,
    rate: str,
    gap: float,
    workdir: Path,
    log: Callable[[str], None] = print,
) -> tuple[Path, list[Segment]]:
    """Speaks every sentence with edge-tts and joins them, returning exact timings."""
    try:
        import edge_tts
    except ImportError as e:
        raise MissingDependency("edge-tts install nahi hai (pip install edge-tts)") from e

    tts_dir = workdir / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, ln in enumerate(lines):
        key = hashlib.sha1(f"{voice}|{rate}|{ln.text}".encode()).hexdigest()[:12]
        jobs.append((ln, tts_dir / f"{i:03d}_{key}.mp3", tts_dir / f"{i:03d}_{key}.wav"))

    async def _speak() -> None:
        for ln, mp3, _ in jobs:
            if not mp3.exists():
                await edge_tts.Communicate(ln.text, voice, rate=rate).save(str(mp3))

    log(f"AI voice ({voice}) se {len(lines)} lines bulwa raha hoon...")
    asyncio.run(_speak())

    segs: list[Segment] = []
    t = 0.0
    list_file = tts_dir / "concat.txt"
    entries = []
    for ln, mp3, wav in jobs:
        media.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                   "-af", f"apad=pad_dur={gap}", "-ar", "24000", "-ac", "1", str(wav)])
        dur = media.probe(wav).duration
        segs.append(Segment(t, t + max(dur - gap, 0.05), ln.text, ln.hint))
        t += dur
        entries.append(media.concat_entry(wav))
    list_file.write_text("\n".join(entries), encoding="utf-8")
    out = workdir / "narration_tts.wav"
    media.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", str(list_file), "-c", "copy", str(out)])
    return out, segs


# --------------------------------------------------------------------------- timeline

def fill_timeline(segs: list[Segment], total: float) -> list[Segment]:
    """Makes segments contiguous over [0, total]: each pause is split down the middle."""
    segs = sorted(segs, key=lambda s: s.start)
    out: list[Segment] = []
    for k, s in enumerate(segs):
        start = 0.0 if k == 0 else out[-1].end
        if k + 1 < len(segs):
            nxt = segs[k + 1].start
            end = (s.end + nxt) / 2 if nxt >= s.end else (s.start + nxt) / 2
        else:
            end = total
        end = min(max(end, start + 0.05), total)
        out.append(Segment(start, end, s.text, s.hint, k))
    if out:
        out[-1].end = total
    return [s for s in out if s.end - s.start > 0.01]


def split_pieces(segs: list[Segment], max_clip: float) -> list[Segment]:
    """Long sentences become several shorter cuts so the video keeps moving."""
    pieces: list[Segment] = []
    for s in segs:
        dur = s.end - s.start
        k = max(1, int(np.ceil(dur / max_clip - 1e-6))) if max_clip > 0 else 1
        edges = np.linspace(s.start, s.end, k + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            pieces.append(Segment(float(a), float(b), s.text, s.hint, s.seg_index))
    return pieces
