"""ffmpeg / ffprobe helpers: probing, scene detection, frame streaming, silence detection."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np


class MediaError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise MediaError(
            f"{', '.join(missing)} nahi mila. ffmpeg install karo aur PATH me daalo "
            "(Windows: `winget install Gyan.FFmpeg`, Mac: `brew install ffmpeg`)."
        )


def run(cmd: list[str], cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise MediaError(f"Command fail hua: {' '.join(cmd[:6])} ...\n{tail}")
    return proc


@dataclass
class MediaInfo:
    duration: float
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = False
    has_audio: bool = False


def probe(path: str | Path) -> MediaInfo:
    proc = run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    )
    data = json.loads(proc.stdout or "{}")
    info = MediaInfo(duration=float(data.get("format", {}).get("duration") or 0.0))
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and not info.has_video:
            if (s.get("disposition") or {}).get("attached_pic"):
                continue
            info.has_video = True
            info.width = int(s.get("width") or 0)
            info.height = int(s.get("height") or 0)
            num, _, den = (s.get("avg_frame_rate") or "0/1").partition("/")
            try:
                info.fps = float(num) / float(den or 1)
            except (ValueError, ZeroDivisionError):
                info.fps = 0.0
            if not info.duration and s.get("duration"):
                info.duration = float(s["duration"])
        elif s.get("codec_type") == "audio":
            info.has_audio = True
    if info.duration <= 0:
        raise MediaError(f"Duration nahi mili: {path}")
    return info


_PTS_RE = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")


def detect_scenes(
    movie: str | Path,
    threshold: float = 0.3,
    duration: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> list[float]:
    """Returns the timestamps (seconds) where a new shot starts (hard cuts)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(movie),
        "-an", "-sn", "-dn",
        "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace",
    )
    cuts: list[float] = []
    assert proc.stderr is not None
    for line in proc.stderr:
        if "showinfo" not in line:
            continue
        m = _PTS_RE.search(line)
        if m:
            t = float(m.group(1))
            cuts.append(t)
            if progress and duration:
                progress(min(t / duration, 1.0))
    if proc.wait() != 0:
        raise MediaError("Scene detection fail hua (ffmpeg).")
    return sorted(set(round(c, 3) for c in cuts))


def iter_frames(
    movie: str | Path, fps: float = 1.0, size: int = 224
) -> Iterator[tuple[float, np.ndarray]]:
    """Streams square RGB frames (size x size) at `fps`, yielding (time, frame)."""
    vf = (
        f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=increase,"
        f"crop={size}:{size}"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(movie),
        "-an", "-sn", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    frame_bytes = size * size * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    k = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield k / fps, np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 3)
            k += 1
    finally:
        proc.stdout.close()
        proc.wait()


_SIL_START = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SIL_END = re.compile(r"silence_end:\s*([0-9.]+)")


def silence_intervals(
    audio: str | Path, noise_db: float = -35.0, min_dur: float = 0.12
) -> list[tuple[float, float]]:
    proc = run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(audio),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}", "-f", "null", "-"]
    )
    out: list[tuple[float, float]] = []
    start: float | None = None
    for line in proc.stderr.splitlines():
        m = _SIL_START.search(line)
        if m:
            start = max(0.0, float(m.group(1)))
            continue
        m = _SIL_END.search(line)
        if m and start is not None:
            out.append((start, float(m.group(1))))
            start = None
    if start is not None:
        out.append((start, probe(audio).duration))
    return out


def concat_entry(path: Path) -> str:
    """One line of an ffmpeg concat list; quotes inside the path are escaped."""
    escaped = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{escaped}'"


def fmt_ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def parse_ts(value: str | float | int) -> float:
    """Accepts seconds ('75.5') or [hh:]mm:ss[.ms] ('1:15', '00:01:15.500')."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        raise ValueError("khaali timestamp")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"galat timestamp: {value}")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total
