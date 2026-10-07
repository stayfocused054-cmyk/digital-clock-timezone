"""Cutting the chosen clips with frame-exact lengths and laying the narration on top."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import media
from .plan import Plan


@dataclass
class RenderOptions:
    fmt: str = "landscape"         # landscape | portrait
    portrait_mode: str = "blur"    # blur | crop
    height: int = 1080
    fps: int = 30
    crf: int = 20
    preset: str = "veryfast"
    movie_volume: float = 0.0      # 0 = only narration, 0.1 = halka movie sound
    burn_subs: bool = False
    workers: int = 0


def _size(opts: RenderOptions) -> tuple[int, int]:
    h = opts.height - opts.height % 2
    if opts.fmt == "portrait":
        w = int(round(h * 9 / 16)) // 2 * 2
        return w, h
    return int(round(h * 16 / 9)) // 2 * 2, h


def video_filter(opts: RenderOptions) -> str:
    w, h = _size(opts)
    head = f"fps={opts.fps}"
    tail = "setsar=1,tpad=stop_mode=clone:stop_duration=3"
    if opts.fmt == "portrait" and opts.portrait_mode == "blur":
        return (
            f"{head},split[a][b];"
            f"[a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=20:3[bg];"
            f"[b]scale={w}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,{tail}"
        )
    if opts.fmt == "portrait":
        return f"{head},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},{tail}"
    return (
        f"{head},scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,{tail}"
    )


def render(
    plan: Plan,
    out: Path,
    workdir: Path,
    opts: RenderOptions,
    log: Callable[[str], None] = print,
) -> Path:
    media.require_ffmpeg()
    movie = Path(plan.movie)
    narration = Path(plan.narration_audio)
    movie_info = media.probe(movie)
    use_movie_audio = opts.movie_volume > 0 and movie_info.has_audio

    clip_dir = workdir / "clips"
    if clip_dir.exists():
        shutil.rmtree(clip_dir)
    clip_dir.mkdir(parents=True)

    # Frame-exact boundaries: round the cumulative timeline, not each clip, so no drift.
    fps = opts.fps
    frames = [round(c.narr_end * fps) - round(c.narr_start * fps) for c in plan.clips]
    vf = video_filter(opts)

    def cut(k: int) -> Path:
        c = plan.clips[k]
        n = max(frames[k], 1)
        dur = n / fps
        path = clip_dir / f"clip_{k:04d}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{c.movie_start:.3f}",
               "-i", str(movie), "-map", "0:v:0", "-vf", vf, "-frames:v", str(n),
               "-c:v", "libx264", "-preset", opts.preset, "-crf", str(opts.crf),
               "-pix_fmt", "yuv420p", "-r", str(fps)]
        if use_movie_audio:
            cmd += ["-map", "0:a:0", "-af", f"apad,atrim=0:{dur:.4f}",
                    "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]
        else:
            cmd += ["-an"]
        media.run(cmd + [str(path)])
        return path

    todo = [k for k in range(len(plan.clips)) if frames[k] > 0]
    workers = opts.workers or max(1, min(4, (os.cpu_count() or 2) // 2))
    log(f"{len(todo)} clips kaat raha hoon ({workers} saath me)...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(cut, todo))

    concat_list = workdir / "concat.txt"
    concat_list.write_text(
        "\n".join(media.concat_entry(p) for p in paths), encoding="utf-8"
    )
    joined = workdir / "joined.mp4"
    media.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", str(concat_list), "-c", "copy", str(joined)])

    log("Narration audio jod raha hoon...")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined.resolve()),
           "-i", str(narration.resolve())]
    if use_movie_audio:
        cmd += ["-filter_complex",
                f"[0:a]volume={opts.movie_volume}[m];[1:a][m]amix=inputs=2:duration=first:normalize=0[a]",
                "-map", "0:v", "-map", "[a]"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
    if opts.burn_subs and plan.srt and Path(plan.srt).exists():
        # Run from the srt's folder so the filter argument needs no path escaping.
        srt = Path(plan.srt).resolve()
        cmd += ["-vf", f"subtitles={srt.name}:force_style='FontSize=18,Outline=2,MarginV=40'",
                "-c:v", "libx264", "-preset", opts.preset, "-crf", str(opts.crf)]
        cwd: Path | None = srt.parent
    else:
        cmd += ["-c:v", "copy"]
        cwd = None
    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
            str(out.resolve())]
    media.run(cmd, cwd=cwd)
    return out
