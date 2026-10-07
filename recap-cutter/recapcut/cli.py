"""Command line: `python -m recapcut make ...`, `render`, `app`, `check`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__, embed, matcher, media, narration
from . import plan as plan_io
from .narration import MissingDependency, Segment
from .render import RenderOptions, render
from .script import parse_script
from .subtitles import Cue, load_subtitles, text_between, write_srt


# Extra listeners (the web app shows progress through this).
log_sinks: list = []


def log(msg: str) -> None:
    print(msg, flush=True)
    for sink in log_sinks:
        sink(msg)


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "recapcut"


# --------------------------------------------------------------------------- narration

def build_narration(args, workdir: Path, cache_dir: Path) -> tuple[Path, float, list[Segment]]:
    lines = []
    if args.script:
        lines = parse_script(Path(args.script).read_text(encoding="utf-8-sig"))
        if not lines:
            raise SystemExit("Script khaali hai.")
        log(f"Script: {len(lines)} lines mili.")

    if args.audio:
        audio = Path(args.audio)
        total = media.probe(audio).duration
        log(f"Narration audio: {total:.1f} sec")
        use_whisper = not args.no_whisper and embed.has_module("faster_whisper")
        if lines:
            segs = None
            if use_whisper:
                prompt = " ".join(ln.text for ln in lines)[:400]
                words, _ = narration.transcribe(audio, cache_dir, args.language,
                                                args.whisper_model, prompt, log)
                segs = narration.align_script_to_words(lines, words)
                if segs is None:
                    log("  Whisper ka text script se match nahi hua, pauses se timing nikal raha hoon.")
                else:
                    log("  Script ki har line whisper timing se jod di.")
            if segs is None:
                if not use_whisper:
                    log("  faster-whisper nahi mila, audio ke pauses se line timing nikal raha hoon.")
                segs = narration.align_script_by_silence(lines, audio, total)
        else:
            if not use_whisper:
                raise SystemExit(
                    "Sirf audio diya hai, script nahi. Audio ka text samajhne ke liye "
                    "`pip install faster-whisper` karo, ya --script bhi do."
                )
            _, segs = narration.transcribe(audio, cache_dir, args.language,
                                           args.whisper_model, None, log)
            if not segs:
                raise SystemExit("Audio me koi bol nahi mila.")
    else:
        if not lines:
            raise SystemExit("--audio ya --script me se kam se kam ek do.")
        audio, segs = narration.synthesize(lines, args.voice, args.tts_rate, args.tts_gap,
                                           workdir, log)
        total = media.probe(audio).duration
    return audio, total, narration.fill_timeline(segs, total)


# --------------------------------------------------------------------------- scenes

def load_scenes(movie: Path, info: media.MediaInfo, cache_dir: Path, threshold: float) -> list[float]:
    cache = cache_dir / f"scenes_{embed.file_key(movie, str(threshold))}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    log("Movie ke scene cuts dhoondh raha hoon (pehli baar thoda time lagega)...")
    last = [-1]

    def progress(p: float) -> None:
        pct = int(p * 100)
        if pct // 10 != last[0] // 10:
            last[0] = pct
            log(f"  scenes: {pct}%")

    cuts = media.detect_scenes(movie, threshold, info.duration, progress)
    cache.write_text(json.dumps(cuts))
    return cuts


# --------------------------------------------------------------------------- make

def cmd_make(args) -> int:
    run_make(args)
    return 0


def run_make(args) -> tuple[Path, Path | None]:
    """Builds the plan (and the video unless --plan-only). Returns (plan.csv, video or None)."""
    media.require_ffmpeg()
    movie = Path(args.movie)
    if not movie.exists():
        raise SystemExit(f"Movie file nahi mili: {movie}")
    out = Path(args.out)
    workdir = Path(args.workdir) if args.workdir else out.with_name(out.stem + "_work")
    workdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    info = media.probe(movie)
    if not info.has_video:
        raise SystemExit("Movie file me video stream nahi mili.")
    log(f"Movie: {movie.name} ({media.fmt_ts(info.duration)[:8]}, {info.width}x{info.height})")

    audio, total, segs = build_narration(args, workdir, cache_dir)
    srt_path = workdir / "narration.srt"
    write_srt([Cue(s.start, s.end, s.text) for s in segs], srt_path)
    pieces = narration.split_pieces(segs, args.max_clip)
    log(f"Narration: {len(segs)} lines -> {len(pieces)} cuts (har cut max {args.max_clip}s)")

    cuts = load_scenes(movie, info, cache_dir, args.scene_threshold)
    units = matcher.build_units(cuts, info.duration, args.skip_start, args.skip_end,
                                max_len=args.max_shot)
    log(f"Movie: {len(cuts)} scene cuts -> {len(units)} candidate shots")

    n, m = len(pieces), len(units)
    seg_texts = [s.text for s in segs]
    piece_seg = np.array([p.seg_index for p in pieces])
    sig = matcher.Signals()
    w = matcher.Weights(visual=args.w_visual, text=args.w_text, llm=args.w_ai,
                        position=args.w_order, back_penalty=args.back_penalty,
                        repeat_penalty=args.repeat_penalty)

    cues: list[Cue] = []
    if args.subs:
        cues = load_subtitles(args.subs)
        log(f"Subtitles: {len(cues)} lines")
        unit_texts = [text_between(cues, u.start - 0.5, u.end + 0.5) for u in units]
        semantic = not args.lite and embed.has_module("sentence_transformers")
        if semantic:
            log("Dialogue matching: multilingual AI model")
            seg_sim = embed.SemanticText().similarity(seg_texts, unit_texts)
        else:
            log("Dialogue matching: simple word matching (AI model ke liye requirements-ai.txt install karo)")
            seg_sim = embed.ngram_similarity(seg_texts, unit_texts)
        sig.text = seg_sim[piece_seg]

    frame_times = frame_sims = None
    if not args.lite and not args.no_visual:
        if embed.has_module("sentence_transformers"):
            clip = embed.ClipVisual()
            frame_times, frame_emb = clip.encode_movie(movie, cache_dir, args.frame_rate,
                                                       info.duration, log)
            if len(frame_times):
                seg_emb = clip.encode_text(seg_texts)
                unit_emb = matcher.unit_frame_means(units, frame_times, frame_emb)
                sig.visual = (seg_emb @ unit_emb.T)[piece_seg]
                frame_sims = (seg_emb @ frame_emb.astype(np.float32).T)[piece_seg]
        else:
            log("Visual matching off: `pip install -r requirements-ai.txt` se CLIP milega.")

    if args.ai:
        if not cues:
            log("--ai ke liye --subs (movie ki .srt) chahiye; AI hints skip.")
        else:
            from .llm import claude_hints

            hints = claude_hints(seg_texts, cues, info.duration, args.movie_title,
                                 args.ai_model, log)
            windows = [(h.start, h.end, h.confidence) if h else None for h in hints]
            sig.llm = matcher.window_bonus([windows[s] for s in piece_seg], units, w.llm_sigma)

    if not args.no_order:
        sig.position = matcher.position_prior(pieces, units, w.position_sigma)
    sig.hard_mask = matcher.hard_mask([p.hint for p in pieces], units, w.hint_pad)

    if sig.visual is None and sig.text is None and sig.llm is None and sig.hard_mask is None:
        log("Dhyan do: koi matching signal nahi hai (na subtitles, na CLIP, na AI, na [time] hints), "
            "isliye clips sirf story order me lagengi.")

    score = matcher.combine(sig, w, n, m)
    same_line = [i > 0 and pieces[i].seg_index == pieces[i - 1].seg_index for i in range(n)]
    path = matcher.align(score, w.back_penalty, w.repeat_penalty, same_line)
    starts = matcher.choose_starts(pieces, units, path, info.duration, frame_times, frame_sims)

    plan = plan_io.Plan(movie=str(movie.resolve()), narration_audio=str(Path(audio).resolve()),
                        narration_duration=total, srt=str(srt_path.resolve()))
    for i, (p, j, st) in enumerate(zip(pieces, path, starts)):
        why = " ".join(f"{k}:{v[i, j]:+.1f}" for k, v in sig.components.items())
        if p.hint:
            why = (why + " hint").strip()
        plan.clips.append(plan_io.Clip(p.start, p.end, st, p.text, float(score[i, j]), why))
    plan_json, plan_csv = workdir / "plan.json", workdir / "plan.csv"
    plan_io.save(plan, plan_json, plan_csv)

    log("\nEdit plan:")
    for k, c in enumerate(plan.clips, 1):
        log(f"  {k:3d}. narr {c.narr_start:6.2f}-{c.narr_end:6.2f}s  <-  movie {media.fmt_ts(c.movie_start)}"
            f"  | {c.text[:60]}")
    log(f"\nPlan save hua: {plan_csv}\n(Koi clip galat lage to CSV me movie_time badlo aur "
        f"`python -m recapcut render --plan \"{plan_csv}\" --out \"{out}\"` chalao.)")

    if args.plan_only:
        return plan_csv, None
    render(plan, out, workdir, render_options(args), log)
    log(f"\nHo gaya! Video: {out.resolve()}")
    return plan_csv, out


def render_options(args) -> RenderOptions:
    return RenderOptions(fmt=args.format, portrait_mode=args.portrait_mode, height=args.height,
                         fps=args.fps, crf=args.crf, movie_volume=args.movie_volume,
                         burn_subs=args.burn_subs, workers=args.workers)


def cmd_render(args) -> int:
    plan = plan_io.load(Path(args.plan))
    out = Path(args.out)
    workdir = Path(args.workdir) if args.workdir else Path(args.plan).resolve().parent
    render(plan, out, workdir, render_options(args), log)
    log(f"\nHo gaya! Video: {out.resolve()}")
    return 0


def cmd_app(args) -> int:
    try:
        from .app import launch
    except ImportError:
        raise SystemExit("App ke liye gradio chahiye: pip install -r requirements-app.txt")
    launch(port=args.port, share=args.share, open_browser=not args.no_browser)
    return 0


def cmd_check(_args) -> int:
    rows = []
    try:
        media.require_ffmpeg()
        rows.append(("ffmpeg", True, "zaroori"))
    except media.MediaError:
        rows.append(("ffmpeg", False, "zaroori - install karo"))
    for mod, use in [("faster_whisper", "audio se line timing (accurate)"),
                     ("sentence_transformers", "CLIP visual + multilingual dialogue matching"),
                     ("edge_tts", "sirf script ho to free AI voice"),
                     ("anthropic", "--ai: Claude se timestamps")]:
        rows.append((mod, embed.has_module(mod), use))
    for name, ok, use in rows:
        log(f"  [{'OK' if ok else '--'}] {name:22s} {use}")
    if embed.has_module("torch"):
        log(f"  device: {embed._device()}")
    return 0


# --------------------------------------------------------------------------- argparse

def add_render_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("video output")
    g.add_argument("--format", choices=["landscape", "portrait"], default="landscape",
                   help="landscape = YouTube 16:9, portrait = Shorts/Reels 9:16")
    g.add_argument("--portrait-mode", choices=["blur", "crop"], default="blur")
    g.add_argument("--height", type=int, default=1080)
    g.add_argument("--fps", type=int, default=30)
    g.add_argument("--crf", type=int, default=20, help="quality: kam = behtar (18-23)")
    g.add_argument("--movie-volume", type=float, default=0.0,
                   help="movie ki original awaaz kitni rakhni hai (0 = band, 0.1 = halki)")
    g.add_argument("--burn-subs", action="store_true", help="narration ke subtitles video par likho")
    g.add_argument("--workers", type=int, default=0)
    g.add_argument("--workdir", help="beech ki files kahan rakhni hain")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="recapcut",
        description="Narration ke hisaab se movie me se clips kaat kar recap video banata hai.",
    )
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make", help="movie + narration se recap video banao")
    mk.add_argument("--movie", required=True, help="poori movie (mp4/mkv/...)")
    mk.add_argument("--audio", help="aapka narration audio (AI voice mp3/wav)")
    mk.add_argument("--script", help="aapki likhi script (.txt), [hh:mm:ss] hints allowed")
    mk.add_argument("--subs", help="movie ke subtitles (.srt/.vtt) - accuracy bahut badhta hai")
    mk.add_argument("--out", default="recap.mp4")
    mk.add_argument("--cache-dir", help="scene/CLIP cache (default ~/.cache/recapcut)")

    g = mk.add_argument_group("narration")
    g.add_argument("--language", default=None, help="narration ki bhasha, jaise hi / en (whisper)")
    g.add_argument("--whisper-model", default="small", help="tiny/base/small/medium/large-v3")
    g.add_argument("--no-whisper", action="store_true", help="whisper mat chalao, pauses se timing lo")
    g.add_argument("--voice", default="hi-IN-MadhurNeural",
                   help="sirf script ho to edge-tts voice (hi-IN-SwaraNeural, en-US-GuyNeural...)")
    g.add_argument("--tts-rate", default="+0%", help="AI voice speed, jaise +10%%")
    g.add_argument("--tts-gap", type=float, default=0.25, help="lines ke beech ka pause (sec)")
    g.add_argument("--max-clip", type=float, default=3.0,
                   help="ek cut zyada se zyada kitne second ka ho (lambi line kai cuts me batt jaati hai)")

    g = mk.add_argument_group("matching")
    g.add_argument("--ai", action="store_true", help="Claude se subtitles padhwa kar timestamps lo (API key chahiye)")
    g.add_argument("--ai-model", default="claude-opus-5-5")
    g.add_argument("--movie-title", help="movie ka naam (--ai ko madad milti hai)")
    g.add_argument("--lite", action="store_true", help="AI models (CLIP/semantic) mat use karo")
    g.add_argument("--no-visual", action="store_true", help="CLIP visual matching band")
    g.add_argument("--no-order", action="store_true", help="story-order wala andaza band")
    g.add_argument("--skip-start", type=float, default=0.0, help="movie ke shuru ke itne sec chhodo (logos)")
    g.add_argument("--skip-end", type=float, default=0.0, help="movie ke aakhri itne sec chhodo (credits)")
    g.add_argument("--scene-threshold", type=float, default=0.3)
    g.add_argument("--max-shot", type=float, default=8.0)
    g.add_argument("--frame-rate", type=float, default=1.0, help="CLIP ke liye frames/sec")
    g.add_argument("--w-visual", type=float, default=1.0)
    g.add_argument("--w-text", type=float, default=1.0)
    g.add_argument("--w-ai", type=float, default=3.0)
    g.add_argument("--w-order", type=float, default=0.4)
    g.add_argument("--back-penalty", type=float, default=3.0)
    g.add_argument("--repeat-penalty", type=float, default=1.5)
    mk.add_argument("--plan-only", action="store_true", help="sirf plan.csv banao, video nahi")
    add_render_args(mk)
    mk.set_defaults(func=cmd_make)

    rd = sub.add_parser("render", help="(edit kiye hue) plan.csv/plan.json se video banao")
    rd.add_argument("--plan", required=True)
    rd.add_argument("--out", default="recap.mp4")
    add_render_args(rd)
    rd.set_defaults(func=cmd_render)

    app = sub.add_parser("app", help="browser wala app kholo (http://127.0.0.1:7860)")
    app.add_argument("--port", type=int, default=7860)
    app.add_argument("--share", action="store_true",
                     help="temporary public link bhi banao (phone se kholne ke liye)")
    app.add_argument("--no-browser", action="store_true")
    app.set_defaults(func=cmd_app)

    ck = sub.add_parser("check", help="kaun se features available hain")
    ck.set_defaults(func=cmd_check)
    return ap


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (media.MediaError, MissingDependency, FileNotFoundError) as e:
        log(f"\nError: {e}")
        return 1
