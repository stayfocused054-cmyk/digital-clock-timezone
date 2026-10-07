"""Local web app: `python -m recapcut app` opens http://127.0.0.1:7860 in the browser."""

from __future__ import annotations

import csv
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import gradio as gr

from . import cli, embed, media
from . import plan as plan_io

OUTPUT_ROOT = Path.home() / "recapcut_output"

VOICES = [
    ("Madhur - Hindi, male", "hi-IN-MadhurNeural"),
    ("Swara - Hindi, female", "hi-IN-SwaraNeural"),
    ("Guy - English, male", "en-US-GuyNeural"),
    ("Jenny - English, female", "en-US-JennyNeural"),
]
LANGUAGES = [("Auto", ""), ("Hindi / Hinglish", "hi"), ("English", "en")]
VIDEO_TYPES = ["video", ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"]
AUDIO_TYPES = ["audio", ".mp3", ".wav", ".m4a", ".aac", ".ogg"]
SCRIPT_PLACEHOLDER = """Raju ek chhote se gaon me rehta hai.
Ek din usse ek ajeeb chitthi milti hai!
[00:25:10] Station par uski mulaqat Meera se hoti hai.

([hh:mm:ss] hint optional hai: us line ki clip movie ke usi time se aayegi)"""

# One job at a time: logs are collected through a shared sink.
_job_lock = threading.Lock()


def status_markdown() -> str:
    try:
        media.require_ffmpeg()
        items = ["✅ ffmpeg"]
    except media.MediaError:
        items = ["❌ **ffmpeg nahi mila - pehle install karo** (`winget install Gyan.FFmpeg`)"]
    for mod, label in [("faster_whisper", "Whisper timing"), ("sentence_transformers", "CLIP visual matching"),
                       ("edge_tts", "AI voice"), ("anthropic", "Claude AI")]:
        items.append(("✅ " if embed.has_module(mod) else "➖ ") + label)
    return " &nbsp;·&nbsp; ".join(items)


def _stream(fn: Callable[[], object]) -> Iterator[tuple[str, dict]]:
    """Runs `fn` in a thread and yields (log so far, result) while it works.
    result gets 'value' or 'error' once finished."""
    q: queue.Queue = queue.Queue()
    result: dict = {}

    def work() -> None:
        with _job_lock:
            cli.log_sinks.append(q.put)
            try:
                result["value"] = fn()
            except SystemExit as e:
                result["error"] = str(e.code) if not isinstance(e.code, int) else "Ruk gaya."
            except Exception as e:  # shown to the user in the log box
                result["error"] = f"{type(e).__name__}: {e}"
            finally:
                cli.log_sinks.remove(q.put)
                q.put(None)

    threading.Thread(target=work, daemon=True).start()
    lines: list[str] = []
    while True:
        msg = q.get()
        if msg is None:
            break
        lines.append(msg)
        yield "\n".join(lines), result
    if "error" in result:
        lines.append(f"\n❌ {result['error']}")
    yield "\n".join(lines), result


def _existing(path: str | None, what: str) -> str | None:
    if not path:
        return None
    p = Path(str(path).strip().strip('"').strip("'"))
    if not p.exists():
        raise gr.Error(f"{what} nahi mila: {p}")
    return str(p)


def _plan_rows(plan_csv: Path) -> list[list[str]]:
    with plan_csv.open(newline="", encoding="utf-8-sig") as f:
        return [[row.get(k, "") for k in plan_io.CSV_FIELDS] for row in csv.DictReader(f)]


def make_video(movie_file, movie_path, audio_file, script_text, subs_file, language, voice,
               fmt, max_clip, movie_volume, burn_subs, skip_start, skip_end,
               use_ai, api_key, movie_title):
    movie = _existing(movie_path, "Movie") or _existing(movie_file, "Movie")
    if not movie:
        raise gr.Error("Movie file upload karo ya uska path paste karo.")
    audio = _existing(audio_file, "Audio")
    subs = _existing(subs_file, "Subtitles")
    script_text = (script_text or "").strip()
    if not audio and not script_text:
        raise gr.Error("Narration audio do, ya script likho (ya dono).")

    job = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    job.mkdir(parents=True, exist_ok=True)
    out = job / "recap.mp4"
    argv = ["make", "--movie", movie, "--out", str(out), "--format", fmt,
            "--max-clip", str(max_clip), "--movie-volume", str(movie_volume),
            "--skip-start", str(skip_start or 0), "--skip-end", str(skip_end or 0)]
    if audio:
        argv += ["--audio", audio]
    else:
        argv += ["--voice", voice]
    if script_text:
        script_path = job / "script.txt"
        script_path.write_text(script_text, encoding="utf-8")
        argv += ["--script", str(script_path)]
    if subs:
        argv += ["--subs", subs]
    if language:
        argv += ["--language", language]
    if burn_subs:
        argv.append("--burn-subs")
    if use_ai:
        if api_key and api_key.strip():
            os.environ["ANTHROPIC_API_KEY"] = api_key.strip()
        argv.append("--ai")
        if movie_title and movie_title.strip():
            argv += ["--movie-title", movie_title.strip()]
    args = cli.build_parser().parse_args(argv)

    for log_text, result in _stream(lambda: cli.run_make(args)):
        if "value" in result:
            plan_csv, video = result["value"]
            yield (log_text, str(video) if video else None, _plan_rows(plan_csv),
                   str(plan_csv), str(plan_csv))
        else:
            yield log_text, gr.skip(), gr.skip(), gr.skip(), gr.skip()


def rerender(plan_csv, table, fmt, movie_volume, burn_subs):
    if not plan_csv:
        raise gr.Error("Pehle 'Video Banao' chalao, phir table edit karke dobara render karo.")
    plan_csv = Path(plan_csv)
    rows = table.values.tolist() if hasattr(table, "values") else list(table or [])
    if not rows:
        raise gr.Error("Table khaali hai.")
    with plan_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(plan_io.CSV_FIELDS)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    try:
        plan_io.load(plan_csv)
    except (ValueError, KeyError) as e:
        raise gr.Error(f"Table me koi value galat hai (movie_time jaise 00:12:30.5 likho): {e}")

    out = plan_csv.parent.parent / f"recap_edit_{datetime.now():%H%M%S}.mp4"
    argv = ["render", "--plan", str(plan_csv), "--out", str(out), "--format", fmt,
            "--movie-volume", str(movie_volume)]
    if burn_subs:
        argv.append("--burn-subs")
    args = cli.build_parser().parse_args(argv)
    for log_text, result in _stream(lambda: cli.cmd_render(args)):
        video = str(out) if "value" in result and out.exists() else gr.skip()
        yield log_text, video


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="recapcut") as demo:
        gr.Markdown("# 🎬 recapcut\nNarration ke hisaab se movie me se clips apne aap kaat kar recap video banao.")
        gr.Markdown(status_markdown())
        plan_state = gr.State("")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Movie")
                movie_file = gr.File(label="Movie file", file_types=VIDEO_TYPES, type="filepath")
                movie_path = gr.Textbox(label="Ya movie ka path paste karo (badi file ke liye tez)",
                                        placeholder=r"D:\Movies\movie.mkv")
                subs_file = gr.File(label="Movie ke subtitles .srt (recommended - accuracy badhti hai)",
                                    file_types=[".srt", ".vtt"], type="filepath")

                gr.Markdown("### 2. Narration")
                audio_file = gr.File(label="Aapki AI voice (mp3/wav)", file_types=AUDIO_TYPES,
                                     type="filepath")
                script_text = gr.Textbox(label="Script (jo voice me bola gaya hai)", lines=8,
                                         placeholder=SCRIPT_PLACEHOLDER)
                with gr.Row():
                    language = gr.Dropdown(LANGUAGES, value="hi", label="Narration ki bhasha")
                    voice = gr.Dropdown(VOICES, value="hi-IN-MadhurNeural",
                                        label="AI voice (sirf jab audio na do)")

                with gr.Accordion("3. Video settings", open=False):
                    fmt = gr.Radio([("YouTube 16:9", "landscape"), ("Shorts / Reels 9:16", "portrait")],
                                   value="landscape", label="Format")
                    max_clip = gr.Slider(1.5, 8, value=3, step=0.5,
                                         label="Ek cut max kitne second (kam = fast-paced)")
                    movie_volume = gr.Slider(0, 0.5, value=0, step=0.02,
                                             label="Movie ki original awaaz (0 = band)")
                    burn_subs = gr.Checkbox(label="Narration subtitles video par likho")
                    with gr.Row():
                        skip_start = gr.Number(value=0, label="Shuru ke itne sec chhodo (logos)")
                        skip_end = gr.Number(value=0, label="Aakhri itne sec chhodo (credits)")

                with gr.Accordion("4. Claude AI - sabse accurate (optional)", open=False):
                    gr.Markdown("Claude movie ke subtitles padh kar har line ka time batata hai. "
                                "Subtitles zaroori hain. Ek movie par lagbhag $0.10-0.30.")
                    use_ai = gr.Checkbox(label="Claude AI use karo")
                    api_key = gr.Textbox(label="Anthropic API key", type="password",
                                         placeholder="sk-ant-... (ya ANTHROPIC_API_KEY set ho to khaali chhodo)")
                    movie_title = gr.Textbox(label="Movie ka naam aur saal", placeholder="Movie Name (2023)")

                go = gr.Button("🎬 Video Banao", variant="primary", size="lg")

            with gr.Column(scale=1):
                video = gr.Video(label="Recap video", interactive=False)
                log_box = gr.Textbox(label="Progress", lines=16, max_lines=16, autoscroll=True,
                                     interactive=False)

        gr.Markdown("### Clips plan\nKoi clip galat lage to us row ka **movie_time** badlo "
                    "(jaise `00:12:30.5`), phir neeche wala button dabao. Matching dobara nahi hogi.")
        table = gr.Dataframe(headers=plan_io.CSV_FIELDS, interactive=True, wrap=True, max_height=420)
        with gr.Row():
            redo = gr.Button("🔁 Table ke hisaab se dobara render")
            plan_download = gr.File(label="plan.csv", interactive=False)

        go.click(
            make_video,
            inputs=[movie_file, movie_path, audio_file, script_text, subs_file, language, voice,
                    fmt, max_clip, movie_volume, burn_subs, skip_start, skip_end,
                    use_ai, api_key, movie_title],
            outputs=[log_box, video, table, plan_state, plan_download],
            concurrency_limit=1, concurrency_id="job",
        )
        redo.click(
            rerender,
            inputs=[plan_state, table, fmt, movie_volume, burn_subs],
            outputs=[log_box, video],
            concurrency_limit=1, concurrency_id="job",
        )
    return demo


def launch(port: int = 7860, share: bool = False, open_browser: bool = True) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Videos yahan save hongi: {OUTPUT_ROOT}")
    build_ui().queue().launch(
        server_name="127.0.0.1", server_port=port, share=share, inbrowser=open_browser,
        allowed_paths=[str(OUTPUT_ROOT)], theme=gr.themes.Soft(),
    )
