"""End-to-end on a synthetic 'movie': 8 coloured scenes, each with its own dialogue line.
The narration mentions scenes 1, 2, 4 and 5 (in story order); the tool must cut exactly those."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from recapcut import cli, media
from recapcut import plan as plan_io

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")

SCENE = 6.0
COLORS = ["red", "black", "yellow", "blue", "white", "magenta", "cyan", "navy"]
RGB = {
    "red": (255, 0, 0), "black": (0, 0, 0), "yellow": (255, 255, 0), "blue": (0, 0, 255),
    "white": (255, 255, 255), "magenta": (255, 0, 255), "cyan": (0, 255, 255),
    "navy": (0, 0, 128),
}
DIALOGUE = [
    "Welcome to the village festival.",
    "Raju, the train to Mumbai leaves tonight.",
    "Inspector Vikram, we found the stolen diamonds.",
    "Nobody eats until the harvest is done.",
    "Meera, I will never forget this lighthouse.",
    "Give me the money or the godown burns!",
    "Rain again, the roof is leaking.",
    "Goodbye everyone, the end.",
]
SCRIPT = [
    (1, "Raju raat ki train se Mumbai jaane wala hai."),
    (2, "Wahan Inspector Vikram ko chori ke diamonds milte hain."),
    (4, "Meera lighthouse par Raju se kabhi na bhoolne ka waada karti hai."),
    (5, "Villain godown jalane ki dhamki deta hai aur money maangta hai."),
]


def _srt(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    inputs = []
    for c in COLORS:
        inputs += ["-f", "lavfi", "-i", f"color=c={c}:s=320x180:r=25:d={SCENE}"]
    n = len(COLORS)
    filt = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs,
                    "-f", "lavfi", "-i", f"sine=f=220:d={SCENE * n}",
                    "-filter_complex", filt, "-map", "[v]", "-map", f"{n}:a",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    str(d / "movie.mp4")], check=True)
    (d / "movie.srt").write_text("\n".join(
        f"{i + 1}\n{_srt(i * SCENE + 1)} --> {_srt(i * SCENE + 4)}\n{t}\n"
        for i, t in enumerate(DIALOGUE)), encoding="utf-8")
    (d / "script.txt").write_text("\n".join(t for _, t in SCRIPT), encoding="utf-8")

    # "AI voice": one tone per sentence, 0.6 s pauses between them.
    parts, filt = [], ""
    for k, (_, text) in enumerate(SCRIPT):
        dur = round(len(text) / 12, 2)
        parts += ["-f", "lavfi", "-i", f"sine=f={400 + 100 * k}:d={dur}",
                  "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=0.6"]
    m = len(SCRIPT)
    filt = "".join(f"[{i}:a]" for i in range(2 * m)) + f"concat=n={2 * m}:v=0:a=1[a]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *parts, "-filter_complex", filt,
                    "-map", "[a]", "-ar", "44100", "-ac", "1", str(d / "voice.wav")], check=True)
    return d


def _frame_rgb(video: Path, t: float) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video),
                          "-frames:v", "1", "-vf", "scale=16:9", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(float).mean(axis=0)


def test_make_lite_picks_right_scenes(assets, tmp_path):
    out = tmp_path / "recap.mp4"
    rc = cli.main(["make", "--movie", str(assets / "movie.mp4"), "--audio", str(assets / "voice.wav"),
                   "--script", str(assets / "script.txt"), "--subs", str(assets / "movie.srt"),
                   "--out", str(out), "--lite", "--no-whisper", "--height", "360",
                   "--cache-dir", str(tmp_path / "cache"), "--burn-subs"])
    assert rc == 0 and out.exists()

    plan = plan_io.load(tmp_path / "recap_work" / "plan.json")
    by_line = {}
    for c in plan.clips:
        by_line.setdefault(c.text, []).append(c)
    for scene, text in SCRIPT:
        for c in by_line[text]:
            assert scene * SCENE <= c.movie_start < (scene + 1) * SCENE, (text, c.movie_start)

    narr = media.probe(assets / "voice.wav").duration
    vid = media.probe(out)
    assert abs(vid.duration - narr) < 0.15
    assert vid.has_audio and vid.height == 360 and vid.width == 640

    # The picture under each clip has the colour of the scene it came from.
    for c in plan.clips:
        scene = int(c.movie_start // SCENE)
        got = _frame_rgb(out, (c.narr_start + c.narr_end) / 2)
        want = np.array(RGB[COLORS[scene]], dtype=float)
        assert np.abs(got - want).max() < 60, (c.text, got, want)


def test_hint_and_csv_edit_then_render(assets, tmp_path):
    script = tmp_path / "script.txt"
    lines = [t for _, t in SCRIPT]
    lines[0] = "[00:42] " + lines[0]  # force line 1 into scene 7 (white) via hint
    script.write_text("\n".join(lines), encoding="utf-8")
    out = tmp_path / "recap.mp4"
    rc = cli.main(["make", "--movie", str(assets / "movie.mp4"), "--audio", str(assets / "voice.wav"),
                   "--script", str(script), "--subs", str(assets / "movie.srt"),
                   "--out", str(out), "--lite", "--no-whisper", "--plan-only",
                   "--cache-dir", str(tmp_path / "cache")])
    assert rc == 0 and not out.exists()
    csv_path = tmp_path / "recap_work" / "plan.csv"
    plan = plan_io.load(csv_path)
    first = [c for c in plan.clips if c.text == lines[0][8:]]
    assert first and all(36 <= c.movie_start < 48 for c in first)

    # Hand-edit: move the last clip to the start of the movie (red), then render from CSV.
    rows = csv_path.read_text(encoding="utf-8-sig").splitlines()
    head, *body = rows
    cols = body[-1].split(",")
    cols[3] = "00:00:01.000"
    body[-1] = ",".join(cols)
    csv_path.write_text("\n".join([head, *body]) + "\n", encoding="utf-8-sig")
    rc = cli.main(["render", "--plan", str(csv_path), "--out", str(out), "--height", "360",
                   "--format", "portrait", "--movie-volume", "0.2"])
    assert rc == 0
    edited = plan_io.load(csv_path).clips[-1]
    assert edited.movie_start == 1.0
    vid = media.probe(out)
    assert vid.width < vid.height
    got = _frame_rgb(out, (edited.narr_start + edited.narr_end) / 2)
    assert got[0] > 150 and got[1] < 90 and got[2] < 90  # red-ish


class _FakeST:
    """Stands in for sentence-transformers: images embed as their mean colour, text as the
    colour words it mentions. Lets the CLIP code path run without downloading models."""

    WORDS = {k: np.array(v, dtype=float) / 255 for k, v in RGB.items()}

    def __init__(self, name, device=None):
        self.name = name

    def encode(self, items, convert_to_numpy=True, normalize_embeddings=True, batch_size=32):
        out = []
        for it in items:
            if isinstance(it, str):
                v = sum((vec for w, vec in self.WORDS.items() if w in it.lower()), np.zeros(3))
            else:
                v = np.asarray(it, dtype=float).reshape(-1, 3).mean(axis=0) / 255
            v = np.append(v, 0.05)
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


def test_visual_path_with_fake_clip(assets, tmp_path, monkeypatch):
    import sys
    import types

    from recapcut import embed

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=_FakeST))
    monkeypatch.setattr(embed, "has_module", lambda name: name == "sentence_transformers")
    script = tmp_path / "script.txt"
    script.write_text("Sab kuch yellow tha.\nPhir magenta raat aayi.\nAakhir me navy samundar.",
                      encoding="utf-8")
    out = tmp_path / "recap.mp4"
    args = ["make", "--movie", str(assets / "movie.mp4"), "--audio", str(assets / "voice.wav"),
            "--script", str(script), "--out", str(out), "--no-order", "--plan-only",
            "--cache-dir", str(tmp_path / "cache")]
    assert cli.main(args) == 0
    plan = plan_io.load(tmp_path / "recap_work" / "plan.json")
    want = {"yellow": 2, "magenta": 5, "navy": 7}
    for c in plan.clips:
        color = next(w for w in want if w in c.text)
        assert int(c.movie_start // SCENE) == want[color], (c.text, c.movie_start)
    assert list((tmp_path / "cache").glob("frames_*.npz"))
    assert cli.main(args) == 0  # second run reads the frame cache
