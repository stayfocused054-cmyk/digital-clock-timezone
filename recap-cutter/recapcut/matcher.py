"""Scoring every (narration piece, movie shot) pair and picking one shot per piece.

Signals (each optional, combined after per-row z-normalisation):
  visual   - CLIP: does the shot look like what the line says?
  text     - does the dialogue around the shot match the line?
  llm      - Claude's timestamp guess for the line (soft window bonus)
  position - recaps follow story order, so line k of N sits near k/N of the movie
User hints in the script ([12:30]) are hard constraints.

A dynamic program then picks one shot per piece maximising total score, with a penalty
for jumping backwards in the movie and for showing the same shot twice in a row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .narration import Segment


@dataclass
class Unit:
    start: float
    end: float

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


@dataclass
class Weights:
    visual: float = 1.0
    text: float = 1.0
    llm: float = 3.0
    position: float = 0.4
    back_penalty: float = 3.0
    repeat_penalty: float = 1.5
    hint_pad: float = 3.0
    llm_sigma: float = 20.0
    position_sigma: float = 0.2


@dataclass
class Signals:
    visual: np.ndarray | None = None
    text: np.ndarray | None = None
    llm: np.ndarray | None = None
    position: np.ndarray | None = None
    hard_mask: np.ndarray | None = None
    components: dict[str, np.ndarray] = field(default_factory=dict)


def build_units(
    cuts: list[float],
    duration: float,
    skip_start: float = 0.0,
    skip_end: float = 0.0,
    min_len: float = 0.6,
    max_len: float = 8.0,
) -> list[Unit]:
    lo = max(0.0, skip_start)
    hi = max(lo + 1.0, duration - max(0.0, skip_end))
    pts = [lo] + [c for c in cuts if lo + 1e-3 < c < hi - 1e-3] + [hi]
    shots: list[Unit] = []
    for a, b in zip(pts[:-1], pts[1:]):
        if shots and (b - a < min_len or shots[-1].end - shots[-1].start < min_len):
            shots[-1].end = b
        else:
            shots.append(Unit(a, b))
    units: list[Unit] = []
    for s in shots:
        length = s.end - s.start
        k = max(1, math.ceil(length / max_len))
        edges = np.linspace(s.start, s.end, k + 1)
        units.extend(Unit(float(x), float(y)) for x, y in zip(edges[:-1], edges[1:]))
    return units


def zrows(m: np.ndarray) -> np.ndarray:
    mean = m.mean(axis=1, keepdims=True)
    std = m.std(axis=1, keepdims=True)
    out = (m - mean) / np.where(std < 1e-6, 1.0, std)
    out[(std < 1e-6).ravel()] = 0.0
    return out


def position_prior(pieces: list[Segment], units: list[Unit], sigma: float) -> np.ndarray:
    total = pieces[-1].end if pieces else 1.0
    lo, hi = units[0].start, units[-1].end
    f = np.array([(p.start + p.end) / 2 / total for p in pieces])
    g = np.array([(u.mid - lo) / max(hi - lo, 1e-6) for u in units])
    return np.maximum(-((g[None, :] - f[:, None]) ** 2) / (2 * sigma ** 2), -4.0)


def window_bonus(
    windows: list[tuple[float, float, float] | None], units: list[Unit], sigma: float
) -> np.ndarray:
    starts = np.array([u.start for u in units])
    ends = np.array([u.end for u in units])
    out = np.zeros((len(windows), len(units)), dtype=np.float32)
    for i, w in enumerate(windows):
        if w is None:
            continue
        a, b, conf = w
        dist = np.maximum(0.0, np.maximum(a - ends, starts - b))
        out[i] = conf * np.exp(-(dist ** 2) / (2 * sigma ** 2))
    return out


def hard_mask(
    hints: list[tuple[float, float] | None], units: list[Unit], pad: float
) -> np.ndarray | None:
    if not any(hints):
        return None
    starts = np.array([u.start for u in units])
    ends = np.array([u.end for u in units])
    mask = np.ones((len(hints), len(units)), dtype=bool)
    for i, h in enumerate(hints):
        if h is None:
            continue
        a, b = h[0] - pad, h[1] + pad
        ok = (ends >= a) & (starts <= b)
        if not ok.any():
            ok = np.zeros(len(units), dtype=bool)
            ok[int(np.argmin(np.abs((starts + ends) / 2 - (a + b) / 2)))] = True
        mask[i] = ok
    return mask


def combine(sig: Signals, w: Weights, n: int, m: int) -> np.ndarray:
    score = np.zeros((n, m), dtype=np.float64)
    if sig.visual is not None:
        part = w.visual * zrows(sig.visual)
        sig.components["visual"] = part
        score += part
    if sig.text is not None:
        part = w.text * zrows(sig.text)
        sig.components["text"] = part
        score += part
    if sig.llm is not None:
        part = w.llm * sig.llm
        sig.components["ai"] = part
        score += part
    if sig.position is not None:
        part = w.position * sig.position
        sig.components["order"] = part
        score += part
    if sig.hard_mask is not None:
        score = np.where(sig.hard_mask, score, score - 1000.0)
    return score


def align(
    score: np.ndarray,
    back_penalty: float,
    repeat_penalty: float,
    same_line: list[bool] | None = None,
) -> list[int]:
    """Viterbi over shots: moving forward is free, staying on a shot costs `repeat_penalty`,
    going backwards costs `back_penalty`. Where `same_line[i]` is true (piece i continues the
    sentence of piece i-1) staying is free, because the footage simply keeps running."""
    n, m = score.shape
    ar = np.arange(m)
    d = score[0].copy()
    back = np.zeros((n, m), dtype=np.int64)
    for i in range(1, n):
        # best strictly-earlier shot (prefix max, exclusive)
        cm = np.maximum.accumulate(d)
        cidx = np.maximum.accumulate(np.where(d >= cm, ar, 0))
        fwd_val = np.concatenate([[-np.inf], cm[:-1]])
        fwd_idx = np.concatenate([[0], cidx[:-1]])
        # best strictly-later shot (suffix max, exclusive)
        rd = d[::-1]
        rcm = np.maximum.accumulate(rd)
        ridx = np.maximum.accumulate(np.where(rd >= rcm, ar, 0))
        bwd_val = np.concatenate([rcm[:-1][::-1], [-np.inf]]) - back_penalty
        bwd_idx = np.concatenate([(m - 1 - ridx[:-1])[::-1], [0]])
        stay_val = d - (0.0 if same_line and same_line[i] else repeat_penalty)

        cand = np.stack([fwd_val, stay_val, bwd_val])
        choice = np.argmax(cand, axis=0)
        best = cand[choice, ar]
        back[i] = np.choose(choice, [fwd_idx, ar, bwd_idx])
        d = score[i] + best
    j = int(np.argmax(d))
    path = [j]
    for i in range(n - 1, 0, -1):
        j = int(back[i, j])
        path.append(j)
    return path[::-1]


def choose_starts(
    pieces: list[Segment],
    units: list[Unit],
    path: list[int],
    movie_duration: float,
    frame_times: np.ndarray | None = None,
    frame_sims: np.ndarray | None = None,
) -> list[float]:
    """Exact movie start time for each piece inside its chosen shot."""
    starts: list[float] = []
    prev_j, prev_end = -1, 0.0
    for i, (p, j) in enumerate(zip(pieces, path)):
        u = units[j]
        d = p.end - p.start
        slack = (u.end - u.start) - d
        if j == prev_j and prev_end + d <= u.end + 0.5 * d:
            start = prev_end  # same shot again: keep the footage running
        elif slack <= 0:
            start = u.start
        elif frame_sims is not None and frame_times is not None:
            cands = np.arange(u.start, u.start + slack + 1e-6, 0.5)
            best, start = -np.inf, u.start
            for c in cands:
                sel = (frame_times >= c) & (frame_times < c + d)
                if sel.any():
                    val = float(frame_sims[i, sel].mean())
                    if val > best:
                        best, start = val, float(c)
        else:
            start = u.start + min(0.3, slack)
        start = float(min(max(start, 0.0), max(movie_duration - d - 0.05, 0.0)))
        starts.append(start)
        prev_j, prev_end = j, start + d
    return starts


def unit_frame_means(
    units: list[Unit], frame_times: np.ndarray, frame_emb: np.ndarray
) -> np.ndarray:
    """Average CLIP embedding of the frames inside each unit."""
    out = np.zeros((len(units), frame_emb.shape[1]), dtype=np.float32)
    emb = frame_emb.astype(np.float32)
    for k, u in enumerate(units):
        sel = (frame_times >= u.start) & (frame_times < u.end)
        if sel.any():
            v = emb[sel].mean(axis=0)
        else:
            v = emb[int(np.argmin(np.abs(frame_times - u.mid)))]
        out[k] = v / (np.linalg.norm(v) or 1.0)
    return out
