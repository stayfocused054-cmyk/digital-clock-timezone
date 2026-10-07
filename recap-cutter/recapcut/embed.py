"""Similarity backends.

* NgramText   : always available. Character n-gram TF-IDF, works when narration and
                subtitles share a language/script (names, key words).
* SemanticText: sentence-transformers multilingual model. Hindi narration vs English
                subtitles works too.
* ClipVisual  : CLIP image embeddings of movie frames vs multilingual CLIP text
                embeddings of narration ("car chase", "gun fight", "rain me rona").
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from . import media
from .script import normalize


def has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# --------------------------------------------------------------------------- text

def _grams(text: str, n_values=(2, 3, 4)) -> Counter:
    t = normalize(text)
    c: Counter = Counter()
    for n in n_values:
        for i in range(len(t) - n + 1):
            c[t[i:i + n]] += 1
    return c


def ngram_similarity(queries: list[str], docs: list[str]) -> np.ndarray:
    """TF-IDF cosine similarity (queries x docs) via an inverted index."""
    doc_grams = [_grams(d) for d in docs]
    q_grams = [_grams(q) for q in queries]
    df: Counter = Counter()
    for g in doc_grams + q_grams:
        df.update(g.keys())
    n_docs = len(doc_grams) + len(q_grams)
    idf = {g: math.log((1 + n_docs) / (1 + c)) + 1.0 for g, c in df.items()}

    def weigh(c: Counter) -> dict[str, float]:
        w = {g: (1 + math.log(v)) * idf[g] for g, v in c.items()}
        norm = math.sqrt(sum(x * x for x in w.values())) or 1.0
        return {g: x / norm for g, x in w.items()}

    index: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for j, g in enumerate(doc_grams):
        for gram, w in weigh(g).items():
            index[gram].append((j, w))
    out = np.zeros((len(queries), len(docs)), dtype=np.float32)
    for i, g in enumerate(q_grams):
        row = out[i]
        for gram, w in weigh(g).items():
            for j, dw in index.get(gram, ()):
                row[j] += w * dw
    return out


class SemanticText:
    MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.MODEL, device=_device())

    def similarity(self, queries: list[str], docs: list[str]) -> np.ndarray:
        has = np.array([bool(d.strip()) for d in docs])
        q = self.model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
        d = self.model.encode([d or "-" for d in docs], convert_to_numpy=True,
                              normalize_embeddings=True, batch_size=64)
        sim = (q @ d.T).astype(np.float32)
        sim[:, ~has] = 0.0
        return sim


# --------------------------------------------------------------------------- visual

class ClipVisual:
    IMAGE_MODEL = "clip-ViT-B-32"
    TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        dev = _device()
        self.image_model = SentenceTransformer(self.IMAGE_MODEL, device=dev)
        self.text_model = SentenceTransformer(self.TEXT_MODEL, device=dev)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        return self.text_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def encode_movie(
        self,
        movie: Path,
        cache_dir: Path,
        fps: float,
        duration: float,
        log: Callable[[str], None] = print,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-frame embeddings at `fps`, cached next to the work dir. Returns (times, emb)."""
        key = file_key(movie, f"clip|{fps}")
        cache = cache_dir / f"frames_{key}.npz"
        if cache.exists():
            data = np.load(cache)
            return data["times"], data["emb"]
        from PIL import Image

        log(f"Movie ke frames dekh raha hoon (CLIP, {fps} frame/sec) - pehli baar time lagega...")
        times: list[float] = []
        chunks: list[np.ndarray] = []
        batch: list = []
        last_pct = -1
        for t, frame in media.iter_frames(movie, fps=fps, size=224):
            times.append(t)
            batch.append(Image.fromarray(frame))
            if len(batch) == 64:
                chunks.append(self._encode_images(batch))
                batch = []
                pct = int(100 * t / max(duration, 1))
                if pct // 10 != last_pct // 10:
                    log(f"  frames: {pct}%")
                    last_pct = pct
        if batch:
            chunks.append(self._encode_images(batch))
        emb = np.concatenate(chunks).astype(np.float16) if chunks else np.zeros((0, 512), np.float16)
        t_arr = np.array(times, dtype=np.float32)
        np.savez(cache, times=t_arr, emb=emb)
        return t_arr, emb

    def _encode_images(self, images: list) -> np.ndarray:
        return self.image_model.encode(images, convert_to_numpy=True,
                                       normalize_embeddings=True, batch_size=len(images))


def file_key(path: Path, extra: str = "") -> str:
    """Identifies a file by its size and a sample of its bytes, so the same movie hits the
    cache even when it is copied, renamed or uploaded again."""
    size = path.stat().st_size
    h = hashlib.sha1(f"{size}|{extra}".encode())
    chunk = 4 * 1024 * 1024
    with path.open("rb") as f:
        h.update(f.read(chunk))
        if size > 2 * chunk:
            f.seek(size // 2)
            h.update(f.read(chunk))
            f.seek(-chunk, 2)
            h.update(f.read(chunk))
    return h.hexdigest()[:16]
