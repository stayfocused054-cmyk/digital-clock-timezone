import numpy as np

from recapcut import matcher
from recapcut.narration import Segment, Word, align_script_to_words, fill_timeline, split_pieces
from recapcut.script import parse_script
from recapcut.embed import ngram_similarity
from recapcut.media import parse_ts, fmt_ts


def test_parse_script_hints_and_sentences():
    text = (
        "# comment\n"
        "Raj gaon me rehta hai. Wo bahut garib hai!\n"
        "[12:30] Ek din usse chitthi milti hai।\n"
        "[1:05:00-1:06:10] Climax me ladai hoti hai.\n"
        "[00:20]\n"
        "Bare hint applies here.\n"
    )
    lines = parse_script(text)
    assert [ln.text for ln in lines] == [
        "Raj gaon me rehta hai.",
        "Wo bahut garib hai!",
        "Ek din usse chitthi milti hai।",
        "Climax me ladai hoti hai.",
        "Bare hint applies here.",
    ]
    assert lines[0].hint is None
    assert lines[2].hint == (750.0, 750.0)
    assert lines[3].hint == (3900.0, 3970.0)
    assert lines[4].hint == (20.0, 20.0)


def test_timestamps():
    assert parse_ts("1:15") == 75
    assert parse_ts("00:01:15.500") == 75.5
    assert parse_ts("42") == 42
    assert fmt_ts(3725.25) == "01:02:05.250"


def test_align_script_to_words_handles_transcription_errors():
    lines = parse_script("Raj gaon me rehta hai.\nEk din police aati hai.\nSab bhaag jaate hain.")
    words = [
        Word(0.0, 0.3, "Raaj"), Word(0.3, 0.6, "gaon"), Word(0.6, 0.8, "mein"),
        Word(0.8, 1.0, "rehta"), Word(1.0, 1.2, "hai."),
        Word(1.8, 2.0, "Ek"), Word(2.0, 2.2, "din"), Word(2.2, 2.6, "police"),
        Word(2.6, 2.9, "aati"), Word(2.9, 3.1, "hai."),
        Word(3.6, 3.8, "Sab"), Word(3.8, 4.2, "bhag"), Word(4.2, 4.5, "jate"), Word(4.5, 4.8, "hain"),
    ]
    segs = align_script_to_words(lines, words)
    assert segs is not None
    assert abs(segs[0].start - 0.0) < 0.31 and abs(segs[0].end - 1.2) < 0.01
    assert abs(segs[1].start - 1.8) < 0.01 and abs(segs[1].end - 3.1) < 0.01
    assert abs(segs[2].start - 3.6) < 0.01 and abs(segs[2].end - 4.8) < 0.01


def test_align_script_rejects_unrelated_transcript():
    lines = parse_script("Raj gaon me rehta hai.")
    words = [Word(0, 1, "राज"), Word(1, 2, "गाँव")]
    assert align_script_to_words(lines, words) is None


def test_fill_timeline_and_split():
    segs = [Segment(0.5, 2.0, "a"), Segment(2.4, 6.0, "b"), Segment(6.2, 7.0, "c")]
    full = fill_timeline(segs, 8.0)
    assert full[0].start == 0.0 and full[-1].end == 8.0
    for x, y in zip(full[:-1], full[1:]):
        assert x.end == y.start
    pieces = split_pieces(full, 2.0)
    assert abs(sum(p.end - p.start for p in pieces) - 8.0) < 1e-9
    assert max(p.end - p.start for p in pieces) <= 2.0 + 1e-9
    assert [p.seg_index for p in pieces][:1] == [0]


def test_build_units_merges_and_splits():
    units = matcher.build_units([5.0, 5.2, 30.0], 40.0, skip_start=1.0, skip_end=2.0,
                                min_len=0.6, max_len=8.0)
    assert units[0].start == 1.0 and units[-1].end == 38.0
    assert all(u.end - u.start <= 8.0 + 1e-9 for u in units)
    assert all(u.end - u.start >= 0.6 - 1e-9 for u in units)
    for x, y in zip(units[:-1], units[1:]):
        assert abs(x.end - y.start) < 1e-9


def test_align_prefers_forward_order():
    # piece 0 likes shot 5, piece 1 likes shot 2 slightly more than shot 7
    s = np.zeros((2, 10))
    s[0, 5] = 3.0
    s[1, 2] = 1.0
    s[1, 7] = 0.8
    assert matcher.align(s, back_penalty=3.0, repeat_penalty=1.5) == [5, 7]
    # a strong enough backward match still wins (4 + 7 - 3 beats 0 + 7)
    s[0, 5] = 4.0
    s[1, 2] = 7.0
    assert matcher.align(s, back_penalty=3.0, repeat_penalty=1.5) == [5, 2]


def test_align_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n, m = rng.integers(1, 5), rng.integers(1, 7)
        s = rng.normal(size=(n, m))
        path = matcher.align(s, 1.3, 0.7)

        def total(p):
            v = sum(s[i, j] for i, j in enumerate(p))
            for a, b in zip(p[:-1], p[1:]):
                v -= 0.7 if a == b else (1.3 if b < a else 0.0)
            return v

        import itertools
        best = max(total(p) for p in itertools.product(range(m), repeat=n))
        assert abs(total(path) - best) < 1e-9


def test_ngram_similarity_finds_shared_names():
    sim = ngram_similarity(
        ["Inspector Vikram police station pahunchta hai"],
        ["Where is the money?", "Vikram, welcome to the police station.", ""],
    )
    assert sim[0].argmax() == 1
    assert sim[0, 2] == 0.0


def test_hard_mask_and_window_bonus():
    units = [matcher.Unit(0, 10), matcher.Unit(10, 20), matcher.Unit(20, 30)]
    mask = matcher.hard_mask([None, (14.0, 14.0)], units, pad=1.0)
    assert mask[0].all() and mask[1].tolist() == [False, True, False]
    bonus = matcher.window_bonus([None, (21.0, 25.0, 1.0)], units, sigma=5.0)
    assert bonus[0].sum() == 0 and bonus[1].argmax() == 2
