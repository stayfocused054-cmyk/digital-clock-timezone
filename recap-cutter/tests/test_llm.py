import json
import sys
import types

from recapcut import llm
from recapcut.subtitles import Cue


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


def _fake_anthropic(monkeypatch, payload, stop_reason="end_turn", seen=None):
    msg = types.SimpleNamespace(
        stop_reason=stop_reason,
        content=[types.SimpleNamespace(type="text", text=json.dumps(payload))],
    )

    class Messages:
        def stream(self, **kw):
            if seen is not None:
                seen.update(kw)
            return _Stream(msg)

    class Client:
        def __init__(self):
            self.beta = types.SimpleNamespace(messages=Messages())

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Client))


def test_claude_hints_parses_and_clamps(monkeypatch):
    seen = {}
    _fake_anthropic(monkeypatch, {"matches": [
        {"line": 0, "start": "0:00:10", "end": "0:00:20", "confidence": 0.9},
        {"line": 1, "start": "0:02:00", "end": "0:01:50", "confidence": 3},
        {"line": 7, "start": "0:00:01", "end": "0:00:02", "confidence": 0.5},
    ]}, seen=seen)
    hints = llm.claude_hints(["a", "b", "c"], [Cue(1, 2, "hi")], 115.0, "Film", "claude-opus-5-5",
                             log=lambda *_: None)
    assert (hints[0].start, hints[0].end, hints[0].confidence) == (10, 20, 0.9)
    assert (hints[1].start, hints[1].end, hints[1].confidence) == (110, 115, 1.0)
    assert hints[2] is None
    assert seen["model"] == "claude-opus-5-5"
    assert seen["output_config"]["format"]["type"] == "json_schema"
    assert "[00:00:01] hi" in seen["messages"][0]["content"]


def test_claude_hints_refusal_returns_none(monkeypatch):
    _fake_anthropic(monkeypatch, {}, stop_reason="refusal")
    assert llm.claude_hints(["a"], [Cue(1, 2, "x")], 10, None, "m", log=lambda *_: None) == [None]
