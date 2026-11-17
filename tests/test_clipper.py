import json

import numpy as np

from clipper import captions, highlights, layout, signals
from clipper.util import parse_time

WORDS = [{"w": w, "s": i * 0.4, "e": i * 0.4 + 0.3} for i, w in enumerate(
    "okay so watch this. no way bro what the hell just happened, that was insane!".split())]
SEGS = [{"s": 0.0, "e": 1.5, "text": "okay so watch this."},
        {"s": 1.6, "e": 5.9, "text": "no way bro what the hell just happened, that was insane!"}]


def test_parse_time():
    assert parse_time("1:02:03.5") == 3723.5
    assert parse_time("90") == 90


def test_group_words_breaks_on_punctuation_and_length():
    groups = captions.group_words(WORDS)
    assert all(len(g) <= 3 for g in groups)
    assert groups[0][-1]["w"] in ("so", "watch", "this.")
    assert any(g[-1]["w"].endswith(".") for g in groups)


def test_build_ass_relative_times_and_highlight():
    ass = captions.build_ass(WORDS, 1.0, 6.0, 900, title="No way")
    lines = [l for l in ass.splitlines() if l.startswith("Dialogue")]
    assert "NO WAY" in lines[0]
    assert all(not l.split(",")[1].startswith("-") for l in lines)
    assert any("\\c&H004DE1FF" in l for l in lines[1:])


def test_refine_snaps_and_clamps():
    c = highlights.refine({"start": 0.5, "end": 200.0}, WORDS, 15, 60, 300)
    assert 59 <= c["end"] - c["start"] <= 61
    c = highlights.refine({"start": 1.65, "end": 5.0}, WORDS, 2, 60, 300)
    assert abs(c["start"] - (1.6 - 0.15)) < 0.5


def test_dedupe_drops_overlaps():
    cands = [{"start": 0, "end": 30, "score": 90}, {"start": 5, "end": 35, "score": 80},
             {"start": 100, "end": 130, "score": 70}]
    out = highlights.dedupe(cands, 5)
    assert [c["score"] for c in out] == [90, 70]


def test_pick_with_llm_parses_model_output(monkeypatch):
    fake = [{"start": "1.6", "end": "5.9", "score": 88, "title": "NO WAY", "caption": "#gaming", "reason": "x"},
            {"start": "bad"}]
    monkeypatch.setattr(highlights, "ask_llm", lambda *a, **k: fake)
    energy = np.zeros(20)
    clips = highlights.pick_with_llm({"words": WORDS, "segments": SEGS}, energy, None, 0, 3, 2, 60, "m", "ctx")
    assert len(clips) == 1 and clips[0]["start"] == 1.6 and clips[0]["score"] == 88


def test_build_lines_tags_loud_segments():
    energy = np.zeros(20)
    energy[6:10] = 3.2
    lines = highlights.build_lines({"words": WORDS, "segments": SEGS}, energy, None, 0)
    assert "LOUD+3" in lines[1][1] and "LOUD" not in lines[0][1]


def test_signal_fallback_finds_peak():
    n = int(600 / signals.HOP)
    energy = np.zeros(n)
    energy[600:640] = 5  # spike at 300s
    tr = {"words": [{"w": "wow!", "s": 301.0, "e": 301.4}], "segments": [{"s": 301.0, "e": 301.4, "text": "wow!"}]}
    clips = highlights.pick_with_signals(tr, energy, None, 0, 2, 15, 60)
    assert clips and clips[0]["start"] < 300 < clips[0]["end"]


def test_load_chat_twitchdownloader_and_csv(tmp_path):
    j = tmp_path / "c.json"
    j.write_text(json.dumps({"comments": [{"content_offset_seconds": 12.5, "message": {"body": "KEKW"}}]}))
    assert signals.load_chat(j) == [(12.5, "KEKW")]
    c = tmp_path / "c.csv"
    c.write_text("3,hi\nbad,row\n4.5,LUL\n")
    assert signals.load_chat(c) == [(3.0, "hi"), (4.5, "LUL")]


def test_filtergraphs_have_even_sizes():
    for info in ({"layout": "split", "facecam": [0.0, 0.4444, 0.2396, 0.2278]},
                 {"layout": "crop", "face": [0.5, 0.3, 0.1, 0.15]}, {"layout": "blur"}):
        g, y = layout.filtergraph(info, 1920, 1080)
        assert "[v]" in g and 0 < y < layout.OUT_H
        for tok in g.replace(";", ",").split(","):
            if tok.startswith("scale=") and ":" in tok:
                w, h = tok[6:].split(":")[:2]
                if w.lstrip("-").isdigit() and h.isdigit():
                    assert int(h) % 2 == 0, tok
