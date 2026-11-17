"""Cheap, non-AI excitement signals: audio loudness spikes and (optionally) chat activity."""
import json
import re
import subprocess
from pathlib import Path

import numpy as np

HOP = 0.5  # seconds per bin


def loudness(wav):
    """dBFS per HOP-second bin, streaming the file so multi-hour VODs stay cheap on memory."""
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(wav), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                            stdout=subprocess.PIPE)
    n = int(16000 * HOP)
    out = []
    while True:
        buf = proc.stdout.read(n * 2 * 600)
        if not buf:
            break
        x = np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768
        k = len(x) // n
        if k:
            out.append(np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, axis=1) + 1e-10))
    proc.wait()
    rms = np.concatenate(out) if out else np.zeros(1)
    return 20 * np.log10(rms + 1e-6)


def _rolling_baseline(x, window_bins, step=20):
    """Rolling median + MAD, evaluated every `step` bins and interpolated (fast for long series)."""
    idx = np.arange(0, len(x), step)
    med, mad = [], []
    for i in idx:
        w = x[max(0, i - window_bins // 2): i + window_bins // 2 + 1]
        m = np.median(w)
        med.append(m)
        mad.append(np.median(np.abs(w - m)) + 1e-3)
    full = np.arange(len(x))
    return np.interp(full, idx, med), np.interp(full, idx, mad)


def smooth(x, bins):
    if bins <= 1:
        return x
    k = np.ones(bins) / bins
    return np.convolve(x, k, mode="same")


def zscore_local(x, window_s=300):
    med, mad = _rolling_baseline(x, int(window_s / HOP))
    return (x - med) / (1.4826 * mad)


def energy_curve(wav):
    db = loudness(wav)
    # Ignore near-silence so a quiet break doesn't make normal talking look "loud".
    db = np.maximum(db, np.percentile(db, 10))
    return smooth(zscore_local(db), int(3 / HOP))


HYPE = re.compile(r"\b(lul|lmao+|lmfao|kekw|omegalul|icant|pog+|poggers|pogchamp|w+|l+|clip( it|ped)?|"
                  r"no way|holy|wtf|sheesh|goat(ed)?|insane|crazy|\?\?+|!!+)\b|😂|💀|🤣|🔥", re.I)


def load_chat(path):
    """Parse a chat replay into [(offset_seconds, text)].

    Supports TwitchDownloader JSON ({"comments": [{"content_offset_seconds", "message": {"body"}}]}),
    a JSON list of {"time"/"offset", "text"/"message"}, or CSV/TSV lines `offset,text`."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    msgs = []
    try:
        d = json.loads(raw)
        items = d.get("comments", d) if isinstance(d, dict) else d
        for c in items:
            t = c.get("content_offset_seconds", c.get("time", c.get("offset", c.get("time_in_seconds"))))
            m = c.get("message", c.get("text", ""))
            if isinstance(m, dict):
                m = m.get("body", "") or "".join(f.get("text", "") for f in m.get("fragments", []))
            if t is not None:
                msgs.append((float(t), str(m)))
    except json.JSONDecodeError:
        for line in raw.splitlines():
            parts = re.split(r"[,\t]", line, maxsplit=1)
            try:
                msgs.append((float(parts[0]), parts[1] if len(parts) > 1 else ""))
            except ValueError:
                continue
    return sorted(msgs)


def chat_curve(msgs, n_bins, offset=0.0, lag=5.0):
    """Chat excitement per bin: message rate + weighted hype keywords, locally z-scored.
    Chat reacts a few seconds after the moment, so messages are shifted back by `lag` seconds."""
    rate = np.zeros(n_bins)
    for t, text in msgs:
        i = int((t - lag - offset) / HOP)
        if 0 <= i < n_bins:
            rate[i] += 1 + 1.5 * len(HYPE.findall(text))
    rate = smooth(rate, int(10 / HOP))
    if rate.max() <= 0:
        return np.zeros(n_bins)
    return smooth(zscore_local(rate, 600), int(4 / HOP))


def window_max(curve, start, end, offset=0.0):
    a, b = int((start - offset) / HOP), int((end - offset) / HOP) + 1
    seg = curve[max(0, a): max(a + 1, b)]
    return float(seg.max()) if len(seg) else 0.0
