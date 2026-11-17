"""Pick the best moments. An LLM reads the timestamped transcript (annotated with loudness / chat spikes)
and acts as a short-form editor; without an LLM, falls back to pure signal peaks."""
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .signals import HOP, smooth, window_max, zscore_local
from .util import fmt_time, log

SYSTEM = """You are a top short-form video editor who cuts Twitch/YouTube livestream VODs into viral \
TikTok / YouTube Shorts / Reels clips.

You get a transcript of part of a stream. Each line is `[start_seconds-end_seconds] text`, optionally \
tagged with signals: `LOUD+n` = the streamer/game got much louder than usual (yelling, laughing, hype; n is \
how extreme, 2 = notable, 4+ = huge), `CHAT+n` = chat exploded. Long silences are shown as `(... Ns no speech ...)` \
— these are often gameplay, and if loud they may be clutch plays.

Find the moments that would perform best as standalone vertical clips:
- funny moments, big reactions, rage, clutch plays/hype, unexpected twists, wholesome or relatable moments, \
spicy takes, great one-liners, interactions with chat/donations that land without context
- each clip must make sense to someone who has never seen the stream: a clear setup and a payoff
- start right before the setup so the first 1-2 seconds hook the viewer (no slow intros, no "uh so anyway")
- end right after the payoff/reaction lands; don't trail off
- skip: dead air, ads, "be right back", technical issues, reading sub/follow alerts with nothing happening, \
anything that relies on long prior context
- {min_len}-{max_len} seconds each, ideally 20-45s. Clips must not overlap.

Score each 0-100 for viral potential, be critical: an average stream moment is 30; only truly great ones get 80+.

Respond with JSON only:
{{"clips": [{{"start": <seconds>, "end": <seconds>, "score": <0-100>, "category": "funny|hype|rage|clutch|wholesome|take|other", \
"title": "<on-screen hook, max 7 words, punchy, no hashtags, no quotes>", \
"caption": "<post caption with 2-4 relevant hashtags>", "reason": "<one sentence: why this works>"}}]}}"""


def build_lines(transcript, energy, chat, offset):
    """Transcript segments as annotated lines, inserting gap markers for silent stretches."""
    lines, prev_end = [], None
    for seg in transcript["segments"]:
        if prev_end is not None and seg["s"] - prev_end > 8:
            loud = window_max(energy, prev_end, seg["s"], offset)
            tag = f" LOUD+{loud:.0f}" if loud >= 2 else ""
            lines.append((prev_end, f"[{prev_end:.0f}-{seg['s']:.0f}] (... {seg['s'] - prev_end:.0f}s no speech ...){tag}"))
        tags = ""
        loud = window_max(energy, seg["s"], seg["e"], offset)
        if loud >= 2:
            tags += f" LOUD+{loud:.0f}"
        if chat is not None:
            c = window_max(chat, seg["s"], seg["e"] + 6, offset)
            if c >= 2:
                tags += f" CHAT+{c:.0f}"
        lines.append((seg["s"], f"[{seg['s']:.1f}-{seg['e']:.1f}]{tags} {seg['text']}"))
        prev_end = seg["e"]
    return lines


def chunk_lines(lines, max_chars=60000, overlap_s=120):
    chunks, cur, size = [], [], 0
    for t, line in lines:
        if size + len(line) > max_chars and cur:
            chunks.append(cur)
            cut = cur[-1][0] - overlap_s
            cur = [(tt, ll) for tt, ll in cur if tt >= cut]
            size = sum(len(ll) for _, ll in cur)
        cur.append((t, line))
        size += len(line) + 1
    if cur:
        chunks.append(cur)
    return ["\n".join(l for _, l in c) for c in chunks]


def ask_llm(text, model, n, min_len, max_len, context):
    from openai import OpenAI

    client = OpenAI()
    user = (f"Stream: {context}\nReturn up to {n} clips, best first. Only include moments that are genuinely good.\n\n"
            f"TRANSCRIPT:\n{text}")
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM.format(min_len=min_len, max_len=max_len)},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
            )
            return json.loads(r.choices[0].message.content).get("clips", [])
        except Exception as e:
            if attempt == 2:
                raise
            log(f"LLM retry ({e})")


def overlap(a, b):
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    return max(0.0, inter) / max(1e-6, min(a["end"] - a["start"], b["end"] - b["start"]))


def dedupe(cands, n):
    picked = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if all(overlap(c, p) < 0.25 for p in picked):
            picked.append(c)
        if len(picked) >= n:
            break
    return picked


def pick_with_llm(transcript, energy, chat, offset, n, min_len, max_len, model, context):
    lines = build_lines(transcript, energy, chat, offset)
    if not lines:
        return []
    chunks = chunk_lines(lines)
    per_chunk = max(4, int(np.ceil(n * 1.5 / len(chunks))) + 2)
    log(f"asking {model} to review {len(chunks)} transcript chunk(s)")
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda t: ask_llm(t, model, per_chunk, min_len, max_len, context), chunks))
    cands = []
    for c in (c for r in results for c in r):
        try:
            c["start"], c["end"], c["score"] = float(c["start"]), float(c["end"]), float(c.get("score", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if c["end"] > c["start"]:
            cands.append(c)
    return dedupe(cands, n)


REACTION = re.compile(
    r"\b(no way|let'?s go+|oh my god|oh my gosh|omg|what the|wtf|holy|bro+|yo+|ha(ha)+|lol|lmao|"
    r"no no|wait wait|are you (serious|kidding)|i can'?t|dude|sheesh|clutch|insane|crazy|"
    r"fuck\w*|shit|damn|hell nah|nah+|why+|how+)\b|!", re.I)


def transcript_curves(transcript, n_bins, offset):
    """Per-bin speech rate and 'reaction phrase' density, both locally z-scored."""
    rate = np.zeros(n_bins)
    react = np.zeros(n_bins)
    for w in transcript["words"]:
        i = int((w["s"] - offset) / HOP)
        if 0 <= i < n_bins:
            rate[i] += 1
    for seg in transcript["segments"]:
        hits = len(REACTION.findall(seg["text"]))
        if hits:
            a, b = int((seg["s"] - offset) / HOP), int((seg["e"] - offset) / HOP) + 1
            react[max(0, a):min(n_bins, b)] += hits / max(1, b - a) * 4
    rate = smooth(rate, int(4 / HOP))
    react = smooth(react, int(6 / HOP))
    rz = np.clip(zscore_local(rate, 300), -2, 4) if rate.any() else rate
    return rz, np.clip(react / (react.std() + 1e-6), 0, 5)


def _auto_title(transcript, start, end):
    segs = [s["text"] for s in transcript["segments"] if start <= s["s"] <= end]
    sents = [x.strip() for t in segs for x in re.split(r"(?<=[.!?])\s+", t) if 2 <= len(x.split()) <= 8]
    if not sents:
        return "WAIT FOR IT"
    best = max(sents, key=lambda x: (len(REACTION.findall(x)), x.count("!"), -len(x)))
    return " ".join(best.rstrip(".").split()[:7])


def pick_with_signals(transcript, energy, chat, offset, n, min_len, max_len):
    """No-LLM fallback: combine loudness, speech bursts, reaction phrases (and chat) and window the peaks."""
    rate, react = transcript_curves(transcript, len(energy), offset)
    curve = np.clip(energy, -2, 6) + 0.5 * rate + 0.8 * react
    if chat is not None:
        curve = curve + np.clip(chat, -2, 6)
    curve = smooth(curve, int(2 / HOP))
    target = min(max_len, max(min_len, 30))
    picked = []
    for i in np.argsort(-curve):
        if len(picked) >= n or curve[i] < 1.0:
            break
        peak = offset + i * HOP
        c = {"start": max(offset, peak - 0.65 * target), "end": peak + 0.35 * target}
        if all(overlap(c, p) == 0 for p in picked):
            c.update(score=round(100 * (1 - np.exp(-float(curve[i]) / 6)), 1), category="hype",
                     title=_auto_title(transcript, c["start"], c["end"]), caption="",
                     reason=f"loudness/reaction spike at {fmt_time(peak)}")
            c["caption"] = c["title"]
            picked.append(c)
    return picked


def refine(clip, words, min_len, max_len, total_end):
    """Fix duration and snap the cut points to natural speech boundaries."""
    s, e = clip["start"], clip["end"]
    if e - s > max_len:  # keep the payoff, trim the run-up
        s = e - max_len
    if e - s < min_len:
        d = min_len - (e - s)
        s, e = s - 0.6 * d, e + 0.4 * d
    starts = [w["s"] for w in words if s - 1.5 <= w["s"] <= s + 1.0]
    if starts:
        # prefer a word that follows a pause (sentence start), else the nearest word
        gaps = {w["s"]: w["s"] - p["e"] for p, w in zip(words, words[1:]) if w["s"] in starts}
        good = [t for t in starts if gaps.get(t, 1) > 0.35]
        s = min(good or starts, key=lambda t: abs(t - s)) - 0.15
    ends = [(w["e"], nxt["s"] - w["e"]) for w, nxt in zip(words, words[1:] + [{"s": 1e12}]) if e - 1.0 <= w["e"] <= e + 2.0]
    if ends:
        good = [t for t, gap in ends if gap > 0.35]
        e = min(good or [t for t, _ in ends], key=lambda t: abs(t - e)) + 0.45
    clip["start"], clip["end"] = round(max(0.0, s), 2), round(min(total_end, e), 2)
    return clip
