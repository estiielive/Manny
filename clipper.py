#!/usr/bin/env python3
"""
Clipping pipeline for the Twitch tracker.

When a stream ends this:
  1. finds the stream's VOD and the most-viewed clips viewers made during it
     (Twitch Helix API),
  2. downloads them with TwitchDownloaderCLI   (github.com/lay295/TwitchDownloader),
  3. turns the VOD into ranked, vertical 9:16 shorts with
     AI-Youtube-Shorts-Generator                (github.com/Anil-matcha/AI-Youtube-Shorts-Generator).

Output lands in clips/<login>/<vod_id>/{twitch_clips,shorts}/.

Run it by hand for the latest VOD:
    python clipper.py              # estiie
    python clipper.py otherlogin
    python clipper.py estiie --vod 2312345678

Or let the tracker run it automatically: python track_estiie.py --clip

Settings (environment variables, all optional):
  CLIP_OUTPUT_DIR        where results go                      (default: clips)
  CLIP_TOP_N             how many Twitch clips to download     (default: 5, 0 = skip)
  CLIP_NUM_SHORTS        how many AI shorts to render          (default: 3, 0 = skip)
  CLIP_SHORTS_MODE       local (yt-dlp/whisper/ffmpeg + OpenAI or Gemini) or api (MuAPI)
                                                               (default: local)
  CLIP_VOD_MAX_MINUTES   only use the first N minutes of the VOD for shorts
                                                               (default: 0 = whole VOD)
  CLIP_ASPECT_RATIO      shorts aspect ratio                   (default: 9:16)
  CLIP_RANKING           ai (LLM picks moments) or loudness (loudest audio peaks,
                         no API key needed)                    (default: ai)
  TWITCHDOWNLOADER_CLI   path to TwitchDownloaderCLI           (default: bin/TwitchDownloaderCLI)
Plus the shorts generator's own keys: OPENAI_API_KEY or GEMINI_API_KEY (+ LLM_PROVIDER)
for local mode, MUAPI_API_KEY for api mode.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHORTS_REPO = ROOT / "vendor" / "AI-Youtube-Shorts-Generator"

_default_cli = ROOT / "bin" / ("TwitchDownloaderCLI.exe" if platform.system() == "Windows"
                               else "TwitchDownloaderCLI")
TD_CLI = os.environ.get("TWITCHDOWNLOADER_CLI", str(_default_cli))
OUT_ROOT = Path(os.environ.get("CLIP_OUTPUT_DIR", ROOT / "clips"))
TOP_N = int(os.environ.get("CLIP_TOP_N", "5"))
NUM_SHORTS = int(os.environ.get("CLIP_NUM_SHORTS", "3"))
SHORTS_MODE = os.environ.get("CLIP_SHORTS_MODE", "local").lower()
RANKING = os.environ.get("CLIP_RANKING", "ai").lower()
VOD_MAX_MINUTES = float(os.environ.get("CLIP_VOD_MAX_MINUTES", "0"))
ASPECT_RATIO = os.environ.get("CLIP_ASPECT_RATIO", "9:16")


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [clipper] {msg}", flush=True)


def _safe(name):
    return re.sub(r"[^\w.-]+", "_", name).strip("_")[:80] or "clip"


# --------------------------------------------------------------------------- #
# Twitch Helix lookups (reuses the tracker's authenticated api_get)
# --------------------------------------------------------------------------- #
def find_vod(api_get, user_id, stream_id=None):
    """The archived VOD for stream_id, or the latest archive if stream_id is None."""
    vods = api_get("videos", {"user_id": user_id, "type": "archive", "first": 20})
    if stream_id:
        vods = [v for v in vods if str(v.get("stream_id")) == str(stream_id)]
    return vods[0] if vods else None


def top_clips(api_get, user_id, started_at, ended_at, n):
    """Most-viewed clips created between started_at and ended_at (RFC3339)."""
    if n <= 0:
        return []
    clips = api_get("clips", {
        "broadcaster_id": user_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "first": 100,
    })
    return sorted(clips, key=lambda c: c.get("view_count", 0), reverse=True)[:n]


def vod_seconds(duration):
    """Parse Helix durations like '3h2m15s'."""
    total = 0
    for value, unit in re.findall(r"(\d+)([hms])", duration or ""):
        total += int(value) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


# --------------------------------------------------------------------------- #
# TwitchDownloader
# --------------------------------------------------------------------------- #
def _td(*args):
    if not Path(TD_CLI).exists() and not _on_path(TD_CLI):
        raise RuntimeError(
            f"TwitchDownloaderCLI not found at {TD_CLI}. "
            "Run scripts/install_twitchdownloader.sh or set TWITCHDOWNLOADER_CLI."
        )
    cmd = [TD_CLI, *args, "--collision", "Overwrite", "--banner", "false"]
    subprocess.run(cmd, check=True)


def _on_path(name):
    from shutil import which
    return which(name) is not None


def download_clip(clip, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{clip.get('view_count', 0):06d}_views_{_safe(clip.get('title', ''))}_{clip['id']}.mp4"
    log(f"Downloading clip '{clip.get('title')}' ({clip.get('view_count')} views)")
    _td("clipdownload", "--id", clip["id"], "-o", str(out))
    return out


def download_vod(vod, out_dir, max_minutes=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"vod_{vod['id']}.mp4"
    args = ["videodownload", "--id", vod["id"], "-o", str(out), "-q", "720p"]
    if max_minutes > 0:
        args += ["-e", f"{int(max_minutes * 60)}s"]
    log(f"Downloading VOD {vod['id']} ({vod.get('duration')})"
        + (f", first {max_minutes:g} min" if max_minutes > 0 else ""))
    _td(*args)
    return out


# --------------------------------------------------------------------------- #
# AI-Youtube-Shorts-Generator
# --------------------------------------------------------------------------- #
_patched = False


def _patch_shorts_generator():
    """Work around two upstream issues in AI-Youtube-Shorts-Generator.

    1. Videos over 30 min are split into chunks, but each chunk's transcript keeps
       absolute timestamps while highlights are validated against the chunk's own
       0..duration range and then shifted by the chunk offset. So every chunk after
       the first loses its highlights (or gets them shifted twice). Stream VODs are
       always long, so rebase each chunk's segments to start at 0.
    2. The google-genai SDK defaults to the v1 API, where current Gemini models
       (e.g. preview ones) aren't served; use v1beta.
    3. Each short is cut with `-ss` after `-i`, so ffmpeg decodes the VOD from the
       start every time (minutes per short on a multi-hour VOD). Seek before -i.
    4. Shorts come out as OpenCV's MPEG-4 Part 2 (~16 Mbps), which is huge and not
       what TikTok/Shorts/Reels expect. Re-encode the final file to H.264.
    """
    global _patched
    if _patched:
        return
    from shorts_generator import highlights as sg_highlights
    from shorts_generator.local import clipper as sg_clipper

    def cut_subclip_fast(source_path, start, end, out_path):
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", source_path, "-t", f"{end - start:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            out_path,
        ], check=True)
        return out_path

    sg_clipper._cut_subclip = cut_subclip_fast

    original_reframe = sg_clipper._reframe_vertical

    def reframe_h264(in_path, out_path, aspect_ratio):
        raw = out_path + ".mp4v.mp4"
        original_reframe(in_path, raw, aspect_ratio)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
                out_path,
            ], check=True)
        finally:
            if os.path.exists(raw):
                os.remove(raw)
        return out_path

    sg_clipper._reframe_vertical = reframe_h264

    original_chunker = sg_highlights.chunk_transcript

    def chunk_transcript_rebased(transcript):
        chunks = original_chunker(transcript)
        for chunk in chunks:
            off = chunk.get("_offset", 0)
            chunk["segments"] = [
                {**s, "start": s["start"] - off, "end": s["end"] - off}
                for s in chunk["segments"]
            ]
        return chunks

    sg_highlights.chunk_transcript = chunk_transcript_rebased

    # Cache the LLM's highlight picks next to the shorts so a failed render can be
    # retried without spending API calls (free tiers allow only ~20 requests/day).
    from shorts_generator import pipeline as sg_pipeline
    original_get_highlights = sg_pipeline.get_highlights

    def get_highlights_cached(transcript, num_clips=3, llm_fn=None):
        cache = Path(sg_clipper.LOCAL_OUTPUT_DIR) / "highlights.json"
        if cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            log(f"Reusing {len(cached['highlights'])} cached highlights from {cache}")
            return cached
        result = original_get_highlights(transcript, num_clips=num_clips, llm_fn=llm_fn)
        cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    sg_pipeline.get_highlights = get_highlights_cached

    # Retry transient LLM errors (overloaded / brief rate limits) instead of failing
    # the whole run on the first one.
    import time as _time
    from shorts_generator.local import llm as sg_llm
    original_llm = sg_llm.call_local_llm

    def call_llm_with_retry(prompt):
        delays = [10, 30, 60, 120]
        for attempt in range(len(delays) + 1):
            try:
                return original_llm(prompt)
            except Exception as e:
                msg = str(e)
                transient = any(k in msg for k in ("503", "UNAVAILABLE", "overloaded",
                                                   "high demand", "500", "502", "504"))
                # A per-minute 429 is worth waiting out; a daily quota is not.
                if "429" in msg and "per day" not in msg.lower() and "no credits" not in msg:
                    transient = True
                if not transient or attempt == len(delays):
                    raise
                log(f"LLM busy ({msg[:60]}...); retrying in {delays[attempt]}s")
                _time.sleep(delays[attempt])

    sg_llm.call_local_llm = call_llm_with_retry

    try:
        from google import genai
        from google.genai import types

        original_client = genai.Client

        def client_v1beta(**kwargs):
            kwargs.setdefault("http_options", types.HttpOptions(api_version="v1beta"))
            return original_client(**kwargs)

        genai.Client = client_v1beta
    except ImportError:
        pass  # Gemini not installed; OpenAI / MuAPI paths don't need this.
    _patched = True


def loudness_highlights(video_path, num_clips, before=20.0, after=10.0, min_gap=90.0,
                        transcript_segments=None):
    """Pick highlight windows around the loudest moments of the stream audio.

    No LLM needed: hype moments on stream (kills, clutches, screaming at chat) are
    loud. Audio RMS is measured per second, smoothed over 3 s, and the top peaks at
    least `min_gap` seconds apart become clips running from `before` seconds ahead
    of the peak (the build-up) to `after` seconds past it (the reaction).
    """
    import numpy as np

    rate = 8000
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(video_path), "-vn", "-ac", "1",
         "-ar", str(rate), "-f", "s16le", "-"],
        check=True, stdout=subprocess.PIPE,
    )
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    seconds = len(audio) // rate
    if seconds == 0:
        raise RuntimeError("No audio found in the VOD.")
    per_sec = audio[: seconds * rate].reshape(seconds, rate)
    rms_db = 20 * np.log10(np.sqrt((per_sec ** 2).mean(axis=1)) + 1e-6)
    # Twitch VODs have digitally silent stretches (muted copyrighted music, BRB
    # screens). Treat them as average level so the audio coming back after a mute
    # doesn't look like a huge spike, and never clip across them.
    silent = rms_db < -60
    if silent.all():
        raise RuntimeError("The VOD's audio is silent.")
    rms_db = np.where(silent, np.median(rms_db[~silent]), rms_db)
    smooth = np.convolve(rms_db, np.ones(3) / 3, mode="same")
    # Score each second against its surroundings (5-minute rolling median), so a
    # loud moment in a quiet stretch counts as much as one in a loud stretch.
    win = 300
    pad = np.pad(smooth, win // 2, mode="edge")
    baseline = np.array([np.median(pad[i:i + win]) for i in range(seconds)])
    score = smooth - baseline

    skip_intro = 120 if seconds > 600 else 0  # stream-starting screens / intro music
    picks = []
    for t in np.argsort(score)[::-1]:
        t = int(t)
        lo, hi = max(0, int(t - before)), min(seconds, int(t + after) + 1)
        if t < skip_intro or silent[lo:hi].any():
            continue
        if all(abs(t - p) >= min_gap for p in picks):
            picks.append(int(t))
        if len(picks) == num_clips:
            break

    def text_near(start, end):
        if not transcript_segments:
            return ""
        return " ".join(s["text"].strip() for s in transcript_segments
                        if s["end"] >= start and s["start"] <= end).strip()

    highlights = []
    for t in sorted(picks):
        start = max(0.0, t - before)
        end = min(float(seconds), t + after)
        words = text_near(t - 5, t + 5)
        stamp = f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"
        highlights.append({
            "title": (words[:60] + "...") if len(words) > 60 else (words or f"Hype moment at {stamp}"),
            "start_time": start,
            "end_time": end,
            "peak_time": float(t),
            "score": round(float(score[t]), 1),  # dB above the local baseline
            "hook_sentence": words[:120],
            "virality_reason": f"Audio peak {score[t]:.1f} dB above the surrounding 5 minutes at {stamp}",
        })
    return highlights


def make_shorts_by_loudness(video_path, out_dir, num_clips, aspect_ratio=ASPECT_RATIO):
    """Render shorts from the loudest moments of a local VOD file (no API key needed)."""
    if str(SHORTS_REPO) not in sys.path:
        sys.path.insert(0, str(SHORTS_REPO))
    from shorts_generator.local import clipper as sg_clipper
    _patch_shorts_generator()

    out_dir.mkdir(parents=True, exist_ok=True)
    segments = None
    srt = out_dir / (Path(video_path).stem + ".srt")
    if srt.exists():  # reuse a transcript if one was made earlier, for titles
        from shorts_generator.local import transcriber as sg_transcriber
        segments = sg_transcriber._load_srt_cache(srt).get("segments")

    log(f"Finding the {num_clips} loudest moments in {video_path}")
    highlights = loudness_highlights(video_path, num_clips, transcript_segments=segments)
    shorts = sg_clipper.crop_highlights_local(str(video_path), highlights,
                                              aspect_ratio=aspect_ratio, out_dir=str(out_dir))
    result = {"mode": "loudness", "source_video_url": str(video_path), "shorts": shorts}
    with open(out_dir / "shorts.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    for s in shorts:
        log(f"  short +{s['score']} dB '{s['title']}' -> "
            f"{s.get('clip_url') or 'FAILED: ' + str(s.get('error'))}")
    return result


def make_shorts(source, out_dir, num_clips, mode=SHORTS_MODE, aspect_ratio=ASPECT_RATIO):
    """Rank highlights in `source` (local path or URL) and render vertical shorts."""
    if not (SHORTS_REPO / "shorts_generator").is_dir():
        raise RuntimeError(
            f"{SHORTS_REPO} is missing. Run: git submodule update --init"
        )
    if str(SHORTS_REPO) not in sys.path:
        sys.path.insert(0, str(SHORTS_REPO))
    import shorts_generator
    from shorts_generator.local import clipper as sg_clipper
    from shorts_generator.local import transcriber as sg_transcriber

    _patch_shorts_generator()

    out_dir.mkdir(parents=True, exist_ok=True)
    # The generator writes to a module-level LOCAL_OUTPUT_DIR; point it at this stream.
    sg_clipper.LOCAL_OUTPUT_DIR = str(out_dir)
    sg_transcriber.LOCAL_OUTPUT_DIR = str(out_dir)

    log(f"Generating {num_clips} shorts ({mode} mode) from {source}")
    result = shorts_generator.generate_shorts(
        youtube_url=str(source),
        num_clips=num_clips,
        aspect_ratio=aspect_ratio,
        mode=mode,
    )
    result.pop("transcript", None)  # large; the generator caches an .srt anyway
    with open(out_dir / "shorts.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    for s in result.get("shorts", []):
        log(f"  short score={s.get('score')} '{s.get('title')}' -> "
            f"{s.get('clip_url') or 'FAILED: ' + str(s.get('error'))}")
    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def process_stream(api_get, user_id, login, stream_id=None, started_at=None, ended_at=None,
                   vod_id=None):
    """Download top clips and generate shorts for one stream. Never raises."""
    try:
        if vod_id:
            vods = api_get("videos", {"id": vod_id})
            vod = vods[0] if vods else None
        else:
            vod = find_vod(api_get, user_id, stream_id)
        if not vod and not started_at:
            log("No VOD found (are past broadcasts enabled on the channel?). Nothing to do.")
            return

        started_at = started_at or vod["created_at"]
        if not ended_at:
            if vod:
                start = datetime.fromisoformat(vod["created_at"].replace("Z", "+00:00"))
                end_ts = start.timestamp() + vod_seconds(vod.get("duration")) + 60
                ended_at = datetime.fromtimestamp(end_ts, timezone.utc)
            else:
                ended_at = datetime.now(timezone.utc)
            ended_at = ended_at.strftime("%Y-%m-%dT%H:%M:%SZ")

        out = OUT_ROOT / login / (vod["id"] if vod else started_at.replace(":", ""))
        log(f"Processing stream -> {out}")

        # 1. Viewer-made clips, via TwitchDownloader.
        try:
            clips = top_clips(api_get, user_id, started_at, ended_at, TOP_N)
            log(f"{len(clips)} viewer clip(s) to download")
            for c in clips:
                try:
                    download_clip(c, out / "twitch_clips")
                except Exception as e:
                    log(f"Clip {c['id']} failed: {e}")
        except Exception as e:
            log(f"Twitch clips step failed: {e}")

        # 2. AI shorts from the VOD, via TwitchDownloader + AI-Youtube-Shorts-Generator.
        if NUM_SHORTS > 0 and vod:
            try:
                if RANKING == "loudness":
                    source = download_vod(vod, out, VOD_MAX_MINUTES)
                    make_shorts_by_loudness(source, out / "shorts", NUM_SHORTS)
                else:
                    if SHORTS_MODE == "local":
                        source = download_vod(vod, out, VOD_MAX_MINUTES)
                    else:
                        source = vod["url"]  # MuAPI fetches it itself
                    make_shorts(source, out / "shorts", NUM_SHORTS)
            except Exception as e:
                log(f"Shorts step failed: {e}")
        elif NUM_SHORTS > 0:
            log("No VOD available, skipping AI shorts.")

        log("Done.")
    except Exception as e:
        log(f"Clipping failed: {e}")


def main():
    import track_estiie as tracker

    p = argparse.ArgumentParser(description="Clip a Twitch stream's VOD.")
    p.add_argument("login", nargs="?", default="estiie")
    p.add_argument("--vod", help="VOD id to process (default: latest archive)")
    args = p.parse_args()

    if not tracker.CLIENT_ID or not tracker.CLIENT_SECRET:
        sys.exit("Set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET first.")
    user = tracker.get_user(args.login.lower())
    process_stream(tracker.api_get, user["id"], user["login"], vod_id=args.vod)


if __name__ == "__main__":
    main()
