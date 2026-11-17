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
