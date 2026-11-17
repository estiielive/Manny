"""Fetching media. Remote links (anything yt-dlp supports: Twitch VODs, Kick, YouTube...) are fetched
audio-only for analysis, then only the chosen clip sections are downloaded in full quality."""
import hashlib
import re
import subprocess
from pathlib import Path

from .util import fmt_time, log, run


def is_local(source):
    return Path(source).exists()


def source_key(source):
    if is_local(source):
        base = Path(source).stem
    else:
        try:
            base = run(["yt-dlp", "--no-warnings", "--print", "%(extractor)s-%(id)s", source]).strip().splitlines()[0]
        except Exception:
            base = source
    clean = re.sub(r"[^A-Za-z0-9_-]+", "-", base).strip("-")
    if not clean or len(clean) > 60:
        clean = "src-" + hashlib.sha1(base.encode()).hexdigest()[:12]
    return clean


def title_of(source):
    if is_local(source):
        return Path(source).stem
    try:
        return run(["yt-dlp", "--no-warnings", "--print", "%(title)s", source]).strip().splitlines()[0]
    except Exception:
        return source


def _sections_arg(start, end):
    return ["--download-sections", f"*{fmt_time(start)}-{fmt_time(end) if end else 'inf'}"]


def fetch_audio(source, workdir, section=None):
    """Mono 16 kHz WAV of the stream (or of `section`). Times in the WAV start at section[0]."""
    workdir = Path(workdir)
    wav = workdir / "audio.wav"
    if wav.exists():
        log("using cached audio", wav)
        return wav
    start, end = section or (0.0, None)
    if is_local(source):
        cmd = ["ffmpeg", "-y", "-v", "error"]
        if start:
            cmd += ["-ss", str(start)]
        if end:
            cmd += ["-to", str(end)]
        log("extracting audio")
        run(cmd + ["-i", source, "-vn", "-ac", "1", "-ar", "16000", str(wav)])
        return wav
    raw = workdir / "audio_raw.m4a"
    if not raw.exists():
        cmd = ["yt-dlp", "--no-warnings", "--no-playlist", "-f", "ba/Audio_Only/worst", "-o", str(raw)]
        if section:
            cmd += _sections_arg(start, end)
        log("downloading audio track", source)
        if subprocess.run(cmd + [source]).returncode != 0 or not raw.exists():
            raise RuntimeError("yt-dlp audio download failed")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    return wav


_INPUTS = {}


def stream_inputs(source, max_height=1080):
    """ffmpeg-readable input(s) for full-quality video: [file_or_url] or [video_url, audio_url].
    ffmpeg seeks accurately inside remote HLS/DASH streams, so we never download the whole VOD."""
    if is_local(source):
        return [str(source)]
    if source not in _INPUTS:
        out = run(["yt-dlp", "--no-warnings", "--no-playlist", "-g",
                   "-f", f"b[height<={max_height}][vcodec!=none][acodec!=none]/bv*[height<={max_height}]+ba/b", source])
        _INPUTS[source] = [u for u in out.strip().splitlines() if u.strip()]
    return _INPUTS[source]
