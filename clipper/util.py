import json
import re
import subprocess
import sys
from pathlib import Path


def log(*args):
    print("[clipper]", *args, file=sys.stderr, flush=True)


def run(cmd, **kw):
    """Run a command, raising with stderr on failure."""
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}\n{p.stderr[-3000:]}")
    return p.stdout


def parse_time(s):
    """'1:02:03.5', '62:03', '3723' -> seconds."""
    if s is None or s == "":
        return None
    if isinstance(s, (int, float)):
        return float(s)
    parts = [float(p) for p in str(s).strip().split(":")]
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t


def fmt_time(t):
    t = max(0.0, float(t))
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def slugify(s, n=40):
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:n].strip("-") or "clip"


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def probe(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
               "-of", "json", str(path)])
    d = json.loads(out)
    v = next((s for s in d["streams"] if s.get("codec_type") == "video"), {})
    return {"duration": float(d["format"]["duration"]), "width": v.get("width"), "height": v.get("height"),
            "has_audio": any(s.get("codec_type") == "audio" for s in d["streams"])}
