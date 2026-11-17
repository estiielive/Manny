"""clipper: give it a stream link, get finished vertical clips with captions.

  python -m clipper https://www.twitch.tv/videos/123456789
  python -m clipper URL --clips 8 --min 20 --max 50 --section 1:00:00-2:30:00
  python -m clipper URL --chat chat.json          # optional chat replay boosts moment detection
  python -m clipper --rerender out/v123/clips.json  # after hand-editing times/titles in the manifest
"""
import argparse
import json
import os
import sys
from pathlib import Path


from . import download, highlights, layout, signals
from .render import render_clip
from .transcribe import transcribe
from .util import fmt_time, load_json, log, parse_time, probe, run, save_json, slugify


def parse_section(s):
    if not s:
        return None
    a, _, b = s.partition("-")
    return (parse_time(a) or 0.0, parse_time(b))


def parse_box(s):
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("expected x,y,w,h")
    return v


def get_args(argv=None):
    p = argparse.ArgumentParser(prog="clipper", description="Stream link in, captioned vertical clips out.")
    p.add_argument("source", nargs="?", help="VOD/stream URL (Twitch, Kick, YouTube... anything yt-dlp supports) or a local video file")
    p.add_argument("-n", "--clips", type=int, default=5, help="number of clips to make (default 5)")
    p.add_argument("--min", dest="min_len", type=float, default=15, help="min clip length, seconds (default 15)")
    p.add_argument("--max", dest="max_len", type=float, default=60, help="max clip length, seconds (default 60)")
    p.add_argument("--section", help="only analyse part of the VOD, e.g. 1:00:00-2:30:00")
    p.add_argument("--chat", help="chat replay file (TwitchDownloader JSON, or CSV `seconds,message`) to boost detection")
    p.add_argument("--layout", choices=["auto", "split", "blur", "crop"], default="auto")
    p.add_argument("--facecam", type=parse_box, help="facecam box as x,y,w,h fractions of the frame (overrides detection)")
    p.add_argument("--zoom", type=float, default=1.0, help="extra zoom on the gameplay / main view (e.g. 1.2)")
    p.add_argument("--model", default=os.environ.get("CLIPPER_MODEL", "gpt-5.5"), help="LLM that picks moments")
    p.add_argument("--no-llm", action="store_true", help="pick moments from loudness/chat spikes only (no API calls for picking)")
    p.add_argument("--transcriber", choices=["auto", "openai", "local"], default="auto",
                   help="openai = Whisper API; local = faster-whisper on this machine (free). auto = openai if OPENAI_API_KEY is set")
    p.add_argument("--whisper-model", help="local whisper model (default: small on CPU, large-v3-turbo on GPU)")
    p.add_argument("--language", help="spoken language code for transcription, e.g. en (auto-detect by default)")
    p.add_argument("--font", default="Inter", help="caption font family (must be installed)")
    p.add_argument("--no-title", action="store_true", help="don't overlay the hook title at the start")
    p.add_argument("--out", default="clips", help="output folder (default ./clips)")
    p.add_argument("--work", default=".clipper_cache", help="cache folder for downloads/transcripts")
    p.add_argument("--dry-run", action="store_true", help="find and list moments, don't render")
    p.add_argument("--rerender", metavar="MANIFEST", help="re-render clips from an (edited) clips.json")
    a = p.parse_args(argv)
    if not a.source and not a.rerender:
        p.error("give a stream link / video file (or --rerender clips.json)")
    return a


def find_moments(a, workdir):
    section = parse_section(a.section)
    offset = section[0] if section else 0.0
    wav = download.fetch_audio(a.source, workdir, section)
    total = probe(wav)["duration"]
    log(f"audio: {fmt_time(total)} starting at {fmt_time(offset)}")

    tr = transcribe(wav, workdir, a.language, a.transcriber, a.whisper_model)
    words = [{**w, "s": w["s"] + offset, "e": w["e"] + offset} for w in tr["words"]]
    segs = [{**s, "s": s["s"] + offset, "e": s["e"] + offset} for s in tr["segments"]]
    transcript = {"words": words, "segments": segs}
    log(f"transcript: {len(words)} words")

    energy = signals.energy_curve(wav)
    chat = None
    if a.chat:
        msgs = signals.load_chat(a.chat)
        log(f"chat: {len(msgs)} messages")
        chat = signals.chat_curve(msgs, len(energy), offset)

    if a.no_llm:
        clips = highlights.pick_with_signals(transcript, energy, chat, offset, a.clips, a.min_len, a.max_len)
    else:
        context = download.title_of(a.source)
        clips = highlights.pick_with_llm(transcript, energy, chat, offset, a.clips, a.min_len, a.max_len,
                                         a.model, context)
    end = offset + total
    clips = [highlights.refine(c, words, a.min_len, a.max_len, end) for c in clips]
    clips.sort(key=lambda c: -c.get("score", 0))
    return clips, words


def detect_layout(a, src, ranges):
    if a.facecam:
        return {"layout": "split", "facecam": a.facecam, "face": None}
    if a.layout == "blur":
        return {"layout": "blur", "facecam": None, "face": None}
    frames = layout.sample_frames(src, ranges)
    info = layout.detect(frames)
    if a.layout == "crop":
        info["layout"] = "crop"
    if a.layout == "split" and info["layout"] != "split":
        log("no facecam found for --layout split; pass --facecam x,y,w,h. Falling back to blur.")
        info["layout"] = "blur"
    if frames:
        info["_preview_frame"] = frames[len(frames) // 2]
    return info


def render_all(a, clips, words, source, outdir, layout_info=None):
    outdir.mkdir(parents=True, exist_ok=True)
    inputs = download.stream_inputs(source)
    size = video_size(inputs[0])
    # One layout decision for the whole stream, from frames spread across all the clips.
    if layout_info is None:
        layout_info = detect_layout(a, inputs[0], [(c["start"], c["end"]) for c in clips])
        frame = layout_info.pop("_preview_frame", None)
        if frame is not None:
            layout.save_preview(frame, layout_info, outdir / "layout_preview.jpg")
    log(f"layout: {layout_info['layout']}" + (f" facecam={layout_info['facecam']}" if layout_info.get("facecam") else ""))
    for i, c in enumerate(clips, 1):
        name = f"{i:02d}_{slugify(c.get('title') or 'clip')}.mp4"
        log(f"rendering {i}/{len(clips)}: {name} ({c['end'] - c['start']:.0f}s)")
        render_clip(inputs, size, c, words, layout_info, outdir / name, font=a.font, zoom=a.zoom,
                    title=not a.no_title)
        c["file"] = name
    return layout_info


def video_size(src):
    st = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                         "-of", "json", src]))["streams"][0]
    return st["width"], st["height"]


def write_summary(outdir, clips, source):
    lines = [f"# Clips from {source}\n"]
    for c in clips:
        lines.append(f"## {c.get('file', '')}\n**{c.get('title', '')}** — score {c.get('score', 0):.0f}, "
                     f"{c.get('category', '')}, {c['end'] - c['start']:.0f}s, VOD {fmt_time(c['start'])}–{fmt_time(c['end'])}\n\n"
                     f"Why: {c.get('reason', '')}\n\nCaption: {c.get('caption', '')}\n")
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    a = get_args(argv)
    if not a.rerender and not a.no_llm and not os.environ.get("OPENAI_API_KEY"):
        log("OPENAI_API_KEY not set -> local transcription + signal-based picking (--no-llm)")
        a.no_llm = True

    if a.rerender:
        man = load_json(a.rerender)
        a.source = a.source or man["source"]
        outdir = Path(a.rerender).parent
        words = load_json(outdir / "words.json") or load_json(Path(a.work) / man["key"] / "words.json")
        if words is None:
            sys.exit("words.json (transcript) not found next to the manifest")
        info = None if (a.facecam or a.layout != "auto") else man.get("layout")
        info = render_all(a, man["clips"], words, a.source, outdir, info)
        man["layout"] = info
        save_json(Path(a.rerender), man)
        write_summary(outdir, man["clips"], a.source)
        log(f"done -> {outdir}")
        return

    key = download.source_key(a.source)
    if a.section:
        sec = parse_section(a.section)
        key += f"_{int(sec[0])}-{int(sec[1]) if sec[1] else 'end'}"
    workdir = Path(a.work) / key
    workdir.mkdir(parents=True, exist_ok=True)
    outdir = Path(a.out) / key

    clips, words = find_moments(a, workdir)
    save_json(workdir / "words.json", words)
    if not clips:
        sys.exit("no clip-worthy moments found")
    print(f"\nTop {len(clips)} moments:")
    for i, c in enumerate(clips, 1):
        print(f"  {i}. [{fmt_time(c['start'])} - {fmt_time(c['end'])}] ({c['end'] - c['start']:.0f}s) "
              f"score {c.get('score', 0):.0f}  {c.get('title', '')}\n     {c.get('reason', '')}")
    if a.dry_run:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    info = render_all(a, clips, words, a.source, outdir)
    save_json(outdir / "clips.json", {"source": a.source, "key": key, "layout": info, "clips": clips})
    save_json(outdir / "words.json", [w for w in words if any(c["start"] - 5 <= w["s"] <= c["end"] + 5 for c in clips)])
    write_summary(outdir, clips, a.source)
    print(f"\nDone: {len(clips)} clips in {outdir}/  (edit clips.json and run --rerender to tweak)")


if __name__ == "__main__":
    main()
