"""ffmpeg render: reframe to 9:16, burn captions, normalize loudness."""
from pathlib import Path

from .captions import build_ass
from .layout import filtergraph
from .util import run


def render_clip(inputs, src_size, clip, words, layout_info, out_path, font="Inter", zoom=1.0, title=True,
                crf=20, preset="medium"):
    """`inputs`: [video(+audio)] or [video, audio] files/URLs in VOD time. clip["start"/"end"] are VOD times."""
    out_path = Path(out_path)
    graph, cap_y = filtergraph(layout_info, src_size[0], src_size[1], zoom)
    title_y = 120 if layout_info["layout"] != "blur" else 300
    ass = out_path.with_suffix(".ass")
    ass.write_text(build_ass(words, clip["start"], clip["end"], cap_y, clip.get("title") if title else None,
                             font=font, title_y=title_y), encoding="utf-8")
    ss = max(0.0, clip["start"])
    dur = clip["end"] - clip["start"]
    ass_arg = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    vf = f"{graph};[v]ass='{ass_arg}',fps=30,format=yuv420p[vo]"
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for src in inputs:
        cmd += ["-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", str(src)]
    afx = f"loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=in:d=0.08,afade=t=out:st={max(0, dur - 0.25):.3f}:d=0.25"
    cmd += ["-filter_complex", vf, "-map", "[vo]", "-map", f"{len(inputs) - 1}:a:0?", "-af", afx,
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high",
            "-movflags", "+faststart", str(out_path)]
    run(cmd)
    return out_path
