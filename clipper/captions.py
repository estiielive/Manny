"""Animated word-by-word captions (ASS subtitles, burned in by ffmpeg/libass)."""
import re

from .layout import OUT_H, OUT_W


def _ts(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _esc(text):
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def group_words(words, max_words=3, max_chars=16, max_gap=0.45):
    """Short punchy groups: break on punctuation, pauses, or length."""
    groups, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["s"] - cur[-1]["e"] > max_gap
                    or len(" ".join(x["w"] for x in cur + [w])) > max_chars):
            groups.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?,;:]$", w["w"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _hex(rgb):  # "#RRGGBB" -> ASS "&H00BBGGRR"
    rgb = rgb.lstrip("#")
    return f"&H00{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}".upper()


def build_ass(words, clip_start, clip_end, caption_y, title=None, font="Inter", highlight="#FFE14D",
              title_y=None, title_secs=3.5, uppercase=True):
    """ASS script for one clip. `words` are in VOD time; output times are relative to clip_start."""
    hl = _hex(highlight)
    title_y = title_y if title_y is not None else 150
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},92,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,7,4,5,60,60,0,1
Style: Title,{font},64,&H00111111,&H00111111,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,18,0,8,80,80,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    if title:
        t = _esc(title.upper() if uppercase else title)
        ev.append(f"Dialogue: 1,{_ts(0)},{_ts(title_secs)},Title,,0,0,0,,"
                  f"{{\\pos({OUT_W // 2},{title_y})\\fad(150,250)\\fscx85\\fscy85\\t(0,140,\\fscx100\\fscy100)}}{t}")

    inside = [w for w in words if w["e"] > clip_start and w["s"] < clip_end]
    groups = group_words(inside)
    for gi, g in enumerate(groups):
        nxt = groups[gi + 1][0]["s"] if gi + 1 < len(groups) else clip_end
        g_end = min(nxt, g[-1]["e"] + 0.5)  # linger briefly, never overlap the next group
        for wi, w in enumerate(g):
            s = w["s"] if wi else max(w["s"], clip_start)
            e = g[wi + 1]["s"] if wi + 1 < len(g) else g_end
            if e <= s:
                continue
            parts = []
            for k, x in enumerate(g):
                txt = _esc(x["w"].upper() if uppercase else x["w"])
                parts.append(f"{{\\c{hl}}}{txt}{{\\c&H00FFFFFF&}}" if k == wi else txt)
            pop = "\\fscx82\\fscy82\\t(0,90,\\fscx100\\fscy100)" if wi == 0 else ""
            ev.append(f"Dialogue: 0,{_ts(s - clip_start)},{_ts(e - clip_start)},Cap,,0,0,0,,"
                      f"{{\\pos({OUT_W // 2},{caption_y}){pop}}}" + " ".join(parts))
    return head + "\n".join(ev) + "\n"
