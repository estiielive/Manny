"""Turn a 16:9 frame into 9:16.

Layouts:
  split - facecam on top, gameplay (center crop) below. Default when a facecam overlay is found.
  crop  - 9:16 crop following the face (camera-only / IRL / just-chatting streams).
  blur  - full frame in the middle over a blurred, zoomed copy. Fallback when there's no face.

Facecam detection: Haar face detection across many frames finds the consistently-placed face; the webcam
box border is where the static webcam picture ends and constantly-moving gameplay begins (per-pixel
temporal variance)."""
import numpy as np

from .util import log

OUT_W, OUT_H = 1080, 1920


def grab_frame(src, t, width=960):
    """One frame at time t (seconds) from a file or stream URL, as a BGR image (or None)."""
    import subprocess

    import cv2

    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", src, "-frames:v", "1",
                        "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    if p.returncode or not p.stdout:
        return None
    return cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)


def sample_frames(src, ranges, n=40):
    """~n frames spread across the given (start, end) time ranges of `src`."""
    from concurrent.futures import ThreadPoolExecutor

    per = max(2, n // max(1, len(ranges)))
    times = [s + (e - s) * (k + 0.5) / per for s, e in ranges for k in range(per)]
    with ThreadPoolExecutor(8) as ex:
        frames = list(ex.map(lambda t: grab_frame(src, t), times))
    return [f for f in frames if f is not None]


YUNET_URLS = [
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]


def _yunet_model():
    """Path to the YuNet face model (~230 KB), downloaded once into ~/.cache/clipper."""
    import urllib.request
    from pathlib import Path

    path = Path.home() / ".cache" / "clipper" / "face_detection_yunet_2023mar.onnx"
    if path.exists() and path.stat().st_size > 100_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for url in YUNET_URLS:
        try:
            urllib.request.urlretrieve(url, path)
            if path.stat().st_size > 100_000:
                return path
        except Exception:
            continue
    return None


def _faces(frames):
    """Face boxes per frame. YuNet (a small CNN) when available, Haar cascades otherwise."""
    import cv2

    H, W = frames[0].shape[:2]
    model = _yunet_model() if hasattr(cv2, "FaceDetectorYN") else None
    out = []
    if model:
        det = cv2.FaceDetectorYN.create(str(model), "", (W, H), score_threshold=0.6)
        for f in frames:
            _, found = det.detect(f)
            out.append([tuple(int(v) for v in b[:4]) for b in (found if found is not None else [])])
        return out
    log("layout: YuNet unavailable, using Haar cascades (less accurate)")
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    for f in frames:
        g = cv2.equalizeHist(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        out.append([tuple(map(int, b)) for b in casc.detectMultiScale(g, 1.1, 6, minSize=(28, 28))])
    return out


def _consistent_face(faces, n_frames, W):
    """Cluster face centers; return the median box of the biggest cluster if it shows up often enough."""
    boxes = [b for fs in faces for b in fs]
    if not boxes:
        return None
    best = []
    for x, y, w, h in boxes:
        cx, cy = x + w / 2, y + h / 2
        members = [b for b in boxes if abs(b[0] + b[2] / 2 - cx) < 0.04 * W and abs(b[1] + b[3] / 2 - cy) < 0.04 * W]
        if len(members) > len(best):
            best = members
    if len(best) < max(3, 0.25 * n_frames):
        return None
    return tuple(int(v) for v in np.median(np.array(best), axis=0))


def _scan(dyn, fixed_lo, fixed_hi, start, stop, step, axis, run=3, thr=0.55):
    """Walk from `start` toward `stop`; return the first line where the frame becomes 'dynamic' (gameplay)."""
    hits = 0
    for i in range(start, stop, step):
        line = dyn[fixed_lo:fixed_hi, i] if axis == "x" else dyn[i, fixed_lo:fixed_hi]
        hits = hits + 1 if line.mean() > thr else 0
        if hits >= run:
            return i - step * (run - 1)
    return stop - step if step > 0 else 0


def detect(frames):
    """Return {"layout": ..., "facecam": [x, y, w, h] | None, "face": [...] | None} in fractions of the frame."""
    import cv2

    if len(frames) < 4:
        return {"layout": "blur", "facecam": None, "face": None}
    H, W = frames[0].shape[:2]
    face = _consistent_face(_faces(frames), len(frames), W)
    if face is None:
        log("layout: no consistent face found -> blur")
        return {"layout": "blur", "facecam": None, "face": None}
    fx, fy, fw, fh = face
    frac = lambda x, y, w, h: [round(x / W, 4), round(y / H, 4), round(w / W, 4), round(h / H, 4)]
    if fw > 0.13 * W:  # big face: it's a camera shot, not an overlay
        log("layout: full-camera stream -> crop on face")
        return {"layout": "crop", "facecam": None, "face": frac(*face)}

    gray = np.stack([cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for f in frames]).astype(np.float32)
    std = gray.std(axis=0)
    dyn = std > max(10.0, 0.5 * np.median(std))  # gameplay dominates the frame, so its median is "moving"
    m = int(0.6 * fw)  # skip the moving head itself
    rows = (fy, fy + fh)
    left = _scan(dyn, *rows, fx - m, -1, -1, "x")
    right = _scan(dyn, *rows, fx + fw + m, W, 1, "x")
    x0, x1 = (0 if left <= 2 else left + 1), (W if right >= W - 3 else right)
    # The person moves too, so for top/bottom look at whole rows of the box: the webcam background keeps
    # most of a row static, gameplay makes almost all of it move.
    cols = (x0 + 2, max(x0 + 4, x1 - 2))
    top = _scan(dyn, *cols, fy, -1, -1, "y", thr=0.6)
    bottom = _scan(dyn, *cols, fy + fh, H, 1, "y", thr=0.6)
    y0, y1 = (0 if top <= 2 else top + 1), (H if bottom >= H - 3 else bottom)
    w, h = x1 - x0, y1 - y0
    # Sanity: a webcam box should be a few faces wide and roughly landscape/square; else fall back to a fixed box.
    if not (2.2 * fw <= w <= 0.6 * W and 1.6 * fh <= h <= 0.7 * H and 0.8 <= w / h <= 2.6):
        log(f"layout: facecam border unclear ({w}x{h}), using a box around the face")
        w, h = int(fw * 3.6), int(fw * 3.6 * 9 / 16)
        x0 = int(np.clip(fx + fw / 2 - w / 2, 0, W - w))
        y0 = int(np.clip(fy + fh * 0.55 - h * 0.45, 0, H - h))
    log("layout: facecam found -> split")
    return {"layout": "split", "facecam": frac(x0, y0, w, h), "face": frac(*face)}


def save_preview(frame, info, path):
    import cv2

    f = frame.copy()
    H, W = f.shape[:2]
    for key, color in (("facecam", (0, 255, 0)), ("face", (0, 128, 255))):
        if info.get(key):
            x, y, w, h = info[key]
            cv2.rectangle(f, (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H)), color, 3)
    cv2.imwrite(str(path), f)


def _even(v):
    return int(v) // 2 * 2


def filtergraph(info, src_w, src_h, zoom=1.0):
    """ffmpeg filter_complex producing [v] at 1080x1920 from [0:v]. Also returns the caption anchor y (px)."""
    lay = info["layout"]
    if lay == "split":
        x, y, w, h = info["facecam"]
        cx, cy, cw, ch = x * src_w, y * src_h, w * src_w, h * src_h
        cam_h = _even(np.clip(OUT_W * ch / cw, 560, 800))  # odd sizes crash some ffmpeg builds
        # Re-crop the cam box to the output aspect, keeping it centered on the cam box.
        target_ratio = OUT_W / cam_h
        if cw / ch > target_ratio:
            nw = ch * target_ratio
            cx, cw = cx + (cw - nw) / 2, nw
        else:
            nh = cw / target_ratio
            cy, ch = cy + (ch - nh) / 2, nh
        game_h = OUT_H - cam_h
        gw = src_h * OUT_W / game_h / zoom
        gh = src_h / zoom
        g = (f"[0:v]split=2[c][g];"
             f"[c]crop={_even(cw)}:{_even(ch)}:{_even(cx)}:{_even(cy)},scale={OUT_W}:{cam_h},setsar=1[cam];"
             f"[g]crop={_even(gw)}:{_even(gh)}:{_even((src_w - gw) / 2)}:{_even((src_h - gh) / 2)},"
             f"scale={OUT_W}:{game_h},setsar=1[game];"
             f"[cam][game]vstack=inputs=2[v]")
        return g, cam_h
    if lay == "crop":
        cw = src_h * 9 / 16 / zoom
        ch = src_h / zoom
        fx = (info["face"][0] + info["face"][2] / 2) * src_w if info.get("face") else src_w / 2
        x = np.clip(fx - cw / 2, 0, src_w - cw)
        return (f"[0:v]crop={_even(cw)}:{_even(ch)}:{_even(x)}:{_even((src_h - ch) / 2)},"
                f"scale={OUT_W}:{OUT_H},setsar=1[v]"), int(OUT_H * 0.70)
    # blur
    fg_w = OUT_W
    vis_w = src_w / max(zoom, 1.0)
    fg_h = _even(OUT_W * src_h / vis_w)
    g = (f"[0:v]split=2[b][f];"
         f"[b]scale=-2:{OUT_H // 4},crop={OUT_W // 4}:{OUT_H // 4},boxblur=12:2,scale={OUT_W}:{OUT_H},"
         f"eq=brightness=-0.12,setsar=1[bg];"
         f"[f]crop={_even(vis_w)}:{src_h}:{_even((src_w - vis_w) / 2)}:0,scale={fg_w}:{fg_h},setsar=1[fg];"
         f"[bg][fg]overlay=0:(H-h)/2[v]")
    return g, int(OUT_H / 2 + fg_h / 2 + 170)
