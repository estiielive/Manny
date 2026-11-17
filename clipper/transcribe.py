"""Speech-to-text with word timestamps.

Backends:
  openai - OpenAI Whisper API (fast, needs OPENAI_API_KEY with credits); audio is chunked so any length works.
  local  - faster-whisper on this machine (free; uses the GPU if there is one, else CPU)."""
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .util import load_json, log, probe, run, save_json

CHUNK_S = 600  # 10 min of 32 kbps mono mp3 ≈ 2.4 MB, well under the 25 MB upload limit


def _transcribe_chunk(client, wav, start, dur, cache, language):
    if cache.exists():
        return load_json(cache)
    mp3 = cache.with_suffix(".mp3")
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(dur), "-i", str(wav),
         "-ac", "1", "-ar", "16000", "-b:a", "32k", str(mp3)])
    kw = {"language": language} if language else {}
    for attempt in range(3):
        try:
            with open(mp3, "rb") as f:
                r = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="verbose_json",
                                                       timestamp_granularities=["word", "segment"], **kw)
            break
        except Exception as e:  # transient API errors
            if attempt == 2:
                raise
            log(f"transcription retry ({e})")
    d = r.model_dump() if hasattr(r, "model_dump") else dict(r)
    out = {
        "words": [{"w": w["word"].strip(), "s": round(w["start"] + start, 2), "e": round(w["end"] + start, 2)}
                  for w in d.get("words") or [] if w["word"].strip()],
        "segments": [{"s": round(s["start"] + start, 2), "e": round(s["end"] + start, 2), "text": s["text"].strip(),
                      "no_speech": s.get("no_speech_prob", 0)}
                     for s in d.get("segments") or []],
    }
    save_json(cache, out)
    mp3.unlink(missing_ok=True)
    return out


def transcribe_local(wav, language=None, model_name=None):
    from faster_whisper import WhisperModel

    try:
        import ctranslate2
        cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda = False
    model_name = model_name or ("large-v3-turbo" if cuda else "small")
    log(f"transcribing locally with faster-whisper '{model_name}' on {'GPU' if cuda else 'CPU'} "
        f"(first run downloads the model)")
    model = WhisperModel(model_name, device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")
    import numpy as np

    total = probe(wav)["duration"]
    out = {"words": [], "segments": []}
    step = 1200  # decode + transcribe 20 min at a time to keep memory flat on long VODs
    for start in np.arange(0, total, step):
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(step), "-i", str(wav),
                              "-f", "s16le", "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        if len(audio) < 1600:
            continue
        segments, info = model.transcribe(audio, language=language, word_timestamps=True, vad_filter=True,
                                          vad_parameters={"min_silence_duration_ms": 700},
                                          language_detection_segments=6)
        if not language and info.language_probability > 0.6:
            language = info.language  # lock it in: game noise in later chunks can fool per-chunk detection
            log(f"  language: {language}")
        for seg in segments:
            out["segments"].append({"s": round(seg.start + start, 2), "e": round(seg.end + start, 2),
                                    "text": seg.text.strip(), "no_speech": seg.no_speech_prob})
            out["words"] += [{"w": w.word.strip(), "s": round(w.start + start, 2), "e": round(w.end + start, 2)}
                             for w in seg.words or [] if w.word.strip()]
        log(f"  transcribed {min(1.0, (start + step) / total):.0%}")
    out["segments"] = [s for s in out["segments"] if s["no_speech"] < 0.8 and s["text"]]
    return out


def transcribe(wav, workdir, language=None, backend="auto", model_name=None, workers=4):
    """Returns {"words": [{w, s, e}], "segments": [{s, e, text}]} with times relative to the WAV start."""
    import os

    workdir = Path(workdir)
    final = workdir / "transcript.json"
    if final.exists():
        log("using cached transcript")
        return load_json(final)
    if backend == "auto":
        backend = "openai" if os.environ.get("OPENAI_API_KEY") else "local"
    if backend == "local":
        out = transcribe_local(wav, language, model_name)
        save_json(final, out)
        return out
    from openai import OpenAI

    cdir = workdir / "transcript_chunks"
    cdir.mkdir(exist_ok=True)
    total = probe(wav)["duration"]
    starts = [i * CHUNK_S for i in range(int(total // CHUNK_S) + 1) if i * CHUNK_S < total - 0.5]
    log(f"transcribing {total / 60:.0f} min of audio in {len(starts)} chunk(s)")
    client = OpenAI()
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(lambda s: _transcribe_chunk(client, wav, s, min(CHUNK_S, total - s),
                                                        cdir / f"{int(s):06d}.json", language), starts))
    out = {"words": [w for p in parts for w in p["words"]],
           "segments": [s for p in parts for s in p["segments"] if s["no_speech"] < 0.8 and s["text"]]}
    save_json(final, out)
    return out
