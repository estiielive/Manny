# Manny

## Clipper: stream link in, finished vertical clips out

Give it a VOD link. It finds the best moments and renders 1080x1920 clips with word-by-word captions and a hook title, ready for TikTok, Shorts and Reels.

```sh
python -m clipper https://www.twitch.tv/videos/2923489389
```

### How it works

1. **Audio first.** For a remote link it downloads only the audio track (about 70 MB per hour instead of about 2.5 GB of 1080p60 video).
2. **Transcribe** with word-level timestamps. It uses the OpenAI Whisper API when `OPENAI_API_KEY` is set. Otherwise it runs faster-whisper locally for free: about 2 minutes for a 46-minute VOD on a 4-core CPU.
3. **Find moments.**
   - With an API key, an LLM (default `gpt-5.5`) acts as a short-form editor. It reads the timestamped transcript, which is tagged where the audio gets unusually loud (`LOUD+n`) or chat spikes (`CHAT+n`). It picks moments that stand alone, with a hook, a setup and a payoff. Each pick gets a title, a post caption, a score and a reason.
   - Without a key, it picks from signal peaks instead: loudness spikes, bursts of fast speech and reaction phrases ("no way", "let's go", laughing, swearing).
4. **Snap cut points** to natural speech boundaries, so clips never start or end mid-word.
5. **Pick a layout automatically:**
   - `split`: when there is a facecam overlay, the webcam goes on top and the gameplay (center crop) below, with captions on the seam. It finds the face with a small neural face detector, then finds the edge of the webcam box: the webcam picture stays mostly still while gameplay keeps moving. Check `layout_preview.jpg` to see what it found.
   - `crop`: a 9:16 crop that follows the face, for camera-only and IRL streams.
   - `blur`: the full frame over a blurred copy of itself. This is the fallback.
6. **Render** with ffmpeg straight from the stream URL, so only the clip ranges are fetched. Captions are burned in (bold, uppercase, current word highlighted), and audio is normalized to -14 LUFS.

### Options

```sh
python -m clipper URL -n 8 --min 20 --max 45       # 8 clips, 20-45 s each
python -m clipper URL --section 1:00:00-2:30:00    # only look at part of a long VOD
python -m clipper URL --chat chat.json             # chat replay boosts detection (see below)
python -m clipper URL --dry-run                    # just list the moments
python -m clipper URL --layout blur                # force a layout (auto|split|crop|blur)
python -m clipper URL --facecam 0,0.444,0.24,0.228 # set the facecam box yourself (x,y,w,h as fractions)
python -m clipper URL --zoom 1.2                   # zoom the gameplay in more
python -m clipper URL --no-llm                     # signal-based picking only
python -m clipper my_recording.mp4                 # local files work too
```

The output goes to `clips/<video-id>/`:

- the `.mp4` clips
- `README.md`, with each clip's title, caption, score and why it was picked
- `clips.json`
- `layout_preview.jpg`

To tweak a clip, edit `start`, `end` or `title` in `clips.json`, then run:

```sh
python -m clipper --rerender clips/<video-id>/clips.json
```

Downloads and transcripts are cached in `.clipper_cache/`, so running again on the same VOD with different settings is fast.

**Chat replay (optional).** Chat is one of the best hype signals. Export it with [TwitchDownloader](https://github.com/lay295/TwitchDownloader) (`TwitchDownloaderCLI chatdownload --id <vod id> -o chat.json`) and pass `--chat chat.json`. A CSV of `seconds,message` lines also works.

**Sources.** Anything yt-dlp supports: Twitch VODs and highlights, Kick, YouTube, and others. Clipping a stream that is still live works once the VOD is published.

### Setup

Requires `ffmpeg` (with libass, which standard builds include) and Python 3.10+.

```sh
pip install -r requirements.txt
export OPENAI_API_KEY=...   # optional; without it everything runs locally
```

To use another OpenAI model for picking, pass `--model` or set `CLIPPER_MODEL`. Run the tests with `python -m pytest tests`.

---

# Twitch channel tracker

Polls the Twitch Helix API every 60 seconds for a channel (default: `estiie`) and logs:

- going live / offline (with the peak viewer count for each stream)
- title and game changes
- viewer counts

Every poll is also written as a row in `<login>_log.csv` (`utc_time, live, viewers, game, title`).

## Setup

1. Register an app at https://dev.twitch.tv/console/apps.
2. `pip install -r requirements.txt`
3. Set your credentials:
   ```sh
   export TWITCH_CLIENT_ID=...
   export TWITCH_CLIENT_SECRET=...
   ```
4. Run it:
   ```sh
   python track_estiie.py              # tracks estiie
   python track_estiie.py otherlogin   # tracks another channel
   ```

Press Ctrl+C to stop.
