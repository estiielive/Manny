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

## Auto-clipping

The bot can turn each stream into clips using two open-source projects:

| Tool | What it does here |
|---|---|
| [lay295/TwitchDownloader](https://github.com/lay295/TwitchDownloader) (MIT) | Downloads the stream's VOD and the most-viewed clips viewers made during it |
| [Anil-matcha/AI-Youtube-Shorts-Generator](https://github.com/Anil-matcha/AI-Youtube-Shorts-Generator) (MIT) | Transcribes the VOD, has an LLM rank the most viral moments, and renders them as vertical 9:16 shorts with face tracking |

When a stream ends, results are written to:

```
clips/<login>/<vod_id>/
  twitch_clips/   top viewer clips, named by view count
  vod_<id>.mp4    the VOD (local mode)
  shorts/         short_01.mp4, short_02.mp4, ... + shorts.json (scores, hooks, timestamps)
```

### Setup

1. Get the shorts generator (git submodule):
   ```sh
   git submodule update --init
   ```
2. Install the Python packages and `ffmpeg` (e.g. `apt install ffmpeg` / `brew install ffmpeg`):
   ```sh
   pip install -r requirements-clip.txt
   ```
3. Install TwitchDownloaderCLI into `bin/`:
   ```sh
   ./scripts/install_twitchdownloader.sh
   ```
   (On Windows, download `TwitchDownloaderCLI-*-Windows-x64.zip` from its releases page into `bin/`, or set `TWITCHDOWNLOADER_CLI` to its path.)
4. Give the shorts generator an LLM key for picking highlights:
   ```sh
   export OPENAI_API_KEY=...          # or: export LLM_PROVIDER=gemini GEMINI_API_KEY=...
   ```
   With Gemini, also set `GEMINI_MODEL` (e.g. `gemini-3-flash-preview`): the generator's default,
   `gemini-2.5-flash`, is no longer available to new API keys.
5. The channel needs **past broadcasts (VODs) enabled** for shorts.

### Running it

```sh
python track_estiie.py --clip     # track, and clip every stream when it ends
python clipper.py                 # clip estiie's latest VOD right now
python clipper.py otherlogin --vod 2312345678
```

### Options (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `CLIP_TOP_N` | `5` | Viewer clips to download (`0` = skip) |
| `CLIP_NUM_SHORTS` | `3` | AI shorts to render (`0` = skip) |
| `CLIP_VOD_MAX_MINUTES` | `0` | Only use the first N minutes of the VOD (`0` = whole VOD). Whisper on a multi-hour VOD is slow on CPU; set `LOCAL_WHISPER_DEVICE=cuda` if you have a GPU |
| `CLIP_RANKING` | `ai` | `ai` (an LLM ranks the transcript) or `loudness` (the loudest audio peaks, skipping muted sections; no API key or transcription needed) |
| `CLIP_SHORTS_MODE` | `local` | `local` (runs on your machine) or `api` (hosted MuAPI, needs `MUAPI_API_KEY`) |
| `CLIP_ASPECT_RATIO` | `9:16` | Shorts aspect ratio, e.g. `1:1` |
| `CLIP_OUTPUT_DIR` | `clips` | Where results go |
| `TWITCHDOWNLOADER_CLI` | `bin/TwitchDownloaderCLI` | Path to the CLI |

The shorts generator's own settings (`LOCAL_WHISPER_MODEL`, `OPENAI_MODEL`, ...) work too; see its README.
