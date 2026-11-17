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
