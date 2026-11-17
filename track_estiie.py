#!/usr/bin/env python3
"""
Track a Twitch channel (default: estiie) via the Twitch Helix API.

Logs: live/offline transitions, title changes, game changes, viewer counts.
Writes every poll to a CSV so you can graph it later.

Setup:
  1. Register an app at https://dev.twitch.tv/console/apps
  2. pip install requests
  3. export TWITCH_CLIENT_ID=...   export TWITCH_CLIENT_SECRET=...
  4. python track_estiie.py            (or: python track_estiie.py otherlogin)
"""
import csv
import os
import sys
import time
from datetime import datetime, timezone

import requests

CLIENT_ID = os.environ.get("TWITCH_CLIENT_ID")
CLIENT_SECRET = os.environ.get("TWITCH_CLIENT_SECRET")
LOGIN = (sys.argv[1] if len(sys.argv) > 1 else "estiie").lower()
POLL_SECONDS = 60
CSV_PATH = f"{LOGIN}_log.csv"

API = "https://api.twitch.tv/helix"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"

_token = None


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_token():
    """App access token (client credentials flow)."""
    global _token
    r = requests.post(
        TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "client_credentials",
        },
        timeout=15,
    )
    r.raise_for_status()
    _token = r.json()["access_token"]
    return _token


def rate_limit_wait(headers):
    """Seconds until the rate-limit bucket refills.

    Twitch's Ratelimit-Reset header is a Unix epoch timestamp, not a duration.
    """
    try:
        reset_at = int(headers.get("Ratelimit-Reset", 0))
    except ValueError:
        return 5
    return min(max(reset_at - int(time.time()), 1), 60)


def api_get(path, params):
    """GET with automatic token refresh on 401 and basic rate-limit handling."""
    if not _token:
        get_token()
    for _ in range(3):
        r = requests.get(
            f"{API}/{path}",
            params=params,
            headers={"Client-Id": CLIENT_ID, "Authorization": f"Bearer {_token}"},
            timeout=15,
        )
        if r.status_code == 401:
            get_token()
            continue
        if r.status_code == 429:
            time.sleep(rate_limit_wait(r.headers))
            continue
        r.raise_for_status()
        return r.json().get("data", [])
    raise RuntimeError(f"Failed calling {path}")


def get_user(login):
    data = api_get("users", {"login": login})
    if not data:
        sys.exit(f"No Twitch user found with login '{login}'.")
    return data[0]


def get_stream(user_id):
    data = api_get("streams", {"user_id": user_id})
    return data[0] if data else None


def get_channel(user_id):
    data = api_get("channels", {"broadcaster_id": user_id})
    return data[0] if data else {}


def log_row(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc_time", "live", "viewers", "game", "title"])
        w.writerow(row)


def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        sys.exit("Set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET first.")

    user = get_user(LOGIN)
    uid = user["id"]
    print(f"[{now()}] Tracking {user['display_name']} (id {uid})")
    print(f"  Account type: {user.get('broadcaster_type') or 'none'} | "
          f"Created: {user['created_at'][:10]}")

    was_live = None
    last_title = None
    last_game = None
    peak = 0

    try:
        while True:
            try:
                stream = get_stream(uid)
                channel = get_channel(uid)
                live = stream is not None
                title = (stream or channel).get("title", "")
                game = (stream or channel).get("game_name", "")
                viewers = stream["viewer_count"] if stream else 0

                if was_live is not None and live != was_live:
                    if live:
                        peak = 0
                        print(f"[{now()}] 🔴 WENT LIVE: {title} ({game})")
                    else:
                        print(f"[{now()}] ⚫ WENT OFFLINE (peak this stream: {peak})")
                if last_title is not None and title != last_title:
                    print(f"[{now()}] Title changed: {title}")
                if last_game is not None and game != last_game:
                    print(f"[{now()}] Game changed: {game}")

                if live:
                    peak = max(peak, viewers)
                    print(f"[{now()}] {viewers} viewers | {game}")

                log_row([now(), int(live), viewers, game, title])
                was_live, last_title, last_game = live, title, game

            except (requests.RequestException, RuntimeError) as e:
                print(f"[{now()}] Network/API error: {e}")

            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
