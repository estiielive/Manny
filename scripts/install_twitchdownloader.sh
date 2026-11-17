#!/usr/bin/env bash
# Download the latest TwitchDownloaderCLI release (github.com/lay295/TwitchDownloader)
# into ./bin/ for this OS/CPU. Needs curl, unzip and python3.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p bin

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)   platform="Linux-x64" ;;
  Linux-aarch64)  platform="LinuxArm64" ;;
  Linux-armv7l)   platform="LinuxArm" ;;
  Darwin-x86_64)  platform="MacOS-x64" ;;
  Darwin-arm64)   platform="MacOSArm64" ;;
  MINGW*|MSYS*|CYGWIN*) platform="Windows-x64" ;;
  *) echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

url=$(curl -fsSL https://api.github.com/repos/lay295/TwitchDownloader/releases/latest |
  python3 -c "import json,sys; d=json.load(sys.stdin); print(next(a['browser_download_url'] for a in d['assets'] if a['name'].startswith('TwitchDownloaderCLI-') and a['name'].endswith('-$platform.zip')))")

echo "Downloading $url"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "$url" -o "$tmp/cli.zip"
unzip -oq "$tmp/cli.zip" -d bin
chmod +x bin/TwitchDownloaderCLI 2>/dev/null || true
echo "Installed: $(ls bin/TwitchDownloaderCLI*)"
