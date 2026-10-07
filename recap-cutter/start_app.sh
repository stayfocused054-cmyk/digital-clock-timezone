#!/usr/bin/env bash
# Mac / Linux: ./start_app.sh
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
command -v ffmpeg >/dev/null || echo "[!] ffmpeg nahi mila (Mac: brew install ffmpeg, Linux: sudo apt install ffmpeg)"
if [ ! -f .installed ]; then
  echo "Pehli baar: packages install ho rahe hain..."
  "$PY" -m pip install -r requirements-app.txt
  "$PY" -m pip install -r requirements-ai.txt || echo "[!] AI packages install nahi hue - basic mode me chalega."
  touch .installed
fi
echo "App khul raha hai: http://127.0.0.1:7860"
exec "$PY" -m recapcut app
