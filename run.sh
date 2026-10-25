#!/usr/bin/env bash
# VERITY//RADAR launcher (macOS / Linux) - no installs needed, only Python 3.8+
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Python 3.8+ is required"; exit 1; }
exec python3 server.py
