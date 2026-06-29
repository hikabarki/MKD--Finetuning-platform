#!/bin/bash
cd "$(dirname "$0")"
PYTHONNOUSERSITE=1 PYTHONPATH="" FT_PROJECT="$(pwd)" venv/bin/python app.py --root-path /proxy/7861 "$@"
