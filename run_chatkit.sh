#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "$PROJECT_ROOT"
exec .venv/bin/uvicorn chatkit_app.app:app --host 0.0.0.0 --port 8000 --reload
