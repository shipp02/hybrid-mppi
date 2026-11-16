#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
export ACADOS_SOURCE_DIR="$SETUP_DIR/acados"
export DYLD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$SETUP_DIR/.venv-acados/bin/python" "$PROJECT_DIR/tools/acados_tracking_demo.py" "$@"
