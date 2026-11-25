#!/usr/bin/env bash
# Copy recorded car runs from the Jetson to this laptop, then delete them on the Jetson
# (run this on the laptop). Runs are recorded by launch/car_launch.py.
#
#   src/hybrid-mppi/tools/fetch_car_run.sh            # every finished run
#   src/hybrid-mppi/tools/fetch_car_run.sh <run>...   # only these run folders
#   src/hybrid-mppi/tools/fetch_car_run.sh --keep     # copy, but keep them on the Jetson
#   src/hybrid-mppi/tools/fetch_car_run.sh --list     # show runs on the Jetson
#
# A run is deleted on the Jetson only after a checksum comparison finds no difference.
# Runs still being recorded (no bag/metadata.yaml yet) are skipped.
# Settings: JETSON_HOST (ssh alias, default jetson), LOCAL_RUN_LOGS (default
# <workspace>/run_logs/hybrid_mppi next to src/).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HOST="${JETSON_HOST:-jetson}"
REMOTE="f1tenth_ws/run_logs/hybrid_mppi"
LOCAL="${LOCAL_RUN_LOGS:-$(cd -- "$HERE/../../.." && pwd -P)/run_logs/hybrid_mppi}"
KEEP=false

case "${1:-}" in
  --list)
    ssh "$HOST" "cd $REMOTE 2>/dev/null && for r in */; do r=\${r%/}; \
      s=finished; [ -f \$r/bag/metadata.yaml ] || s='recording or unclosed'; \
      printf '%-40s %6s  %s\n' \$r \$(du -sh \$r | cut -f1) \"\$s\"; done" \
      || echo "No runs on $HOST:~/$REMOTE"
    exit 0 ;;
  --keep) KEEP=true; shift ;;
esac

if (( $# )); then
  RUNS=("$@")
else
  mapfile -t RUNS < <(ssh "$HOST" "cd $REMOTE 2>/dev/null && ls -1d */ | tr -d / | sort")
fi
if (( ${#RUNS[@]} == 0 )); then
  echo "No runs on $HOST:~/$REMOTE"
  exit 0
fi

mkdir -p "$LOCAL"
fetched=()
for RUN in "${RUNS[@]}"; do
  if ! ssh "$HOST" "test -f $REMOTE/$RUN/bag/metadata.yaml"; then
    echo "Skip $RUN: still recording, or not closed (repair on the Jetson: ros2 bag reindex $REMOTE/$RUN/bag)"
    continue
  fi
  need_kb=$(ssh "$HOST" "du -sk $REMOTE/$RUN" | cut -f1)
  free_kb=$(df -Pk "$LOCAL" | awk 'NR==2 {print $4}')
  if (( need_kb + 1048576 > free_kb )); then
    echo "Stop: $RUN needs $((need_kb / 1024)) MB, only $((free_kb / 1024)) MB free (keeping 1 GB spare)." >&2
    break
  fi

  echo "Copying $RUN ($((need_kb / 1024)) MB)"
  rsync -a --partial --info=progress2 "$HOST:$REMOTE/$RUN" "$LOCAL/"
  differences=$(rsync -a --checksum --dry-run --itemize-changes "$HOST:$REMOTE/$RUN" "$LOCAL/")
  if [[ -n "$differences" ]]; then
    echo "Verification failed for $RUN; kept on the Jetson:" >&2
    echo "$differences" >&2
    continue
  fi
  if $KEEP; then
    echo "Verified $RUN (kept on the Jetson)"
  else
    ssh "$HOST" "rm -rf -- $REMOTE/$RUN"
    echo "Verified $RUN, deleted it on the Jetson"
  fi
  fetched+=("$LOCAL/$RUN")
done

(( ${#fetched[@]} )) || exit 0
cat <<EOF

Fetched to $LOCAL:
$(printf '  %s\n' "${fetched[@]##*/}")

Replay in Foxglove (no ROS needed):
  1. Open Foxglove (desktop app or https://app.foxglove.dev) > Open local file > <run>/bag/*.mcap
  2. Layout > Import from file: $HERE/foxglove/mppi_car_layout.json
Replay through ROS instead: $HERE/replay_car_run.sh <run folder>
EOF
