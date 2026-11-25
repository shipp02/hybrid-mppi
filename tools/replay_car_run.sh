#!/usr/bin/env bash
# Replay a fetched car run through ROS on the laptop, viewed in Foxglove via the bridge.
# For viewing only, opening bag/*.mcap directly in Foxglove is simpler (see fetch_car_run.sh).
#
#   src/hybrid-mppi/tools/replay_car_run.sh <run dir> [extra ros2 bag play args]
#   then in Foxglove: Open connection > ws://localhost:8765, import tools/foxglove/mppi_car_layout.json
#
# Safety: the car listens on ROS_DOMAIN_ID 10 and discovers peers on the subnet. Replay runs
# on REPLAY_DOMAIN_ID (default 42) and renames the recorded drive commands, so a replayed
# /drive can never reach the car.
set -euo pipefail
DIR="${1:?usage: replay_car_run.sh <run dir> [ros2 bag play args]}"
shift
DOMAIN="${REPLAY_DOMAIN_ID:-42}"
if [[ "$DOMAIN" == 10 ]]; then
  echo "Refusing to replay on ROS_DOMAIN_ID 10: that is the car's domain." >&2
  exit 1
fi
if ! command -v ros2 >/dev/null; then
  echo "ros2 not found: source /opt/ros/jazzy/setup.bash first" >&2
  exit 1
fi
export ROS_DOMAIN_ID="$DOMAIN"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

ros2 launch foxglove_bridge foxglove_bridge_launch.xml &
BRIDGE=$!
trap 'kill $BRIDGE 2>/dev/null' EXIT
sleep 2
echo "Replaying $DIR on ROS_DOMAIN_ID=$DOMAIN (localhost only). Foxglove: ws://localhost:8765"
ros2 bag play "$DIR/bag" \
  --remap /drive:=/replay/drive /teleop:=/replay/teleop /ackermann_cmd:=/replay/ackermann_cmd \
  "$@"
