#!/bin/bash

# TODO: 
#   * Consider more user friendly arguments passing
#   * Consider parametrizing paths
#   * Might require providing mapping for neural network output

export FILE_SYSTEM_POLL_WAIT_SECONDS=0

AMS_PORT=${1:-5000}
OVMS_PORT=${2:-9000}

. /ie-serving-py/.venv/bin/activate
/ie-serving-py/start_server.sh ie_serving config --config_path /opt/models/config.json --port $OVMS_PORT & 
python /ams_wrapper/src/wrapper.py --port $AMS_PORT
