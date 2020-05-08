#!/bin/bash -e

export FILE_SYSTEM_POLL_WAIT_SECONDS=0
AMS_PORT=5000
OVMS_PORT=9000

for i in "$@"
do
case $i in
    --ams_port=*)
        AMS_PORT="${i#*=}"
        shift # past argument=value
    ;;
    --ovms_port=*)
        OVMS_PORT="${i#*=}"
        shift # past argument=value
    ;;
    *)
        exit 0
    ;;
esac
done

. /ie-serving-py/.venv/bin/activate
/ie-serving-py/start_server.sh ie_serving model --model_path /opt/models/vehicle_detection_adas --model_name bad_name --port $OVMS_PORT & 
cd /ams_wrapper && python -m src.wrapper --port $AMS_PORT
