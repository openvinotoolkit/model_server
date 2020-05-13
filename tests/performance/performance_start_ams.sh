#!/bin/bash -e

# TODO:
#   * Consider parametrizing paths
#   * Might require providing mapping for neural network output


export FILE_SYSTEM_POLL_WAIT_SECONDS=0
AMS_PORT=5000
OVMS_PORT=9000
WORKERS=20
GRPC_WORKERS={{ grpc_workers }}

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
    --workers=*)
        WORKERS="${i#*=}"
        shift # past argument=value
    ;;
    --grpc_workers=*)
        GRPC_WORKERS="${i#*=}"
        shift # past argument=value
    ;;
    *)
        exit 0
    ;;
esac
done

. /ie-serving-py/.venv/bin/activate
/ie-serving-py/start_server.sh ie_serving config --config_path /opt/ams_models/ovms_config.json --grpc_workers $GRPC_WORKERS --port $OVMS_PORT &
cd /ams_wrapper && python -m src.wrapper --port $AMS_PORT --workers $WORKERS
