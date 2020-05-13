#!/bin/bash -e

# TODO: 
#   * Consider parametrizing paths
#   * Might require providing mapping for neural network output

help_message="
    This script runs OpenVINO Model Server and AMS Service in the background.
    OVMS will served models available under path /opt/models with configuration
    defined in /opt/models/config.json file. 
    
    Options:
    --help                          Show help message
    --ams_port <ams_port>           Port for AMS Service to listen on
                                    (default: 5000)
    --ovms_port <ovms_port>         Port for OVMS to listen on
                                    (default: 9000)

    Example:
    ./start_ams.sh --ams_port=4000 --ovms_port=8080
    
    This command will start AMS service listening on port 4000 and OVMS service
    listening on port 8080.
    "


export FILE_SYSTEM_POLL_WAIT_SECONDS=0
AMS_PORT=5000
OVMS_PORT=9000
WORKERS=20
GRPC_WORKERS=10

for i in "$@"
do
case $i in
    --help=*)
        echo "$help_message"
        exit 0
    ;;
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
        echo "$help_message"
        exit 0
    ;;
esac
done

. /ie-serving-py/.venv/bin/activate
/ie-serving-py/start_server.sh ie_serving config --config_path /opt/ams_models/ovms_config.json --grpc_workers $GRPC_WORKERS --port $OVMS_PORT &
cd /ams_wrapper && python -m src.wrapper --port $AMS_PORT --workers $WORKERS
