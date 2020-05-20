#!/bin/bash -e
#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
    --workers <workers>             AMS service workers (default: 20)
    --grpc_workers <grpc_workers>   OVMS service workers (default: 10)

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
