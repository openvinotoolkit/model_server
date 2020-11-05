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

/ovms/bin/ovms --model_path /opt/models/vehicle_detection_adas --model_name bad_name --port $OVMS_PORT &
cd /ams_wrapper && python3 -m src.wrapper --port $AMS_PORT
