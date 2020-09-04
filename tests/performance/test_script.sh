#!/bin/bash
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

NU_CLIENTS=$1
shift 1
sshpass -p user ssh user@10.237.114.207 docker restart ovms-c2 && sshpass -p user ssh user@10.237.114.207 docker exec -e GRPC_SERVERS=$1 -e NIREQ=$2 -e CPU_THROUGHPUT_STREAMS=$3 ovms-c2 /ovms/bazel-bin/ovms --config_path /models/config.json > /dev/null &
sleep 5
ITER=1000
A=$(./grpc_throughput.sh ${NU_CLIENTS} --grpc_address 192.168.0.4 --grpc_port 9179 --images_numpy_path imgs.npy --iteration ${ITER} --batchsize 1 --input_name "data" --report_every ${ITER}  2>&1 | grep -E "\[${NU_CLIENTS}\] Iterations:|real" | tr '\n' ' ')
echo "$A" ${1} ${2} ${3} >> results_${ITER}
