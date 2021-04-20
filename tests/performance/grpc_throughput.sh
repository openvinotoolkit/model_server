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

CLIENT_COUNT=${1}
shift 1
SECONDS=0

ITERATION=-1
BATCHSIZE=1
for var in "$@"
do
    if [ $ITERATION = 0 ]; then
        ITERATION=$var
    fi

    if [ $var = "--iteration" ]; then
        ITERATION=0
    fi

    if [ $BATCHSIZE = 0 ]; then
        BATCHSIZE=$var
    fi

    if [ $var = "--batchsize" ]; then
        BATCHSIZE=0
    fi
done

if [ $ITERATION -lt 1 ]; then
    echo "Cannot find --iteration in command line"
    exit 1
fi

time printf %s\\n $(seq 1 $CLIENT_COUNT) | xargs -n 1 -P $CLIENT_COUNT ./grpc_start_one_stream.sh $* --id

if [ $? = 1 ]; then
    echo "grpc_start_one_stream.sh has failed"
    exit 1
fi

FPS=$((CLIENT_COUNT*ITERATION*BATCHSIZE/SECONDS))
echo "$FPS FPS"
