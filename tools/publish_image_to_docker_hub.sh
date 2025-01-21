#!/bin/bash
#
# Copyright (c) 2024 Intel Corporation
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

TEMP=$(getopt -n "$0" -a -l "image:,image_gpu:,release_tag:,help" -- -- "$@")

[ $? -eq 0 ] || exit

eval set --  "$TEMP"

HELP=false

while [ $# -gt 0 ]
do
            case "$1" in
                --image) IMAGE="$2"; shift;;
                --image_gpu) IMAGE_GPU="$2"; shift;;
                --release_tag) RELEASE_TAG="$2"; shift;;
                --help) HELP=true; shift;;
                --) shift;;
            esac
            shift;
done
if [ "$HELP" = true ] ; then
    echo 'Example usage: publish_image_to_docker_hub.sh --image registry.toolbox.iotg.sclab.intel.com/openvino/model_server:ubuntu22_main --image_gpu registry.toolbox.iotg.sclab.intel.com/openvino/model_server-gpu:ubuntu22_main --release_tag 2024.3'
    exit 1
fi

if [ "$IMAGE" = "" ] || [ "$IMAGE_GPU" = "" ] || [ "$RELEASE_TAG" = "" ] ; then
    echo '--image, --image_gpu, --release_tag parameters are required'
    exit 1
fi

echo "Given parameters: ";
echo "image: $IMAGE";
echo "image_gpu: $IMAGE_GPU";
echo "release_tag: $RELEASE_TAG";
echo "Press y to continue..."
read
if [ "$REPLY" != "y" ] ; then
    exit 1
fi

docker pull $IMAGE;
docker tag $IMAGE openvino/model_server:$RELEASE_TAG;
docker run openvino/model_server:$RELEASE_TAG --version;
echo "Press y to continue..."
read
if [ "$REPLY" != "y" ] ; then
    exit 1
fi
docker push openvino/model_server:$RELEASE_TAG;

docker tag $IMAGE openvino/model_server:latest;
docker run openvino/model_server:latest --version;
echo "Press y to continue..."
read
if [ "$REPLY" != "y" ] ; then
    exit 1
fi
docker push openvino/model_server:latest;

docker pull $IMAGE_GPU;
docker tag $IMAGE_GPU openvino/model_server:$RELEASE_TAG-gpu;
docker run openvino/model_server:$RELEASE_TAG-gpu --version;
echo "Press y to continue..."
read
if [ "$REPLY" != "y" ] ; then
    exit 1
fi
docker push openvino/model_server:$RELEASE_TAG-gpu;

docker tag $IMAGE_GPU openvino/model_server:latest-gpu;
docker run openvino/model_server:latest-gpu --version;
echo "Press y to continue..."
read
if [ "$REPLY" != "y" ] ; then
    exit 1
fi
docker push openvino/model_server:latest-gpu;
