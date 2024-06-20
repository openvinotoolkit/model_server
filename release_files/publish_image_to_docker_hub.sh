#!/bin/bash

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
    echo 'Example usage: publish_image_to_docker_hub.sh --image registry.toolbox.iotg.sclab.intel.com/openvino/model_server:ubuntu22_main --image_gpu registry.toolbox.iotg.sclab.intel.com/openvino/model_server-gpu:ubuntu22_main --release_tag 2023.2'
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

docker pull $IMAGE;
docker tag $IMAGE openvino/model_server:$RELEASE_TAG;
docker push openvino/model_server:$RELEASE_TAG;

docker tag $IMAGE openvino/model_server:latest;
docker push openvino/model_server:latest;

echo "Image pushed to $REPOSITORY";

docker pull $IMAGE_GPU;
docker tag $IMAGE_GPU openvino/model_server:$RELEASE_TAG-gpu;
docker push openvino/model_server:$RELEASE_TAG-gpu;

docker tag $IMAGE_GPU openvino/model_server:latest-gpu;
docker push openvino/model_server:latest-gpu;

echo "Image with gpu pushed to $REPOSITORY";