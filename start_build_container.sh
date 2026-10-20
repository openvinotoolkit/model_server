#!/usr/bin/env bash
IMAGE=docker.io/openvino/model_server-build:audio-placement
IMAGE=openvino/model_server-build:krz-local-genai-1792251046 
mkdir -p /home/krz/.cache/ovms-bazel
chmod a+rwx /home/krz/.cache/ovms-bazel
docker stop ovms-build-krz
docker rm ovms-build-krz
docker run -it \
  --env-file /home/krz/.env \
  -p 9112:9112 \
  -p 52000:52000/udp \
  --name ovms-build-krz \
  -v /home/krz/models:/ovms/models\
  -v /home/krz/model_server:/ovms \
  -v /home/krz/.cache/ovms-bazel:/root/.cache/bazel \
  -v /home/krz/.gitconfig:/root/.gitconfig:ro \
  -v /usr/bin/gh:/usr/bin/gh:ro \
  -v /home/krz/.config/gh:/root/.config/gh:ro \
  $IMAGE \
  bash
