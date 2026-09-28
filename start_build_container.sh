#!/usr/bin/env bash
mkdir -p /home/krz/.cache/ovms-bazel
chmod a+rwx /home/krz/.cache/ovms-bazel
docker rm ovms-build-krz
docker run -it \
  --env-file /home/krz/.env \
  -p 9112:9112 \
  -p 52000:52000/udp \
  --name ovms-build-krz \
  -v /home/krz/model_server:/ovms \
  -v /home/krz/.cache/ovms-bazel:/root/.cache/bazel \
  -v /home/krz/.gitconfig:/root/.gitconfig:ro \
  -v /usr/bin/gh:/usr/bin/gh:ro \
  -v /home/krz/.config/gh:/root/.config/gh:ro \
  openvino/model_server-build:krz \
  bash
