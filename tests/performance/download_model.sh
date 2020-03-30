#!/bin/bash

mkdir -p /tmp/resnet50/1
if [ ! -f "/tmp/resnet50/1/resnet_50_i8.xml" ]; then
    wget https://storage.googleapis.com/public-artifacts/intelai_public_models/resnet_50_i8/1/resnet_50_i8.xml -o /tmp/resnet50/1/resnet_50_i8.xml
fi
if [ ! -f "/tmp/resnet50/1/resnet_50_i8.bin" ]; then
    wget https://storage.googleapis.com/public-artifacts/intelai_public_models/resnet_50_i8/1/resnet_50_i8.bin -o /tmp/resnet50/1/resnet_50_i8.bin
fi
