#!/bin/bash

mkdir -p $HOME/resnet50/1
if [ ! -f "$HOME/resnet50/1/resnet_50_i8.xml" ]; then
    wget https://storage.googleapis.com/public-artifacts/intelai_public_models/resnet_50_i8/1/resnet_50_i8.xml --directory-prefix $HOME/resnet50/1/
fi
if [ ! -f "$HOME/resnet50/1/resnet_50_i8.bin" ]; then
    wget https://storage.googleapis.com/public-artifacts/intelai_public_models/resnet_50_i8/1/resnet_50_i8.bin --directory-prefix $HOME/resnet50/1/
fi
