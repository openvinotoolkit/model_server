#!/bin/bash -e

# script should be run from model server directory

python example_client/face_detection_performance.py --model_name $1 --input_images_dir $5 --grpc_port $4 --width $2 --height $3