#!/bin/bash -e

# script should be run from model server directory

python example_client/face_detection_performance.py --model_name vehicle_detection --input_images_dir /root/model_server/tests/functional/fixtures/test_images/performance/ --grpc_port 9007 --width 672 --height 384