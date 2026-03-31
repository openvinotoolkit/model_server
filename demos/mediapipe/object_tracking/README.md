
Build a custom docker image
```
cd src

mkdir mediapipe_calculators

mv test/mediapipe/calculators/ovmscalculator.proto src/mediapipe_calculators

# Build custom docker image
make docker_build   OVMS_CPP_DOCKER_IMAGE=ovms_custom   TARGET=release
```

Container command
```
docker run -d -v $PWD:/mediapipe -p 9000:9000 ovms_custom:latest --config_path /mediapipe/config.json --port 9000
```

run command
```
python mediapipe_object_tracking.py --grpc_port 9000 --input_video test_video.mp4
```