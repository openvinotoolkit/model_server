
Build a custom docker image
```
cd src

mkdir mediapipe_calculators

mv test/mediapipe/calculators/ovmscalculator.proto src/mediapipe_calculators

# Build custom docker image
make docker_build   OVMS_CPP_DOCKER_IMAGE=ovms_custom   TARGET=release
```