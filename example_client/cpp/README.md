# gRPC client in c++

This basic C++ client can be used to run inference for OVMS endpoints. The client is reading jpeg file and sendng it to OVMS.

## Building the client

```bash
docker build .
```

## Starting docker container with OVMS
```bash
docker run -it -u $(id -u):$(id -g) --rm -v ~/models/:/models:ro -p 9100:9100  openvino/model_server:latest --model_name resnet --model_path gs://ovms-public-eu/resnet50-binary --port 9100
```

## Starting docker container with the client in dev mode:
```
docker run -it  -v ${PWD}:/tensorflow-serving/tensorflow_serving/example_client <image_name> bash
bazel build tensorflow_serving/example_client:resnet_client_cc

```


## Using the client
```
/tensorflow-serving# ./bazel-bin/tensorflow_serving/example_client/resnet_client_cc --server_port="<OVMS IP address>:9100" --image_file=/tensorflow-serving/bee.jpeg --model_name="resnet"
calling predict using file: /tensorflow-serving/bee.jpeg  ...
Image imported /tensorflow-serving/bee.jpegin1.951600ms
request serialized in0.765600ms
Prediction received in 71.569800ms
call predict ok
outputs size is 1
Response postprocessing in 0.046800ms
max class309 max value:-9.73885
the result tensor[0] is:
[-29.7019196 -29.0975952 -34.293457 -32.9355698 -37.1733398 -27.7851906 -35.7623711 -32.1231308 -32.5356178 -28.3972416...]...
Shape [1,1000]
Done.
```


