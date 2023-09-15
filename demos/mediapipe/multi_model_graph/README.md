# MediaPipe Multi Model Demo {#ovms_docs_demo_mediapipe_multi_model}

This guide shows how to implement [MediaPipe](../../../docs/mediapipe.md) graph using OVMS.

Example usage:

## Prepare the repository

Clone the repository and enter mediapipe image_classification directory
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/mediapipe/multi_model_graph
```

## Download ResNet50 model

```bash
cp -r ../../../src/test/add_two_inputs_model ./
cp -r ../../../src/test/dummy ./
```

## Run OpenVINO Model Server
Prepare virtualenv according to [kserve samples readme](https://github.com/openvinotoolkit/model_server/blob/main/client/python/kserve-api/samples/README.md)
```bash
docker run -d -v $PWD:/mediapipe -p 9000:9000 openvino/model_server:latest --config_path /mediapipe/config.json --port 9000
```

## Run the client:
```bash
python mediapipe_multi_model_client.py --grpc_port 9000
Output:
[[ 3.  5.  7.  9. 11. 13. 15. 17. 19. 21.]]
```
