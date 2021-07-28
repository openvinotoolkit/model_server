# Example clients in C++

To build examplary clients, run `make` command in this directory. It will build docker image with all dependencies.

There are multiple parameters to choose from:
```bash
docker run --rm -it --network host -v $(pwd)/../:/workspace:rw cpp_clients_build_image                                                                                                
usage: /build/bazel-bin/src/resnet_client
Flags:
        --grpc_address="localhost"              string  url to grpc service
        --grpc_port="9000"                      string  port to grpc service
        --model_name="resnet"                   string  model name to request
        --input_name="0"                        string  input tensor name with image
        --output_name="1463"                    string  output tensor name with classification result
        --iterations=0                          int64   number of images per thread, by default each thread will use all images from list
        --images_list=""                        string  path to a file with a list of labeled images
        --layout="binary"                       string  binary, nhwc or nchw
```

Start OVMS with resnet50-binary model:
```bash
curl -L --create-dir https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin -o resnet50-binary/1/model.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o resnet50-binary/1/model.xml
```

```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/resnet50-binary:/model -p 9001:9001 openvino/model_server:latest \
--model_path /model --model_name resnet --port 9001 --layout NHWC
```

Run the client:
```bash
docker run --rm -it --network host -v $(pwd)/../:/workspace:rw cpp_clients_build_image --grpc_port=9001 --images_list="/workspace/cpp/input_images.txt" --iterations=200

call predict ok
call predict time: 23ms
outputs size is 1
most probable label: 144; expected: 144; OK
call predict ok
call predict time: 22ms
outputs size is 1
most probable label: 113; expected: 113; OK
call predict ok
call predict time: 26ms
outputs size is 1
most probable label: 340; expected: 340; OK
...
```

The client will send binary images to OVMS and select most probable label from model output.

You can also change image format to be either in binary (by default), in NCHW or NHWC format using `--layout [binary|nchw|nhwc]` flag.