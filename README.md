
# Minimal gRPC serving with TFS API
How to run:

1. Build image with TF Serving:
```bash
~/inference-experiments/tfs_c_frontend$ docker build . --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t cpp-experiments
```

2. Run container and mount server sources:
```bash
~/inference-experiments/tfs_c_frontend$ docker run -it -v $(pwd)/src:/tensorflow-serving/tensorflow_serving/ovms/:rw -p 9000:9000 -e "http_proxy=$http_proxy" -e "https_proxy=$https_proxy" cpp-experiments bash
```

3. Compile server source inside container:
```bash
root@7148320dfffd:/tensorflow-serving$ bazel build //tensorflow_serving/ovms:server_cc
```

4. Run the server:
```bash
root@7148320dfffd:/tensorflow-serving$ ./bazel-bin/tensorflow_serving/ovms/server_cc
Initializing
Server started on port 9000
```

5. Test the server (run client from host in second terminal)
```
$ git clone https://github.com/IntelAI/OpenVINO-model-server
$ cd OpenVINO-model-server
$ make install
$ . .venv/bin/activate
(.venv) $ cd example_client
(.venv) $ python3 grpc_serving_client.py --grpc_port 9000 --images_numpy_path imgs.npy --input_name data --output_name prob --transpose_input False --labels_numpy lbs.npy
```

You should see:  
  
```
(.venv3) wihajster@dev-115-178:~/OpenVINO-model-server/example_client$ python grpc_serving_client.py --grpc_port 9000 --images_numpy_path imgs.npy --input_name data --output_name prob --transpose_input False --labels_numpy lbs.npy
Image data range: 0.0 : 255.0
Start processing:
        Model name: resnet
        Iterations: 10
        Images numpy path: imgs.npy       
        Images in shape: (10, 3, 224, 224)

Invalid output name prob
Available outputs:
```
