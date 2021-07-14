# TODO
# docker build . --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --build-arg no_proxy=$no_proxy -t cpp_client_v2

# docker cp

# 
docker run -it --rm --network="host" -v ~/workspace/:/workspace:rw --name cpp_client --entrypoint bazel-bin/src/resnet_client cpp_client_v2 --grpc_port="11337" --model_name="resnet" --input_name="0" --output_name="1463" --images_list="/workspace/cpp/input_images.txt"
