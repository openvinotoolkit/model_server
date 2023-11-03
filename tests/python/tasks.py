from invoke import task

supported_clients = {"ovmsclient":["3.7", "3.8", "3.9"],
                     "kserve-api":["3.8", "3.9", "3.10", "3.11"],
                     "tensorflow-serving-api":["3.7", "3.8", "3.9", "3.10"]}

@task
def deps(c, verbose=False, fastFail=False):
    with c.cd("model_server/client/python"):
        for client in supported_clients:
            with c.cd(f"{client}/samples"):
                for v in supported_clients[client]:
                    c.run(f"echo {client} $(python{v} --version)")
                    try:
                        c.run(f"python{v} -m venv .venv{client}{v}")
                        c.run(f". .venv{client}{v}/bin/activate && pip install --upgrade pip && python{v} -m pip install -r requirements.txt", hide=not(verbose))
                        c.run("echo OK")
                    except:
                        c.run("echo FAIL")
                        if fastFail:
                            raise Exception

@task
def test(c, rest=8000, grpc=9000, verbose=False, fastFail=False):
    with c.cd("model_server/client/python"):
        for client in supported_clients:
            with c.cd(f"{client}/samples"):
                for v in supported_clients[client]:
                    c.run(f"echo {client} $(python{v} --version)")
                    try:
                        if client == "kserve-api":
                            c.run(f". .venv{client}{v}/bin/activate && python{v} http_infer_binary_resnet.py --http_port {rest} --images_list ../../resnet_input_images.txt --input_name map/TensorArrayStack/TensorArrayGatherV3 --output_name softmax_tensor --model_name resnet", hide=not(verbose))
                            c.run(f". .venv{client}{v}/bin/activate && python{v} grpc_infer_binary_resnet.py --grpc_port {grpc} --images_list ../../resnet_input_images.txt --input_name map/TensorArrayStack/TensorArrayGatherV3 --output_name softmax_tensor --model_name resnet", hide=not(verbose))
                        if client == "tensorflow-serving-api":
                            c.run(f". .venv{client}{v}/bin/activate && python{v} grpc_predict_binary_resnet.py --grpc_address localhost --model_name resnet --input_name map/TensorArrayStack/TensorArrayGatherV3 --output_name softmax_tensor --grpc_port {grpc} --images ../../resnet_input_images.txt", hide=not(verbose))
                        if client == "ovmsclient":
                            c.run(f". .venv{client}{v}/bin/activate && python{v} grpc_predict_resnet.py --images_numpy ../../imgs_nhwc.npy --model_name resnet --service_url localhost:{grpc}", hide=not(verbose))
                            c.run(f". .venv{client}{v}/bin/activate && python{v} http_predict_resnet.py --images_numpy ../../imgs_nhwc.npy --model_name resnet --service_url localhost:{rest}", hide=not(verbose))
                        c.run("echo OK")
                    except:
                        c.run("echo FAIL")
                        if fastFail:
                            raise Exception

