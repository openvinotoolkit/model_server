#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from invoke import task

supported_clients = {"ovmsclient":["3.8","3.9","3.10","3.11","3.12"],
                     "kserve-api":["3.8","3.9","3.10","3.11","3.12"],
                     "tensorflow-serving-api":["3.8","3.9","3.10","3.11","3.12"]}

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
                            c.run(f". .venv{client}{v}/bin/activate && python{v} http_infer_binary_resnet.py --http_port {rest} --images_list ../../resnet_input_images.txt --input_name map/TensorArrayStack/TensorArrayGatherV3 --output_name softmax_tensor:0 --model_name resnet", hide=not(verbose))
                            c.run(f". .venv{client}{v}/bin/activate && python{v} grpc_infer_binary_resnet.py --grpc_port {grpc} --images_list ../../resnet_input_images.txt --input_name map/TensorArrayStack/TensorArrayGatherV3 --output_name softmax_tensor:0 --model_name resnet", hide=not(verbose))
                        if client == "tensorflow-serving-api":
                            c.run(f". .venv{client}{v}/bin/activate && python{v} grpc_predict_binary_resnet.py --grpc_address localhost --model_name resnet --input_name map/TensorArrayStack/TensorArrayGatherV3 --output_name softmax_tensor:0 --grpc_port {grpc} --images ../../resnet_input_images.txt", hide=not(verbose))
                        if client == "ovmsclient":
                            c.run(f". .venv{client}{v}/bin/activate && python{v} grpc_predict_resnet.py --images_numpy ../../imgs_nhwc.npy --model_name resnet --service_url localhost:{grpc}", hide=not(verbose))
                            c.run(f". .venv{client}{v}/bin/activate && python{v} http_predict_resnet.py --images_numpy ../../imgs_nhwc.npy --model_name resnet --service_url localhost:{rest}", hide=not(verbose))
                        c.run("echo OK")
                    except:
                        c.run("echo FAIL")
                        if fastFail:
                            raise Exception

