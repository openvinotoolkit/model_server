#
# Copyright (c) 2021 Intel Corporation
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

"""Inference client for benchmarks for Triton Inference Server"""

import sys
import json
from http import HTTPStatus

import grpc
import requests
from retry.api import retry_call
from tritonclient.grpc import service_pb2, service_pb2_grpc
from ovms_benchmark_client.client import BaseClient


class NvTrtClient(BaseClient):
    """
    Inference client for benchmarks
    for Triton Inference Server
    """

    # override
    status_endpoint = "/v2/repository/index"
    DTYPE_FLOAT_32 = "FP32"
    DTYPE_INT_32 = "INT32"

    # override
    def get_stub(self):
        return service_pb2_grpc.GRPCInferenceServiceStub

    # override
    def show_server_status(self):
        assert self.rest_port is not None, "checking only via REST port, which is not set"
        status_url = f"http://{self.address}:{self.rest_port}{self.status_endpoint}"
        self.print_info(f"try to send request to endpoint: {status_url}")
        response = requests.post(url=status_url, params={})
        self.print_info(f"received status code is {response.status_code}.")
        message = "It seems to REST service is not runnig"
        assert response.status_code == HTTPStatus.OK.value, message
        self.print_info("found models and their status:")
        for model in list(response.json()):
            name, version, state = model.values()
            self.print_info(
                f"{self.indent}model: {name:20s} version: {version} - {state}")
        jstatus = response.json()
        if not self.jsonout:
            return jstatus
        jout = json.dumps(jstatus)
        print(f"{self.json_prefix}###{self.worker_id}###STATUS###{jout}")
        return jstatus

    # override
    def get_model_metadata(self, model_name, model_version=None, timeout=60):
        self.print_info(f"request for metadata of model {model_name}...")
        kwargs = {"name": model_name}
        if model_version is not None:
            kwargs["version"] = int(model_version)
        try:
            request = service_pb2.ModelMetadataRequest(**kwargs)
            rargs = (request, int(timeout))
            func = self.stub.ModelMetadata
            response = retry_call(func, rargs, **self.retry_setup)
            self.print_info(f"Metadata for model {model_name} is downloaded...")
        except grpc.RpcError as err:
            self.print_error(f"Metadata could not be downloaded: {err}")
            self.final_status = False
            sys.exit(-1)

        assert response.name == model_name
        versions = [int(v) for v in response.versions]
        if model_version is not None:
            assert int(model_version) in versions, \
                f"{model_version} is not in {versions}"
        else:
            model_version = versions[-1]
            self.print_info(
                f"set version of model {model_name}: {model_version}")
        self.model_name, self.model_version = model_name, model_version

        self.print_info(response)
        for resp_input in response.inputs:
            self.inputs[resp_input.name] = {
                "shape": [1] + [abs(int(i)) for i in resp_input.shape],
                "dtype": str(resp_input.datatype)
            }
        for resp_output in response.outputs:
            self.outputs[resp_output.name] = {
                "shape": [1] + [abs(int(i)) for i in resp_output.shape],
                "dtype": str(resp_output.datatype)
            }

        metadict = {
            "model_name": model_name,
            "model_version": model_version,
            "inputs": self.inputs,
            "outputs": self.outputs
        }
        if not self.jsonout:
            return metadict

        jout = json.dumps(metadict)
        print(f"{self.json_prefix}###{self.worker_id}###METADATA###{jout}")
        return metadict

    # override
    def prepare_batch_requests(self):
        for key, values in self.xdata.items():
            assert len(values) == self.dataset_length, f"{key} data has wrong length"

        for index in range(self.dataset_length):
            request = service_pb2.ModelInferRequest()
            request.model_name = self.model_name
            request.model_version = str(self.model_version)

            all_input_bytes = []
            for input_name, xbatches in self.xdata.items():
                if self.inputs[input_name]["dtype"] == self.DTYPE_FLOAT_32:
                    np_dtype = "float32"
                elif self.inputs[input_name]["dtype"] == self.DTYPE_INT_32:
                    np_dtype = "int8"
                else:
                    raise ValueError(f"not supported type: {self.inputs[input_name]['dtype']}")

                input_bytes = bytes()
                data = xbatches[index][0]
                img_bytes = data[0].astype(np_dtype).tobytes()
                input_bytes += img_bytes
                shape = data[0].shape

                request_input = service_pb2.ModelInferRequest().InferInputTensor()
                request_input.name = input_name
                request_input.datatype = self.inputs[input_name]["dtype"]
                request_input.shape.extend(shape)
                request.inputs.extend([request_input])
                all_input_bytes.append(input_bytes)
            request.raw_input_contents.extend(all_input_bytes)

            for output_name in self.outputs:
                output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
                output.name = output_name
                request.outputs.extend([output])

            if self.stateful_length > 0:
                request.parameters["sequence_id"].int64_param = self.stateful_id
                if self.stateful_counter == 0:
                    request.parameters["sequence_start"].bool_param = True
                elif self.stateful_counter >= int(self.stateful_length) - 1:
                    request.parameters["sequence_end"].bool_param = True
                    self.stateful_id += self.stateful_hop
                    self.stateful_counter = -1
                self.stateful_counter += 1

            batch_index = index % len(self.batchsizes)
            batch_length = self.batchsizes[batch_index]
            self.requests.append((batch_length, request))

        del self.xdata

    # ovrride
    def predict(self, request, timeout):
        return self.stub.ModelInfer(request, timeout=timeout)
