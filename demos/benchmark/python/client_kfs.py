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
from tritonclient.grpc import service_pb2, service_pb2_grpc, InferResult
try:
    from ovms_benchmark_client.client import BaseClient
except ModuleNotFoundError:
    from client import BaseClient


class KFS_Client(BaseClient):
    """
    Inference client for benchmarks
    for Triton Inference Server
    """

    # override
    status_endpoint = "/v2/repository/index"
    DTYPE_FLOAT_64 = "FP64"
    DTYPE_FLOAT_32 = "FP32"
    DTYPE_FLOAT_16 = "FP16"

    DTYPE_INT_8 = "INT8"
    DTYPE_INT_16 = "INT16"
    DTYPE_INT_32 = "INT32"
    DTYPE_INT_64 = "INT64"

    DTYPE_UINT_8 = "UINT8"
    DTYPE_UINT_16 = "UINT16"
    DTYPE_UINT_32 = "UINT32"
    DTYPE_UINT_64 = "UINT64"

    # override
    def get_stub(self):
        return service_pb2_grpc.GRPCInferenceServiceStub

    # override
    def show_server_status(self):
        assert self.rest_port is not None, "checking only via REST port, which is not set"
        status_url = f"http://{self.address}:{self.rest_port}{self.status_endpoint}"
        self.print_info(f"try to send request to endpoint: {status_url}")
        response = requests.post(url=status_url, params={}, timeout=15)
        self.print_info(f"received status code is {response.status_code}.")
        message = "It seems to REST service is not running"
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
                "shape": [abs(int(i)) for i in resp_input.shape],
                "dtype": str(resp_input.datatype)
            }
        for resp_output in response.outputs:
            self.outputs[resp_output.name] = {
                "shape": [abs(int(i)) for i in resp_output.shape],
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

        # self.xdata -> {
        #     input-name-0: [ (data-0-0, meta-0-0), (data-0-1, meta-0-1), ...  ],
        #     input-name-1: [ (data-1-0, meta-1-0), (data-1-1, meta-1-1), ...  ],
        # }

        for index in range(self.dataset_length):
            batch_index = index % len(self.batchsizes)
            batch_length = self.batchsizes[batch_index]

            request = service_pb2.ModelInferRequest()
            request.model_version = str(self.model_version)
            request.model_name = self.model_name

            batch_input_bytes = []
            for input_name, xbatches in self.xdata.items():
                assert len(xbatches[index][0]) == batch_length

                # xbatches -> [ (data-0-0, meta-0-0), (data-0-1, meta-0-1), ...  ]
                # meta_data <- xbatches[index][1]  ->  ["dtype", "shape"]
                # pure_data <- xbatches[index][0]

                single_input_bytes = bytes()
                if "dtype" not in xbatches[index][1]:
                    for binary_data in xbatches[index][0]:
                        single_input_bytes += binary_data.tobytes()

                    shape = xbatches[index][1]["shape"]
                    shape[0] = batch_length
                    if "float32" in str(xbatches[index][0][0].dtype):
                        np_dtype, self.inputs[input_name]["dtype"] = "float32", "FP32"
                    elif "int8" in str(xbatches[index][0][0].dtype):
                        np_dtype, self.inputs[input_name]["dtype"] = "int8", "INT8"
                    else: raise ValueError(f"not supported type: {xbatches[index][0][0].dtype}")

                else:
                    if self.inputs[input_name]["dtype"] == self.DTYPE_INT_8: np_dtype = "int8"
                    elif self.inputs[input_name]["dtype"] == self.DTYPE_INT_32: np_dtype = "int32"
                    elif self.inputs[input_name]["dtype"] == self.DTYPE_FLOAT_32: np_dtype = "float32"
                    else: raise ValueError(f"not supported type: {xbatches[index][1]['dtype']}")

                    for numeric_data in xbatches[index][0]:
                        single_input_bytes += numeric_data.astype(np_dtype).tobytes()
                    shape = xbatches[index][1]["shape"]

                single_input_request = service_pb2.ModelInferRequest().InferInputTensor()
                single_input_request.datatype = self.inputs[input_name]["dtype"]
                single_input_request.name = input_name
                single_input_request.shape.extend(shape)

                request.inputs.append(single_input_request)
                batch_input_bytes.append(single_input_bytes)
            request.raw_input_contents.extend(batch_input_bytes)

            for output_name in self.outputs:
                output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
                output.name = output_name
                request.outputs.append(output)


            if self.stateful_length > 0:
                raise NotImplementedError("KFS / stateful")

            self.requests.append((batch_length, request))
        del self.xdata

    # override
    def predict(self, request, timeout):
        return self.stub.ModelInfer(request, timeout=timeout)
