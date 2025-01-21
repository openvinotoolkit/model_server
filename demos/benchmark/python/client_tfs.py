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

"""Inference client for benchmarks for OpenVINO Model Server"""

import sys
import json
import numpy
from http import HTTPStatus

import requests
from retry.api import retry_call
from tensorflow import make_tensor_proto
from google.protobuf.json_format import MessageToJson
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
try:
    from ovms_benchmark_client.client import BaseClient
except ModuleNotFoundError:
    from client import BaseClient


class TFS_Client(BaseClient):
    """
    Inference client for benchmarks
    for Open Vino Model Server
    """

    # override
    status_endpoint = "/v1/config"

    DTYPE_FLOAT_16 = "DT_FLOAT16"
    DTYPE_FLOAT_32 = "DT_FLOAT"
    DTYPE_FLOAT_64 = "DT_FLOAT64"

    DTYPE_INT_8 = "DT_INT8"
    DTYPE_INT_16 = "DT_INT16"
    DTYPE_INT_32 = "DT_INT32"
    DTYPE_INT_64 = "DT_INT64"

    DTYPE_UINT_8 = "DT_UINT8"
    DTYPE_UINT_16 = "DT_UINT16"
    DTYPE_UINT_32 = "DT_UINT32"
    DTYPE_UINT_64 = "DT_UINT64"

    # override
    def get_stub(self):
        return prediction_service_pb2_grpc.PredictionServiceStub

    # override
    def show_server_status(self):
        assert self.rest_port is not None, "checking only via REST port, which is not set"
        status_url = f"http://{self.address}:{self.rest_port}{self.status_endpoint}"
        self.print_info(f"try to send request to endpoint: {status_url}")
        response = requests.get(url=status_url, params={})
        self.print_info(f"received status code is {response.status_code}.")
        message = "It seems to REST service is not running or OVMS version is too old!"
        assert response.status_code == HTTPStatus.OK.value, message
        self.print_info("found models and their status:")
        for model, status in response.json().items():
            for item in status["model_version_status"]:
                model_name_version = f"model: {model}, version: {item['version']}"
                self.print_info(
                    f"{self.indent}{model_name_version} - {item['state']}")
            if not status["model_version_status"]:
                self.print_info(f"{self.indent}{model} - EMPTY")
        jstatus = response.json()
        if not self.jsonout:
            return jstatus
        jout = json.dumps(jstatus)
        print(f"{self.json_prefix}###{self.worker_id}###STATUS###{jout}")
        return jstatus

    # override
    def get_model_metadata(self, model_name, model_version=None, timeout=60):
        self.print_info(f"request for metadata of model {model_name}...")
        request = get_model_metadata_pb2.GetModelMetadataRequest()
        request.metadata_field.append("signature_def")
        request.model_spec.name = model_name
        if model_version is not None:
            request.model_spec.version.value = int(model_version)
        try:
            rargs = (request, int(timeout))
            func = self.stub.GetModelMetadata
            response = retry_call(func, rargs, **self.retry_setup)
            self.print_info(
                f"Metadata for model {model_name} is downloaded...")
        except Exception as err:
            self.print_error(f"Metadata could not be downloaded: {err}")
            self.final_status = False
            sys.exit(-1)

        rjson = json.loads(MessageToJson(response))
        assert rjson["modelSpec"]["name"] == model_name
        if model_version is not None:
            version = int(rjson["modelSpec"]["version"])
            assert version == int(
                model_version), f"{version} != {model_version}"
        else:
            model_version = int(rjson["modelSpec"]["version"])
            self.print_info(
                f"set version of model {model_name}: {model_version}")
        self.model_name, self.model_version = model_name, model_version
        io_config = rjson["metadata"]["signature_def"]["signatureDef"]["serving_default"]

        for topic, config in io_config.items():
            self.print_info(f"{topic}:")
            if topic == "inputs":
                container = self.inputs
            elif topic == "outputs":
                container = self.outputs
            else:
                self.print_info(f"unknown topic: {topic}")
                continue
            for name, details in config.items():
                self.print_info(f"{self.indent}{name}:")
                for key, val in details.items():
                    self.print_info(f"{self.indent2}{key}: {val}")
                self.shape_description_reduce(details)
                container[name] = details
        metadict = {
            "model_name": model_name, "model_version": model_version,
            "inputs": self.inputs, "outputs": self.outputs
        }
        if not self.jsonout:
            return metadict

        jout = json.dumps(metadict)
        print(f"{self.json_prefix}###{self.worker_id}###METADATA###{jout}")
        return metadict

    # used only in XClient by get_model_metadata()
    def shape_description_reduce(self, config):
        assert "tensorShape" in config, f"no tensorShape in config: {config}"
        shape_int_list = []
        for dim in config["tensorShape"]["dim"]:
            if "size" not in dim.keys():
                continue
            shape_int_list.append(int(dim["size"]))
        config["shape"] = shape_int_list
        del config["tensorShape"]
        del config["name"]

    # override
    def prepare_batch_requests(self):
        for key, values in self.xdata.items():
            assert len(values) == self.dataset_length, f"{key} data has wrong length"

        # self.xdata -> {
        #     input-name-0: [  (data-0-0, meta-0-0), (data-0-1, meta-0-1), ...  ],
        #     input-name-1: [  (data-1-0, meta-1-0), (data-1-1, meta-1-1), ...  ],
        # }

        for index in range(self.dataset_length):
            request = predict_pb2.PredictRequest()
            request.model_spec.name = self.model_name
            for input_name, xbatches in self.xdata.items():
                try: tensor = make_tensor_proto(xbatches[index][0], **xbatches[index][1])
                except: raise NotImplementedError("TFS / numpy")
                request.inputs[input_name].CopyFrom(tensor)

            if self.stateful_length > 0:
                tensor_id = make_tensor_proto([numpy.uint64(self.stateful_id)], dtype="uint64")
                if self.stateful_counter == 0:
                    tensor_ctrl = make_tensor_proto([self.STATEFUL_START], dtype="uint32")
                elif self.stateful_counter >= int(self.stateful_length) - 1:
                    tensor_ctrl = make_tensor_proto([self.STATEFUL_STOP], dtype="uint32")
                    self.stateful_id += self.stateful_hop
                    self.stateful_counter = -1
                else: tensor_ctrl = make_tensor_proto([self.STATEFUL_WIP], dtype="uint32")
                request.inputs["sequence_control_input"].CopyFrom(tensor_ctrl)
                request.inputs["sequence_id"].CopyFrom(tensor_id)
                self.stateful_counter += 1

            batch_index = index % len(self.batchsizes)
            batch_length = self.batchsizes[batch_index]
            self.requests.append((batch_length, request))
        del self.xdata

    # override
    def predict(self, request, timeout):
        return self.stub.Predict(request, int(timeout))
