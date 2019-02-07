#
# Copyright (c) 2018-2019 Intel Corporation
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

import numpy as np
import sys
sys.path.append(".")
from conftest import infer, get_model_metadata, model_metadata_response, \
    wait_endpoint_setup, ERROR_SHAPE # noqa


class TestSingleModelInferenceGc():

    def test_run_inference(self, input_data_downloader_v1_224,
                           start_server_single_model_from_gc,
                           create_channel_for_port_single_server):
        """
        <b>Description</b>
        Submit request to gRPC interface serving a single resnet model

        <b>input data</b>
        - directory with the model in IR format
        - docker image with ie-serving-py service
        - input data in numpy format

        <b>fixtures used</b>
        - model downloader
        - input data downloader
        - service launching

        <b>Expected results</b>
        - response contains proper numpy shape

        """

        # Starting docker with ie-serving
        container = start_server_single_model_from_gc
        running, logs = wait_endpoint_setup(container)
        print("Logs from container: ", logs)
        assert running is True, "docker container was not started successfully"

        # Connect to grpc service
        stub = create_channel_for_port_single_server

        imgs_v1_224 = np.array(input_data_downloader_v1_224)
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        for x in range(0, 10):
            output = infer(imgs_v1_224, slice_number=x,
                           input_tensor='input', grpc_stub=stub,
                           model_spec_name='resnet',
                           model_spec_version=None,
                           output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1000), ERROR_SHAPE

    def test_get_model_metadata(self, start_server_single_model_from_gc,
                                create_channel_for_port_single_server):

        container = start_server_single_model_from_gc
        running, logs = wait_endpoint_setup(container)
        print("Logs from container: ", logs)
        assert running is True, "docker container was not started successfully"

        stub = create_channel_for_port_single_server

        model_name = 'resnet'
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [1, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [1, 1000]}}
        request = get_model_metadata(model_name='resnet')
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata
