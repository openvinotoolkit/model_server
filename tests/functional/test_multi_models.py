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
from conftest import infer, infer_batch, get_model_metadata, \
    model_metadata_response, ERROR_SHAPE # noqa


class TestMuiltModelInference():

    def test_run_inference(self, download_two_models,
                           input_data_downloader_v1_224,
                           input_data_downloader_v3_331,
                           start_server_multi_model,
                           create_channel_for_port_multi_server):
        """
        <b>Description</b>
        Execute inference request using gRPC interface hosting multiple models

        <b>input data</b>
        - directory with 2 models in IR format
        - docker image
        - input data in numpy format

        <b>fixtures used</b>
        - model downloader
        - input data downloader
        - service launching

        <b>Expected results</b>
        - response contains proper numpy shape for both models set in config
        file: model resnet_v1_50, pnasnet_large
        - both served models handles appropriate input formats

        """

        print("Downloaded model files:", download_two_models)

        print("Start Ovms image: ", start_server_multi_model)

        # Connect to grpc service
        stub = create_channel_for_port_multi_server

        input_data = input_data_downloader_v1_224[:2, :, :, :]
        print("Starting inference using resnet model")
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        for x in range(0, 10):
            output = infer_batch(input_data, input_tensor='input',
                                 grpc_stub=stub,
                                 model_spec_name='resnet_V1_50',
                                 model_spec_version=None,
                                 output_tensors=[out_name])
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == (2, 1000), ERROR_SHAPE

        imgs_v1_224 = np.array(input_data_downloader_v1_224)
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        for x in range(0, 10):
            output = infer(imgs_v1_224, slice_number=x,
                           input_tensor='input', grpc_stub=stub,
                           model_spec_name='resnet_gs',
                           model_spec_version=None,
                           output_tensors=[out_name])
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == (1, 1000), ERROR_SHAPE

        out_name = 'resnet_v1_50/predictions/Reshape_1'
        for x in range(0, 10):
            output = infer(imgs_v1_224, slice_number=x,
                           input_tensor='input', grpc_stub=stub,
                           model_spec_name='resnet_s3',
                           model_spec_version=None,
                           output_tensors=[out_name])
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == (1, 1000), ERROR_SHAPE

        input_data = input_data_downloader_v3_331[:4, :, :, :]
        print("Starting inference using pnasnet_large model")
        out_name = 'final_layer/predictions'
        for x in range(0, 10):
            output = infer_batch(input_data, input_tensor='input',
                                 grpc_stub=stub,
                                 model_spec_name='pnasnet_large',
                                 model_spec_version=None,
                                 output_tensors=[out_name])
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == (4, 1001), ERROR_SHAPE

    def test_get_model_metadata(self, download_two_models,
                                start_server_multi_model,
                                create_channel_for_port_multi_server):
        """
        <b>Description</b>
        Execute inference request using gRPC interface hosting multiple models

        <b>input data</b>
        - directory with 2 models in IR format
        - docker image

        <b>fixtures used</b>
        - model downloader
        - input data downloader
        - service launching

        <b>Expected results</b>
        - response contains proper response about model metadata for both
        models set in config file:
        model resnet_v1_50, pnasnet_large
        - both served models handles appropriate input formats

        """
        print("Downloaded model files:", download_two_models)

        print("Start Ovms image: ", start_server_multi_model)

        # Connect to grpc service
        stub = create_channel_for_port_multi_server

        print("Getting info about resnet model")
        model_name = 'resnet_V1_50'
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [1, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [1, 1000]}}
        request = get_model_metadata(model_name='resnet_V1_50')
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

        model_name = 'pnasnet_large'
        out_name = 'final_layer/predictions'
        request = get_model_metadata(model_name='pnasnet_large')
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [4, 3, 331, 331]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [4, 1001]}}
        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata
