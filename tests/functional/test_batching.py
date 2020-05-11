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
import pytest
import numpy as np
import json
from constants import ERROR_SHAPE
from model.models_information import ResnetBS8, AgeGender
from utils.grpc import create_channel, infer, get_model_metadata, model_metadata_response
from utils.rest import infer_rest, get_model_metadata_response_rest


class TestBatchModelInference:

    @pytest.fixture()
    def mapping_names(self):
        with open("mapping_config.json", 'r') as f:
            json_string = f.read()
            try:
                json_dict = json.loads(json_string)
                return json_dict
            except ValueError as e:
                print("Error while loading json: {}".format(json_string))
                raise e

        in_name = list(json_dict["inputs"].keys())[0]
        out_names = list(json_dict["outputs"].keys())
        return in_name, out_names, json_dict["outputs"]

    def test_run_inference(self, resnet_multiple_batch_sizes,
                           start_server_batch_model):
        """
        <b>Description</b>
        Submit request to gRPC interface serving a single resnet model

        <b>input data</b>
        - directory with the model in IR format
        - docker image with ie-serving-py service

        <b>fixtures used</b>
        - model downloader
        - service launching

        <b>Expected results</b>
        - response contains proper numpy shape

        """

        _, ports = start_server_batch_model
        print("Downloaded model files:", resnet_multiple_batch_sizes)

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])

        batch_input = np.ones(ResnetBS8.input_shape, ResnetBS8.dtype)
        output = infer(batch_input, input_tensor=ResnetBS8.input_name,
                       grpc_stub=stub, model_spec_name=ResnetBS8.name,
                       model_spec_version=None,
                       output_tensors=[ResnetBS8.output_name])
        print("output shape", output[ResnetBS8.output_name].shape)
        assert output[ResnetBS8.output_name].shape == ResnetBS8.output_shape, ERROR_SHAPE

    def test_run_inference_bs4(self, resnet_multiple_batch_sizes,
                               start_server_batch_model_bs4):

        _, ports = start_server_batch_model_bs4
        print("Downloaded model files:", resnet_multiple_batch_sizes)

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])

        batch_input = np.ones((4,) + ResnetBS8.input_shape[1:], ResnetBS8.dtype)
        output = infer(batch_input, input_tensor=ResnetBS8.input_name,
                       grpc_stub=stub, model_spec_name=ResnetBS8.name,
                       model_spec_version=None,
                       output_tensors=[ResnetBS8.output_name])
        print("output shape", output[ResnetBS8.output_name].shape)
        assert output[ResnetBS8.output_name].shape == (4,) + ResnetBS8.output_shape[1:], ERROR_SHAPE

    @pytest.mark.skip(reason="not implemented yet")
    def test_run_inference_auto(self, resnet_multiple_batch_sizes,
                                start_server_batch_model_auto):

        _, ports = start_server_batch_model_auto
        print("Downloaded model files:", resnet_multiple_batch_sizes)

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])

        for batch_size in [1, 6]:
            batch_input = np.ones((batch_size,) + ResnetBS8.input_shape[1:], ResnetBS8.dtype)
            output = infer(batch_input, input_tensor=ResnetBS8.input_name,
                           grpc_stub=stub, model_spec_name=ResnetBS8.name,
                           model_spec_version=None,
                           output_tensors=[ResnetBS8.output_name])
            print("output shape", output[ResnetBS8.output_name].shape)
            assert output[ResnetBS8.output_name].shape == (batch_size,) + ResnetBS8.output_shape[1:], ERROR_SHAPE

    def test_get_model_metadata(self, resnet_multiple_batch_sizes,
                                start_server_batch_model):

        _, ports = start_server_batch_model
        print("Downloaded model files:", resnet_multiple_batch_sizes)

        stub = create_channel(port=ports["grpc_port"])

        print("Getting info about {} model".format(ResnetBS8.name))
        expected_input_metadata = {ResnetBS8.input_name: {'dtype': 1, 'shape': list(ResnetBS8.input_shape)}}
        expected_output_metadata = {ResnetBS8.output_name: {'dtype': 1, 'shape': list(ResnetBS8.output_shape)}}
        request = get_model_metadata(model_name=ResnetBS8.name)
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(response=response)

        print(output_metadata)
        assert response.model_spec.name == ResnetBS8.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    @pytest.mark.skip(reason="not implemented yet")
    @pytest.mark.parametrize("request_format",
                             ['row_name', 'row_noname',
                              'column_name', 'column_noname'])
    def test_run_inference_rest(self, age_gender_model_downloader,
                                start_server_batch_model_2out, mapping_names, request_format):
        """
            <b>Description</b>
            Submit request to REST API interface serving
            a single age-gender model with 2 outputs.
            No batch_size parameter specified.

            <b>input data</b>
            - directory with the model in IR format
            - docker image with ie-serving-py service

            <b>fixtures used</b>
            - model downloader
            - service launching

            <b>Expected results</b>
            - response contains proper numpy shape

        """

        _, ports = start_server_batch_model_2out
        print("Downloaded model files:", age_gender_model_downloader)
        in_name, out_names, out_mapping = mapping_names

        batch_input = np.ones(AgeGender.input_shape, AgeGender.dtype)
        rest_url = 'http://localhost:{}/v1/models/age_gender:predict'.format(
                   ports["rest_port"])
        output = infer_rest(batch_input, input_tensor=in_name,
                            rest_url=rest_url,
                            output_tensors=out_names,
                            request_format=request_format)
        for output_names in out_names:
            assert output[output_names].shape == AgeGender.output_shape[out_mapping[out_names]], ERROR_SHAPE

    @pytest.mark.skip(reason="not implemented yet")
    @pytest.mark.parametrize("request_format",
                             ['row_name', 'row_noname',
                              'column_name', 'column_noname'])
    def test_run_inference_bs4_rest(self, age_gender_model_downloader,
                                    start_server_batch_model_auto_bs4_2out,
                                    mapping_names,
                                    request_format):
        """
            <b>Description</b>
            Submit request to REST API interface serving
            a single age-gender model with 2 outputs.
            Parameter batch_size explicitly set to 4.

            <b>input data</b>
            - directory with the model in IR format
            - docker image with ie-serving-py service

            <b>fixtures used</b>
            - model downloader
            - service launching

            <b>Expected results</b>
            - response contains proper numpy shape

        """

        _, ports = start_server_batch_model_auto_bs4_2out
        print("Downloaded model files:", age_gender_model_downloader)

        in_name, out_names, out_mapping = mapping_names

        batch_size = 4
        batch_input = np.ones((batch_size,) + AgeGender.input_shape[1:], AgeGender.dtype)
        rest_url = 'http://localhost:{}/v1/models/age_gender:predict'.format(
                   ports["rest_port"])
        output = infer_rest(batch_input, input_tensor=in_name,
                            rest_url=rest_url,
                            output_tensors=out_names,
                            request_format=request_format)
        for output_names in out_names:
            expected_shape = (batch_size,) + AgeGender.output_shape[out_mapping[out_names]][1:]
            assert output[output_names].shape == expected_shape, ERROR_SHAPE

    @pytest.mark.skip(reason="not implemented yet")
    @pytest.mark.parametrize("request_format",
                             ['row_name', 'row_noname',
                              'column_name', 'column_noname'])
    def test_run_inference_rest_auto(self, age_gender_model_downloader,
                                     start_server_batch_model_auto_2out,
                                     mapping_names,
                                     request_format):
        """
            <b>Description</b>
            Submit request to REST API interface serving a single resnet model.
            Parameter batch_size set to auto.

            <b>input data</b>
            - directory with the model in IR format
            - docker image with ie-serving-py service

            <b>fixtures used</b>
            - model downloader
            - service launching

            <b>Expected results</b>
            - response contains proper numpy shape

        """

        _, ports = start_server_batch_model_auto_2out
        print("Downloaded model files:", age_gender_model_downloader)
        in_name, out_names, out_mapping = mapping_names

        batch_size = 6
        batch_input = np.ones((batch_size,) + AgeGender.input_shape[1:], AgeGender.dtype)
        rest_url = 'http://localhost:{}/v1/models/age_gender:predict'.format(
                   ports["rest_port"])
        output = infer_rest(batch_input,
                            input_tensor=in_name, rest_url=rest_url,
                            output_tensors=out_names,
                            request_format=request_format)
        for output_names in out_names:
            expected_shape = (batch_size,) + AgeGender.output_shape[out_mapping[out_names]][1:]
            assert output[output_names].shape == expected_shape, ERROR_SHAPE

        batch_size = 3
        batch_input = np.ones((batch_size,) + AgeGender.input_shape[1:], AgeGender.dtype)
        output = infer_rest(batch_input, input_tensor=in_name,
                            rest_url=rest_url,
                            output_tensors=out_names,
                            request_format=request_format)
        for output_names in out_names:
            expected_shape = (batch_size,) + AgeGender.output_shape[out_mapping[out_names]][1:]
            assert output[output_names].shape == expected_shape, ERROR_SHAPE

    @pytest.mark.skip(reason="not implemented yet")
    def test_get_model_metadata_rest(self, resnet_multiple_batch_sizes,
                                     start_server_batch_model):

        _, ports = start_server_batch_model
        print("Downloaded model files:", resnet_multiple_batch_sizes)

        print("Getting info about {} model".format(ResnetBS8.name))
        expected_input_metadata = {ResnetBS8.input_name: {'dtype': 1, 'shape': list(ResnetBS8.input_shape)}}
        expected_output_metadata = {ResnetBS8.output_name: {'dtype': 1, 'shape': list(ResnetBS8.output_shape)}}
        rest_url = 'http://localhost:{}/v1/models/{}/metadata'.format(ports["rest_port"], ResnetBS8.name)
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(response=response)

        print(output_metadata)
        assert response.model_spec.name == ResnetBS8.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata
