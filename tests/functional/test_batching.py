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
import os
from tests.functional.constants.constants import ERROR_SHAPE, TARGET_DEVICE_MYRIAD, NOT_TO_BE_REPORTED_IF_SKIPPED, \
    TARGET_DEVICE_HDDL
from tests.functional.config import skip_nginx_test
from tests.functional.conftest import devices_not_supported_for_test
from tests.functional.model.models_information import ResnetBS8, AgeGender
from tests.functional.utils.grpc import create_channel, infer, get_model_metadata_request, get_model_metadata, \
    model_metadata_response
import logging
from tests.functional.utils.rest import get_predict_url, get_metadata_url, infer_rest, get_model_metadata_response_rest

logger = logging.getLogger(__name__)


@pytest.mark.skipif(skip_nginx_test, reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
class TestBatchModelInference:

    @pytest.fixture()
    def mapping_names(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, "mapping_config.json")
        with open(config_path, 'r') as f:
            json_string = f.read()
            try:
                json_dict = json.loads(json_string)
            except ValueError as e:
                logger.error("Error while loading json: {}".format(json_string))
                raise e

        in_name = list(json_dict["inputs"].keys())[0]
        out_names = list(json_dict["outputs"].keys())
        return in_name, out_names, json_dict["outputs"]

    @devices_not_supported_for_test([TARGET_DEVICE_HDDL])
    @pytest.mark.api_enabling
    def test_run_inference(self, start_server_batch_model):
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

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])

        batch_input = np.ones(ResnetBS8.input_shape, ResnetBS8.dtype)
        output = infer(batch_input, input_tensor=ResnetBS8.input_name,
                       grpc_stub=stub, model_spec_name=ResnetBS8.name,
                       model_spec_version=None,
                       output_tensors=[ResnetBS8.output_name])
        logger.info("Output shape: {}".format(output[ResnetBS8.output_name].shape))
        assert output[ResnetBS8.output_name].shape == ResnetBS8.output_shape, ERROR_SHAPE

    @pytest.mark.api_enabling
    def test_get_model_metadata(self, start_server_batch_model):

        _, ports = start_server_batch_model

        stub = create_channel(port=ports["grpc_port"])

        logger.info("Getting info about {} model".format(ResnetBS8.name))
        expected_input_metadata = {ResnetBS8.input_name: {'dtype': 1, 'shape': list(ResnetBS8.input_shape)}}
        expected_output_metadata = {ResnetBS8.output_name: {'dtype': 1, 'shape': list(ResnetBS8.output_shape)}}
        request = get_model_metadata_request(model_name=ResnetBS8.name)
        response = get_model_metadata(stub, request)
        input_metadata, output_metadata = model_metadata_response(response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert response.model_spec.name == ResnetBS8.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    @devices_not_supported_for_test([TARGET_DEVICE_MYRIAD])
    @pytest.mark.parametrize("request_format",
                             ['row_name', 'row_noname',
                              'column_name', 'column_noname'])
    @pytest.mark.api_enabling
    def test_run_inference_rest(self, start_server_batch_model_2out, mapping_names, request_format):
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

        in_name, out_names, out_mapping = mapping_names

        batch_input = np.ones(AgeGender.input_shape, AgeGender.dtype)
        rest_url = get_predict_url(model=AgeGender.name, port=ports["rest_port"])
        output = infer_rest(batch_input, input_tensor=in_name,
                            rest_url=rest_url,
                            output_tensors=out_names,
                            request_format=request_format)
        for output_names in out_names:
            assert output[output_names].shape == AgeGender.output_shape[out_mapping[output_names]], ERROR_SHAPE

    @pytest.mark.api_enabling
    def test_get_model_metadata_rest(self, resnet_multiple_batch_sizes,
                                     start_server_batch_model):

        _, ports = start_server_batch_model
        logger.info("Downloaded model files: {}".format(resnet_multiple_batch_sizes))

        logger.info("Getting info about {} model".format(ResnetBS8.name))
        expected_input_metadata = {ResnetBS8.input_name: {'dtype': 1, 'shape': list(ResnetBS8.input_shape)}}
        expected_output_metadata = {ResnetBS8.output_name: {'dtype': 1, 'shape': list(ResnetBS8.output_shape)}}
        rest_url = get_metadata_url(model=ResnetBS8.name, port=ports["rest_port"])
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert response.model_spec.name == ResnetBS8.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata
