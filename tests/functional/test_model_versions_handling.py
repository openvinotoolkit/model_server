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

from constants import PREDICTION_SERVICE, MODEL_SERVICE
from model.models_information import PVBDetection, PVBDetectionV2
from utils.grpc import infer, get_model_metadata, model_metadata_response, \
    get_model_status
from utils.models_utils import ModelVersionState, ErrorCode, \
    ERROR_MESSAGE  # noqa
from utils.rest import infer_rest, get_model_metadata_response_rest, \
    get_model_status_response_rest


class TestModelVersionHandling:
    model_name = "pvb_face_multi_version"

    @pytest.mark.parametrize("version", [1, 2, None], ids=("version 1", "version 2", "no version specified"))
    def test_run_inference(self, download_two_model_versions,
                           start_server_multi_model,
                           create_grpc_channel, version):

        _, ports = start_server_multi_model
        print("Downloaded model files:", download_two_model_versions)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:{}'.format(ports["grpc_port"]),
                                   PREDICTION_SERVICE)
        model_info = PVBDetectionV2 if version is None else PVBDetection[version-1]

        img = np.ones(model_info.input_shape, dtype=model_info.dtype)

        output = infer(img, input_tensor=model_info.input_name,
                       grpc_stub=stub, model_spec_name=self.model_name,
                       model_spec_version=version,  # face detection
                       output_tensors=[model_info.output_name])
        print("output shape", output[model_info.output_name].shape)
        assert output[model_info.output_name].shape == model_info.output_shape, \
            '{} with version 1 has invalid output'.format(self.model_name)

    @pytest.mark.skip(reason="not implemented yet")
    @pytest.mark.parametrize("version", [1, 2, None], ids=("version 1", "version 2", "no version specified"))
    def test_get_model_metadata(self, download_two_model_versions,
                                start_server_multi_model,
                                create_grpc_channel, version):

        _, ports = start_server_multi_model
        print("Downloaded model files:", download_two_model_versions)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:{}'.format(ports["grpc_port"]),
                                   PREDICTION_SERVICE)
        model_info = PVBDetectionV2 if version is None else PVBDetection[version-1]

        print("Getting info about pvb_face_detection model "
              "version:".format("no_version" if version is None else version))
        expected_input_metadata = {model_info.input_name: {'dtype': 1, 'shape': list(model_info.input_shape)}}
        expected_output_metadata = {model_info.output_name: {'dtype': 1, 'shape': list(model_info.output_shape)}}

        request = get_model_metadata(model_name=self.model_name,
                                     version=version)
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert response.model_spec.name == self.model_name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    @pytest.mark.skip(reason="not implemented yet")
    @pytest.mark.parametrize("version", [1, 2, None], ids=("version 1", "version 2", "no version specified"))
    def test_get_model_status(self, download_two_model_versions,
                              start_server_multi_model,
                              create_grpc_channel, version):

        _, ports = start_server_multi_model
        print("Downloaded model files:", download_two_model_versions)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:{}'.format(ports["grpc_port"]),
                                   MODEL_SERVICE)
        request = get_model_status(model_name=self.model_name,
                                   version=version)
        response = stub.GetModelStatus(request, 10)

        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        if version is None:
            assert len(versions_statuses) == 2
        else:
            assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]

    @pytest.mark.skip(reason="not implemented yet")
    @pytest.mark.parametrize("version", [1, 2, None], ids=("version 1", "version 2", "no version specified"))
    def test_run_inference_rest(self, download_two_model_versions,
                                start_server_multi_model, version):

        _, ports = start_server_multi_model
        print("Downloaded model files:", download_two_model_versions)

        model_info = PVBDetectionV2 if version is None else PVBDetection[version-1]

        img = np.ones(model_info.input_shape, dtype=model_info.dtype)
        if version is None:
            rest_url = 'http://localhost:{}/v1/models/{}:predict'.format(ports["rest_port"], self.model_name)
        else:
            rest_url = 'http://localhost:{}/v1/models/{}' \
                       '/versions/{}:predict'.format(ports["rest_port"], self.model_name, version)
        output = infer_rest(img,
                            input_tensor=model_info.input_name, rest_url=rest_url,
                            output_tensors=[model_info.output_name],
                            request_format='column_name')
        print("output shape", output[model_info.output_name].shape)
        assert output[model_info.output_name].shape == model_info.output_shape, \
            '{} with version 1 has invalid output'.format(self.model_name)

    @pytest.mark.skip(reason="not implemented yet")
    @pytest.mark.parametrize("version", [1, 2, None], ids=("version 1", "version 2", "no version specified"))
    def test_get_model_metadata_rest(self, download_two_model_versions,
                                     start_server_multi_model, version):

        _, ports = start_server_multi_model
        print("Downloaded model files:", download_two_model_versions)
        model_info = PVBDetectionV2 if version is None else PVBDetection[version-1]

        if version is None:
            rest_url = 'http://localhost:{}/v1/models/{}/metadata'.format(ports["rest_port"], self.model_name)
        else:
            rest_url = 'http://localhost:{}/v1/models/{}/versions/{}/metadata'.format(ports["rest_port"],
                                                                                      self.model_name, version)

        expected_input_metadata = {model_info.input_name: {'dtype': 1, 'shape': list(model_info.input_shape)}}
        expected_output_metadata = {model_info.output_name: {'dtype': 1, 'shape': list(model_info.output_shape)}}
        print("Getting info about resnet model version:".format(rest_url))
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(response=response)

        print(output_metadata)
        assert response.model_spec.name == self.model_name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    @pytest.mark.skip(reason="not implemented yet")
    @pytest.mark.parametrize("version", [1, 2, None], ids=("version 1", "version 2", "no version specified"))
    def test_get_model_status_rest(self, download_two_model_versions,
                                   start_server_multi_model, version):

        _, ports = start_server_multi_model
        print("Downloaded model files:", download_two_model_versions)

        if version is None:
            rest_url = 'http://localhost:{}/v1/models/{}'.format(ports["rest_port"], self.model_name)
        else:
            rest_url = 'http://localhost:{}/v1/models/{}/versions/{}'.format(ports["rest_port"],
                                                                             self.model_name, version)

        response = get_model_status_response_rest(rest_url)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        if version is None:
            assert len(versions_statuses) == 2
        else:
            assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]
