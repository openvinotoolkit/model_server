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
import sys

import numpy as np
from constants import PREDICTION_SERVICE, MODEL_SERVICE
from utils.grpc import infer, get_model_metadata, model_metadata_response, \
    get_model_status
from utils.rest import infer_rest, get_model_metadata_response_rest, \
    get_model_status_response_rest

sys.path.append(".")
from ie_serving.models.models_utils import ModelVersionState, _ERROR_MESSAGE, \
    ErrorCode  # noqa


class TestModelVersionHandling():
    model_name = "pvb_face_multi_version"

    def test_run_inference(self, download_two_model_versions,
                           start_server_multi_model,
                           create_grpc_channel):
        print("Downloaded model files:", download_two_model_versions)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9001', PREDICTION_SERVICE)

        face_img = np.ones((1, 3, 300, 300))
        pvb_img = np.ones((1, 3, 1024, 1024))

        out_name = "detection_out"
        in_name = "data"
        output = infer(face_img, input_tensor=in_name,
                       grpc_stub=stub, model_spec_name=self.model_name,
                       model_spec_version=1,  # face detection
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1, 200, 7), \
            '{} with version 1 has invalid output'.format(self.model_name)

        output = infer(pvb_img, input_tensor=in_name,
                       grpc_stub=stub,
                       model_spec_name='pvb_face_multi_version',
                       model_spec_version=None,  # PVB detection
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1, 200, 7), \
            '{} with version latest has invalid output'.format(self.model_name)

    def test_get_model_metadata(self, download_two_model_versions,
                                start_server_multi_model,
                                create_grpc_channel):

        print("Downloaded model files:", download_two_model_versions)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9001', PREDICTION_SERVICE)
        versions = [None, 1]

        expected_inputs_metadata = \
            [{'data': {'dtype': 1, 'shape': [1, 3, 1024, 1024]}},
             # PVB detection
             {'data': {'dtype': 1, 'shape': [1, 3, 300, 300]}}
             # face detection
             ]
        # Same output shape for both versions
        expected_output_metadata = {
            'detection_out': {'dtype': 1, 'shape': [1, 1, 200, 7]}
        }
        for i in range(len(versions)):
            print("Getting info about pvb_face_detection model "
                  "version:".format(versions[i]))
            expected_input_metadata = expected_inputs_metadata[i]
            request = get_model_metadata(model_name=self.model_name,
                                         version=versions[i])
            response = stub.GetModelMetadata(request, 10)
            input_metadata, output_metadata = model_metadata_response(
                response=response)

            print(output_metadata)
            assert self.model_name == response.model_spec.name
            assert expected_input_metadata == input_metadata
            assert expected_output_metadata == output_metadata

    def test_get_model_status(self, download_two_model_versions,
                              start_server_multi_model,
                              create_grpc_channel):

        print("Downloaded model files:", download_two_model_versions)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9001', MODEL_SERVICE)
        versions = [None, 1]
        for x in range(len(versions)):
            request = get_model_status(model_name=self.model_name,
                                       version=versions[x])
            response = stub.GetModelStatus(request, 10)

            versions_statuses = response.model_version_status
            version_status = versions_statuses[0]
            if x == 0:
                assert len(versions_statuses) == 2
            else:
                assert version_status.version == 1
            assert version_status.state == ModelVersionState.AVAILABLE
            assert version_status.status.error_code == ErrorCode.OK
            assert version_status.status.error_message == _ERROR_MESSAGE[
                ModelVersionState.AVAILABLE][ErrorCode.OK]

    def test_run_inference_rest(self, download_two_model_versions,
                                start_server_multi_model):
        print("Downloaded model files:", download_two_model_versions)

        face_img = np.ones((1, 3, 300, 300))
        pvb_img = np.ones((1, 3, 1024, 1024))
        out_name = "detection_out"

        in_name = "data"
        rest_url = 'http://localhost:5561/v1/models/{}' \
                   '/versions/1:predict'.format(self.model_name)
        output = infer_rest(face_img,
                            input_tensor=in_name, rest_url=rest_url,
                            output_tensors=[out_name],
                            request_format='column_name')
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1, 200, 7), \
            '{} with version 1 has invalid output'.format(self.model_name)

        rest_url = 'http://localhost:5561/v1/models/{}:predict'.format(
            self.model_name)
        output = infer_rest(pvb_img,
                            input_tensor=in_name, rest_url=rest_url,
                            output_tensors=[out_name],
                            request_format='column_name')
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1, 200, 7), \
            '{} with version latest has invalid output'.format(self.model_name)

        # both model versions use the same input data shape

    def test_get_model_metadata_rest(self, download_two_model_versions,
                                     start_server_multi_model):

        print("Downloaded model files:", download_two_model_versions)

        urls = ['http://localhost:5561/v1/models/{}'
                '/metadata'.format(self.model_name),
                'http://localhost:5561/v1/models/{}'
                '/versions/1/metadata'.format(self.model_name)]

        expected_inputs_metadata = \
            [{'data': {'dtype': 1, 'shape': [1, 3, 1024, 1024]}},
             # PVB detection
             {'data': {'dtype': 1, 'shape': [1, 3, 300, 300]}}
             # face detection
             ]
        # Same output shape for both versions
        expected_output_metadata = {
            'detection_out': {'dtype': 1, 'shape': [1, 1, 200, 7]}
        }
        for i in range(len(urls)):
            print("Getting info about resnet model version:".format(
                urls[i]))
            expected_input_metadata = expected_inputs_metadata[i]
            response = get_model_metadata_response_rest(urls[i])
            input_metadata, output_metadata = model_metadata_response(
                response=response)

            print(output_metadata)
            assert self.model_name == response.model_spec.name
            assert expected_input_metadata == input_metadata
            assert expected_output_metadata == output_metadata

    def test_get_model_status_rest(self, download_two_model_versions,
                                   start_server_multi_model):

        print("Downloaded model files:", download_two_model_versions)

        urls = ['http://localhost:5561/v1/models/{}'.format(self.model_name),
                'http://localhost:5561/v1/models/{}'
                '/versions/1'.format(self.model_name)]

        for x in range(len(urls)):
            response = get_model_status_response_rest(urls[x])
            versions_statuses = response.model_version_status
            version_status = versions_statuses[0]
            if x == 0:
                assert len(versions_statuses) == 2
            else:
                assert version_status.version == 1
            assert version_status.state == ModelVersionState.AVAILABLE
            assert version_status.status.error_code == ErrorCode.OK
            assert version_status.status.error_message == _ERROR_MESSAGE[
                ModelVersionState.AVAILABLE][ErrorCode.OK]
