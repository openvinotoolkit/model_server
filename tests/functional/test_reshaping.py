#
# Copyright (c) 2019 Intel Corporation
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
import pytest
from constants import PREDICTION_SERVICE, ERROR_SHAPE
from utils.grpc import infer_batch
from utils.rest import infer_batch_rest

shapes = [
    {'in': (1, 3, 300, 300), 'out': (1, 1, 200, 7)},
    {'in': (1, 3, 500, 500), 'out': (1, 1, 200, 7)},
    {'in': (1, 3, 224, 224), 'out': (1, 1, 200, 7)},
    {'in': (4, 3, 224, 224), 'out': (1, 1, 800, 7)},
    {'in': (8, 3, 312, 142), 'out': (1, 1, 1600, 7)},
    {'in': (1, 3, 1024, 1024), 'out': (1, 1, 200, 7)},
]

fixed_input_shape = (1, 3, 600, 600)
fixed_output_shape = (1, 1, 200, 7)


class TestModelReshaping:

    def test_single_local_model_reshaping_auto(
            self, face_detection_model_downloader,
            start_server_face_detection_model_auto_shape,
            create_grpc_channel):

        print("Downloaded model files:", face_detection_model_downloader)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9010', PREDICTION_SERVICE)

        out_name = 'detection_out'
        for shape in shapes:
            imgs = np.zeros(shape['in'])
            output = infer_batch(batch_input=imgs,
                                 input_tensor='data', grpc_stub=stub,
                                 model_spec_name='face_detection',
                                 model_spec_version=None,
                                 output_tensors=[out_name])
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == shape['out'], ERROR_SHAPE

    @pytest.mark.skip(reason="Fixed shape option not yet merged")
    @pytest.mark.parametrize("shape, is_correct",
                             [(fixed_input_shape, True), ((1, 3, 300, 300),
                                                          False)])
    def test_single_local_model_reshaping_fixed(
            self, face_detection_model_downloader,
            start_server_face_detection_model_named_shape,
            start_server_face_detection_model_nonamed_shape,
            create_grpc_channel, shape, is_correct):

        print("Downloaded model files:", face_detection_model_downloader)

        # Connect to grpc service
        stubs = [create_grpc_channel('localhost:9011', PREDICTION_SERVICE),
                 create_grpc_channel('localhost:9012', PREDICTION_SERVICE)]

        out_name = 'detection_out'

        for stub in stubs:
            imgs = np.zeros(shape)
            if is_correct:
                output = infer_batch(batch_input=imgs,
                                     input_tensor='data', grpc_stub=stub,
                                     model_spec_name='face_detection',
                                     model_spec_version=None,
                                     output_tensors=[out_name])
                print("output shape", output[out_name].shape)
                assert output[out_name].shape == fixed_output_shape, \
                    ERROR_SHAPE
            else:
                with pytest.raises(Exception):
                    infer_batch(batch_input=imgs,
                                input_tensor='data', grpc_stub=stub,
                                model_spec_name='face_detection',
                                model_spec_version=None,
                                output_tensors=[out_name])

    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_single_local_model_reshaping_auto_rest(
            self, face_detection_model_downloader,
            start_server_face_detection_model_auto_shape, request_format):

        print("Downloaded model files:", face_detection_model_downloader)
        out_name = 'detection_out'
        for shape in shapes:
            imgs = np.zeros(shape['in'])
            rest_url = 'http://localhost:5565/v1/models/face_detection' \
                       ':predict'
            output = infer_batch_rest(batch_input=imgs,
                                      input_tensor='data', rest_url=rest_url,
                                      output_tensors=[out_name],
                                      request_format=request_format)
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == shape['out'], ERROR_SHAPE

    def test_multi_local_model_reshaping_auto(
            self, face_detection_model_downloader,
            start_server_multi_model,
            create_grpc_channel):

        print("Downloaded model files:", face_detection_model_downloader)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9001', PREDICTION_SERVICE)

        out_name = 'detection_out'
        for shape in shapes:
            imgs = np.zeros(shape['in'])
            output = infer_batch(batch_input=imgs,
                                 input_tensor='data', grpc_stub=stub,
                                 model_spec_name='face_detection_auto',
                                 model_spec_version=None,
                                 output_tensors=[out_name])
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == shape['out'], ERROR_SHAPE

    @pytest.mark.skip(reason="Fixed shape option not yet merged")
    @pytest.mark.parametrize("shape, is_correct",
                             [(fixed_input_shape, True), ((1, 3, 300, 300),
                              False)])
    def test_multi_local_model_reshaping_fixed(
            self, face_detection_model_downloader,
            start_server_multi_model,
            create_grpc_channel, shape, is_correct):

        print("Downloaded model files:", face_detection_model_downloader)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9001', PREDICTION_SERVICE)

        models_names = ["face_detection_fixed_noname",
                        "face_detection_fixed_name"]

        out_name = 'detection_out'

        imgs = np.zeros(shape)
        for model_name in models_names:
            if is_correct:
                output = infer_batch(batch_input=imgs,
                                     input_tensor='data', grpc_stub=stub,
                                     model_spec_name=model_name,
                                     model_spec_version=None,
                                     output_tensors=[out_name])
                print("output shape", output[out_name].shape)
                assert output[out_name].shape == fixed_output_shape, \
                    ERROR_SHAPE
            else:
                with pytest.raises(Exception):
                    infer_batch(batch_input=imgs,
                                input_tensor='data', grpc_stub=stub,
                                model_spec_name=model_name,
                                model_spec_version=None,
                                output_tensors=[out_name])

    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_mutli_local_model_reshaping_auto_rest(
            self, face_detection_model_downloader,
            start_server_multi_model, request_format):

        print("Downloaded model files:", face_detection_model_downloader)
        out_name = 'detection_out'
        for shape in shapes:
            imgs = np.zeros(shape['in'])
            rest_url = 'http://localhost:5561/v1/models/face_detection_auto' \
                       ':predict'
            output = infer_batch_rest(batch_input=imgs,
                                      input_tensor='data', rest_url=rest_url,
                                      output_tensors=[out_name],
                                      request_format=request_format)
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == shape['out'], ERROR_SHAPE
