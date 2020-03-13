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
from utils.grpc import infer
from utils.rest import infer_rest

auto_shapes = [
    {'in': (1, 3, 300, 300), 'out': (1, 1, 200, 7)},
    {'in': (1, 3, 500, 500), 'out': (1, 1, 200, 7)},
    {'in': (1, 3, 224, 224), 'out': (1, 1, 200, 7)},
    {'in': (4, 3, 224, 224), 'out': (1, 1, 800, 7)},
    {'in': (8, 3, 312, 142), 'out': (1, 1, 1600, 7)},
    {'in': (1, 3, 1024, 1024), 'out': (1, 1, 200, 7)},
]

fixed_shape = {'in': (1, 3, 600, 600), 'out': (1, 1, 200, 7)}


class TestModelReshaping:

    def test_single_local_model_reshaping_auto(
            self, face_detection_model_downloader,
            start_server_face_detection_model_auto_shape,
            create_grpc_channel):

        print("Downloaded model files:", face_detection_model_downloader)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9010', PREDICTION_SERVICE)

        out_name = 'detection_out'
        model_name = 'face_detection'
        for shape in auto_shapes:
            imgs = np.zeros(shape['in'])
            self.run_inference_grpc(imgs, out_name, shape['out'],
                                    True, model_name, stub)

    @pytest.mark.parametrize("shape, is_correct",
                             [(fixed_shape['in'], True), ((1, 3, 300, 300),
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
        model_name = 'face_detection'
        for stub in stubs:
            imgs = np.zeros(shape)
            self.run_inference_grpc(imgs, out_name, fixed_shape['out'],
                                    is_correct, model_name, stub)

    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_single_local_model_reshaping_auto_rest(
            self, face_detection_model_downloader,
            start_server_face_detection_model_auto_shape, request_format):

        print("Downloaded model files:", face_detection_model_downloader)
        out_name = 'detection_out'
        for shape in auto_shapes:
            imgs = np.zeros(shape['in'])
            rest_url = 'http://localhost:5565/v1/models/face_detection' \
                       ':predict'
            self.run_inference_rest(imgs, out_name, shape['out'], True,
                                    request_format, rest_url)

    @pytest.mark.parametrize("shape, is_correct",
                             [(fixed_shape['in'], True), ((1, 3, 300, 300),
                                                          False)])
    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_single_local_model_reshaping_fixed_rest(
            self, face_detection_model_downloader,
            start_server_face_detection_model_named_shape,
            start_server_face_detection_model_nonamed_shape,
            shape, is_correct, request_format):

        print("Downloaded model files:", face_detection_model_downloader)

        out_name = 'detection_out'
        rest_ports = ['5566', '5567']
        for rest_port in rest_ports:
            imgs = np.zeros(shape)
            rest_url = 'http://localhost:{}/v1/models/face_detection:predict' \
                .format(rest_port)
            self.run_inference_rest(imgs, out_name, fixed_shape['out'],
                                    is_correct, request_format, rest_url)

    def test_multi_local_model_reshaping_auto(
            self, face_detection_model_downloader,
            start_server_multi_model,
            create_grpc_channel):

        print("Downloaded model files:", face_detection_model_downloader)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9001', PREDICTION_SERVICE)

        out_name = 'detection_out'
        model_name = 'face_detection_auto'
        for shape in auto_shapes:
            imgs = np.zeros(shape['in'])
            self.run_inference_grpc(imgs, out_name, shape['out'], True,
                                    model_name, stub)

    @pytest.mark.parametrize("shape, is_correct",
                             [(fixed_shape['in'], True), ((1, 3, 300, 300),
                                                          False)])
    def test_multi_local_model_reshaping_fixed(
            self, face_detection_model_downloader,
            start_server_multi_model,
            create_grpc_channel, shape, is_correct):

        print("Downloaded model files:", face_detection_model_downloader)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9001', PREDICTION_SERVICE)

        models_names = ["face_detection_fixed_nonamed",
                        "face_detection_fixed_named"]

        out_name = 'detection_out'

        imgs = np.zeros(shape)
        for model_name in models_names:
            self.run_inference_grpc(imgs, out_name, fixed_shape['out'],
                                    is_correct, model_name, stub)

    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_mutli_local_model_reshaping_auto_rest(
            self, face_detection_model_downloader,
            start_server_multi_model, request_format):

        print("Downloaded model files:", face_detection_model_downloader)
        out_name = 'detection_out'
        for shape in auto_shapes:
            imgs = np.zeros(shape['in'])
            rest_url = 'http://localhost:5561/v1/models/face_detection_auto' \
                       ':predict'
            self.run_inference_rest(imgs, out_name, shape['out'], True,
                                    request_format, rest_url)

    @pytest.mark.parametrize("shape, is_correct",
                             [(fixed_shape['in'], True), ((1, 3, 300, 300),
                                                          False)])
    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_multi_local_model_reshaping_fixed_rest(
            self, face_detection_model_downloader,
            start_server_multi_model, shape, is_correct, request_format):

        print("Downloaded model files:", face_detection_model_downloader)

        models_names = ["face_detection_fixed_nonamed",
                        "face_detection_fixed_named"]
        out_name = 'detection_out'
        imgs = np.zeros(shape)
        for model_name in models_names:
            rest_url = 'http://localhost:5561/v1/models/{}' \
                       ':predict'.format(model_name)
            self.run_inference_rest(imgs, out_name, fixed_shape['out'],
                                    is_correct, request_format, rest_url)

    def run_inference_rest(self, imgs, out_name, out_shape, is_correct,
                           request_format, rest_url):
        if is_correct:
            output = infer_rest(imgs, input_tensor='data',
                                rest_url=rest_url,
                                output_tensors=[out_name],
                                request_format=request_format)
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == out_shape, \
                ERROR_SHAPE
        else:
            output = infer_rest(imgs, input_tensor='data',
                                rest_url=rest_url,
                                output_tensors=[out_name],
                                request_format=request_format)
            assert not output

    def run_inference_grpc(self, imgs, out_name, out_shape, is_correct,
                           model_name, stub):
        if is_correct:
            output = infer(imgs, input_tensor='data', grpc_stub=stub,
                           model_spec_name=model_name,
                           model_spec_version=None,
                           output_tensors=[out_name])
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == out_shape, \
                ERROR_SHAPE
        else:
            with pytest.raises(Exception):
                infer(imgs, input_tensor='data', grpc_stub=stub,
                      model_spec_name=model_name,
                      model_spec_version=None,
                      output_tensors=[out_name])
