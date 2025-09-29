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
from tests.functional.constants.constants import ERROR_SHAPE, NOT_TO_BE_REPORTED_IF_SKIPPED, TARGET_DEVICE_HDDL, \
    TARGET_DEVICE_MYRIAD, TARGET_DEVICE_CUDA, TARGET_DEVICE_GPU
from tests.functional.config import skip_nginx_test
from tests.functional.conftest import devices_not_supported_for_test
from tests.functional.model.models_information import FaceDetection
from tests.functional.utils.grpc import create_channel, infer
import logging
from tests.functional.utils.rest import get_predict_url, infer_rest

logger = logging.getLogger(__name__)

auto_shapes = [
    {'in': (1, 3, 300, 300), 'out': (1, 1, 200, 7)},
    {'in': (1, 3, 500, 500), 'out': (1, 1, 200, 7)},
    {'in': (1, 3, 224, 224), 'out': (1, 1, 200, 7)},
    {'in': (4, 3, 224, 224), 'out': (1, 1, 800, 7)},
    {'in': (8, 3, 312, 142), 'out': (1, 1, 1600, 7)},
    {'in': (1, 3, 1024, 1024), 'out': (1, 1, 200, 7)},
]

fixed_shape = {'in': (1, 3, 600, 600), 'out': (1, 1, 200, 7)}


@pytest.mark.skipif(skip_nginx_test, reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
@devices_not_supported_for_test([TARGET_DEVICE_MYRIAD, TARGET_DEVICE_HDDL, TARGET_DEVICE_CUDA, TARGET_DEVICE_GPU])
class TestModelReshaping:

    @pytest.mark.api_enabling
    def test_single_local_model_reshaping_auto(self, start_server_face_detection_model_auto_shape):

        _, ports = start_server_face_detection_model_auto_shape

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])

        for shape in auto_shapes:
            imgs = np.zeros(shape['in'], FaceDetection.dtype)
            self.run_inference_grpc(imgs, FaceDetection.output_name, shape['out'],
                                    True, FaceDetection.name, stub)

    @pytest.mark.parametrize("shape, is_correct",
                             [(fixed_shape['in'], True), (FaceDetection.input_shape,
                                                          False)])
    @pytest.mark.api_enabling
    def test_single_local_model_reshaping_fixed(self, start_server_face_detection_model_named_shape,
                                                start_server_face_detection_model_nonamed_shape, shape, is_correct):

        _, ports_named = start_server_face_detection_model_named_shape
        _, ports_nonamed = start_server_face_detection_model_nonamed_shape

        # Connect to grpc service
        stubs = [create_channel(port=ports_named["grpc_port"]), create_channel(port=ports_nonamed["grpc_port"])]
        imgs = np.zeros(shape, FaceDetection.dtype)

        for stub in stubs:
            self.run_inference_grpc(imgs, FaceDetection.output_name, fixed_shape['out'],
                                    is_correct, FaceDetection.name, stub)

    @pytest.mark.parametrize("request_format",
                             ['row_name', 'row_noname',
                              'column_name', 'column_noname'])
    @pytest.mark.api_enabling
    def test_single_local_model_reshaping_auto_rest(self, start_server_face_detection_model_auto_shape, request_format):

        _, ports = start_server_face_detection_model_auto_shape

        for shape in auto_shapes:
            imgs = np.zeros(shape['in'], FaceDetection.dtype)
            rest_url = get_predict_url(model="face_detection", port=ports["rest_port"])
            self.run_inference_rest(imgs, FaceDetection.output_name, shape['out'], True,
                                    request_format, rest_url)

    @pytest.mark.parametrize("shape, is_correct",
                             [(fixed_shape['in'], True), (FaceDetection.input_shape,
                                                          False)])
    @pytest.mark.parametrize("request_format",
                             ['row_name', 'row_noname',
                              'column_name', 'column_noname'])
    @pytest.mark.api_enabling
    def test_single_local_model_reshaping_fixed_rest(self, start_server_face_detection_model_named_shape,
                                                     start_server_face_detection_model_nonamed_shape, shape, is_correct,
                                                     request_format):

        _, ports_named = start_server_face_detection_model_named_shape
        _, ports_nonamed = start_server_face_detection_model_nonamed_shape

        imgs = np.zeros(shape, FaceDetection.dtype)
        rest_ports = [ports_named["rest_port"], ports_nonamed["rest_port"]]
        for rest_port in rest_ports:
            rest_url = get_predict_url(model="face_detection", port=rest_port)
            self.run_inference_rest(imgs, FaceDetection.output_name, fixed_shape['out'],
                                    is_correct, request_format, rest_url)

    @pytest.mark.api_enabling
    def test_multi_local_model_reshaping_auto(self, start_server_multi_model):

        _, ports = start_server_multi_model

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])

        for shape in auto_shapes:
            imgs = np.zeros(shape['in'], FaceDetection.dtype)
            self.run_inference_grpc(imgs, FaceDetection.output_name, shape['out'], True,
                                    "face_detection_auto", stub)

    @pytest.mark.parametrize("shape, is_correct",
                             [(fixed_shape['in'], True), (FaceDetection.input_shape,
                                                          False)])
    @pytest.mark.api_enabling
    def test_multi_local_model_reshaping_fixed(self, start_server_multi_model, shape, is_correct):

        _, ports = start_server_multi_model

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])

        models_names = ["face_detection_fixed_nonamed",
                        "face_detection_fixed_named"]

        imgs = np.zeros(shape, FaceDetection.dtype)
        for model_name in models_names:
            self.run_inference_grpc(imgs, FaceDetection.output_name, fixed_shape['out'],
                                    is_correct, model_name, stub)

    @pytest.mark.parametrize("request_format",
                             ['row_name', 'row_noname',
                              'column_name', 'column_noname'])
    @pytest.mark.api_enabling
    def test_mutli_local_model_reshaping_auto_rest(self, start_server_multi_model, request_format):

        _, ports = start_server_multi_model

        for shape in auto_shapes:
            imgs = np.zeros(shape['in'], FaceDetection.dtype)
            rest_url = get_predict_url(model="face_detection_auto", port=ports["rest_port"])
            self.run_inference_rest(imgs, FaceDetection.output_name, shape['out'], True,
                                    request_format, rest_url)

    @pytest.mark.parametrize("shape, is_correct",
                             [(fixed_shape['in'], True), (FaceDetection.input_shape,
                                                          False)])
    @pytest.mark.parametrize("request_format",
                             ['row_name', 'row_noname',
                              'column_name', 'column_noname'])
    @pytest.mark.api_enabling
    def test_multi_local_model_reshaping_fixed_rest(self, start_server_multi_model, shape, is_correct, request_format):

        _, ports = start_server_multi_model

        models_names = ["face_detection_fixed_nonamed",
                        "face_detection_fixed_named"]
        imgs = np.zeros(shape, FaceDetection.dtype)
        for model_name in models_names:
            rest_url = get_predict_url(model=model_name, port=ports["rest_port"])
            self.run_inference_rest(imgs, FaceDetection.output_name, fixed_shape['out'],
                                    is_correct, request_format, rest_url)

    @staticmethod
    def run_inference_rest(imgs, out_name, out_shape, is_correct,
                           request_format, rest_url):
        logger.info("Running rest inference call")
        output = infer_rest(imgs, input_tensor='data',
                            rest_url=rest_url,
                            output_tensors=[out_name],
                            request_format=request_format,
                            raise_error=is_correct)
        if is_correct:
            logger.info("Output shape: {}".format(output[out_name].shape))
            assert output[out_name].shape == out_shape, ERROR_SHAPE
        else:
            assert not output

    @staticmethod
    def run_inference_grpc(imgs, out_name, out_shape, is_correct, model_name, stub):
        logger.info(f"Running grpc inference call")
        if is_correct:
            output = infer(imgs, input_tensor=FaceDetection.input_name, grpc_stub=stub,
                           model_spec_name=model_name,
                           model_spec_version=None,
                           output_tensors=[out_name])
            logger.info("Output shape: {}".format(output[out_name].shape))
            assert output[out_name].shape == out_shape, ERROR_SHAPE
        else:
            with pytest.raises(Exception):
                infer(imgs, input_tensor=FaceDetection.input_name, grpc_stub=stub,
                      model_spec_name=model_name,
                      model_spec_version=None,
                      output_tensors=[out_name])
