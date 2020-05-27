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
import cv2
import os
import json
import numpy as np

from constants import PREDICTION_SERVICE, ERROR_SHAPE
from utils.grpc import infer


class TestVehicleDetection():

    def load_image(self, file_path, width, height):
        img = cv2.imread(file_path)  # BGR color format, shape HWC
        img = cv2.resize(img, (width, height))
        img = img.transpose(2, 0, 1).reshape(1, 3, height, width)
        # change shape to NCHW
        return img

    def test_run_inference(self, vehicle_adas_model_downloader,
                           vehicle_adas_data_downloader,
                           start_server_single_vehicle_model,
                           create_grpc_channel):
        """
        <b>Description</b>
        Submit request to gRPC interface serving a single vehicle model

        <b>input data</b>
        - directory with the model in IR format
        - docker image with ie-serving-py service

        <b>fixtures used</b>
        - model downloader
        - service launching

        <b>Expected results</b>
        - response contains proper numpy shape

        """

        _, ports = start_server_single_vehicle_model

        # Connect to grpc service
        stub = create_grpc_channel('localhost:{}'.format(ports["grpc_port"]),
                                   PREDICTION_SERVICE)

        imgs_v1_384 = np.ones((1, 3, 384, 672))
        in_name = 'data'
        out_name = 'detection_out'
        output = infer(imgs_v1_384, input_tensor=in_name, grpc_stub=stub,
                       model_spec_name='vehicle-detection',
                       model_spec_version=None,
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1, 200, 7), ERROR_SHAPE

    def test_run_inference_img(self, vehicle_adas_model_downloader,
                               vehicle_adas_data_downloader,
                               start_server_single_vehicle_model,
                               create_grpc_channel):
        """
        <b>Description</b>
        Submit request to gRPC interface serving a single vehicle model

        <b>input data</b>
        - directory with the model in IR format
        - docker image with ie-serving-py service

        <b>fixtures used</b>
        - model downloader
        - service launching

        <b>Expected results</b>
        - response contains proper numpy shape

        """

        _, ports = start_server_single_vehicle_model
        imgs_path = os.path.join(
            vehicle_adas_data_downloader, "data", "annotation_val_images")

        input_img = self.load_image(os.path.join(
            imgs_path, "image_000015.jpg"), 672, 384)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:{}'.format(ports["grpc_port"]),
                                   PREDICTION_SERVICE)

        in_name = 'data'
        out_name = 'detection_out'
        output = infer(input_img, input_tensor=in_name, grpc_stub=stub,
                       model_spec_name='vehicle-detection',
                       model_spec_version=None,
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1, 200, 7), ERROR_SHAPE

        detections_sum = 0
        result = output[out_name]
        print("result:" + str(result))

        LABEL = 1
        for i in range(0, 200):
            detection = result[0, 0, i]
            print("detection: " + str(detection))
            if not detection[LABEL] == 0.0:
                detections_sum += 1

        print("detections_sum= " + str(detections_sum))
        assert detections_sum == 19

    def test_run_inference_postprocess(self, vehicle_adas_model_downloader,
                                       vehicle_adas_data_downloader,
                                       start_server_single_vehicle_model,
                                       create_grpc_channel):
        """
        <b>Description</b>
        Submit request to gRPC interface serving a single vehicle model

        <b>input data</b>
        - directory with the model in IR format
        - docker image with ie-serving-py service

        <b>fixtures used</b>
        - model downloader
        - service launching

        <b>Expected results</b>
        - response contains proper numpy shape

        """
        _, ports = start_server_single_vehicle_model
        imgs_path = os.path.join(
            vehicle_adas_data_downloader, "data", "annotation_val_images")

        imgs = np.zeros((0, 3, 384, 672), np.dtype('<f'))
        input_img = self.load_image(os.path.join(
            imgs_path, "image_000015.jpg"), 672, 384)
        imgs = np.append(imgs, input_img, axis=0)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:{}'.format(ports["grpc_port"]),
                                   PREDICTION_SERVICE)

        in_name = 'data'
        out_name = 'detection_out'
        output = infer(imgs, input_tensor=in_name, grpc_stub=stub,
                       model_spec_name='vehicle-detection',
                       model_spec_version=None,
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1, 200, 7), ERROR_SHAPE

        sys.path.append(os.path.abspath(os.path.join(os.path.realpath(__file__),
                                                     '../../../extras/ams_wrapper/')))

        from src.api.models.model_builder import ModelBuilder

        config_path = os.path.abspath(os.path.join(os.path.realpath(__file__),
                                                   '../../../extras/ams_models/vehicle_detection_adas_model.json'))
        model_adas = ModelBuilder.build_model(config_path, 4000)

        json_response = model_adas.postprocess_inference_output(output)
        print("json_response=  " + str(json_response))

        boxes_count = str(json_response).count("box")

        print("detected boxes:" + str(boxes_count))
        assert boxes_count == 19

        try:
            format_check = json.loads(json_response)
        except Exception as e:
            print("json loads exception:" + str(e))
            assert False

        print("format_check:" + str(format_check))
        assert format_check["subtype"] == "vehicleDetection"
