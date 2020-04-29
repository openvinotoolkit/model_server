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
import numpy as np
import pytest
from constants import MODEL_SERVICE, PREDICTION_SERVICE, ERROR_SHAPE
from utils.grpc import infer, get_model_metadata, model_metadata_response, \
    get_model_status
from utils.rest import infer_rest, get_model_metadata_response_rest, \
    get_model_status_response_rest

sys.path.append(".")
sys.path.append("extras/ams_wrapper/src/api")
from ie_serving.models.models_utils import ModelVersionState, ErrorCode, \
    _ERROR_MESSAGE  # noqa


class TestVehicleDetection():

    def load_image(self, file_path, width, height):
        img = cv2.imread(file_path)  # BGR color format, shape HWC
        img = cv2.resize(img, (width, height))
        img = img.transpose(2,0,1).reshape(1,3,height,width)
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
        imgs_path =  os.path.join(vehicle_adas_data_downloader, "data", "annotation_val_images")

        img_files = os.listdir(imgs_path)
        imgs = np.zeros((0,3,384,672), np.dtype('<f'))
        input_img = self.load_image(os.path.join(imgs_path,"image_000015.jpg"), 672, 384)
        imgs = np.append(imgs, input_img, axis=0)
        
        batch_size = 1
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
        
        detections_sum = 0
        result = output[out_name]
        print("result:" + str(result))
        for x in range(0, imgs.shape[0] - batch_size + 1, batch_size):
            img = imgs[x:(x + batch_size)]       
            print("img: " + str(img))
            for y in range(0,img.shape[0]):
                for i in range(0, 200*batch_size-1):
                    detection = result[:,:,i,:]
                    print("detection: " + str(detection))
                    if detection[0,0,2] > 0.5 and int(detection[0,0,0]) == y:
                        detections_sum+=1

        print("detections_sum= " + str(detections_sum))
        assert detections_sum == 2


    def test_run_inference_ams(self, vehicle_adas_model_downloader,
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
        imgs_path =  os.path.join(vehicle_adas_data_downloader, "data", "annotation_val_images")

        img_files = os.listdir(imgs_path)
        imgs = np.zeros((0,3,384,672), np.dtype('<f'))
        input_img = self.load_image(os.path.join(imgs_path,"image_000015.jpg"), 672, 384)
        imgs = np.append(imgs, input_img, axis=0)
        
        batch_size = 1
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
        
        os.chdir("extras/ams_wrapper/src/")

        from api.models.vehicle_detection_adas_model import VehicleDetectionAdas

        model_adas = VehicleDetectionAdas("ovms_connector")
        model_adas.load_default_labels()

        json_response = model_adas.postprocess_inference_output(output)
        print("json_response=  " + str(json_response))

        assert str(json_response).count("box") == 2        


"""
    def test_get_model_metadata(self, resnet_multiple_batch_sizes,
                                start_server_single_model,
                                create_grpc_channel):

        _, ports = start_server_single_model
        print("Downloaded model files:", resnet_multiple_batch_sizes)
        stub = create_grpc_channel('localhost:{}'.format(ports["grpc_port"]),
                                   PREDICTION_SERVICE)

        model_name = 'resnet'
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        expected_input_metadata = {in_name: {'dtype': 1,
                                             'shape': [1, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [1, 1001]}}
        request = get_model_metadata(model_name='resnet')
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    def test_get_model_status(self, resnet_multiple_batch_sizes,
                              start_server_single_model,
                              create_grpc_channel):

        print("Downloaded model files:", resnet_multiple_batch_sizes)

        _, ports = start_server_single_model
        stub = create_grpc_channel('localhost:{}'.format(ports["grpc_port"]),
                                   MODEL_SERVICE)
        request = get_model_status(model_name='resnet')
        response = stub.GetModelStatus(request, 10)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]
"""

"""
    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_run_inference_rest(self, resnet_multiple_batch_sizes,
                                start_server_single_model,
                                request_format):
        """"""
        <b>Description</b>
        Submit request to REST API interface serving a single resnet model

        <b>input data</b>
        - directory with the model in IR format
        - docker image with ie-serving-py service

        <b>fixtures used</b>
        - model downloader
        - service launching

        <b>Expected results</b>
        - response contains proper numpy shape

        """"""

        print("Downloaded model files:", resnet_multiple_batch_sizes)

        _, ports = start_server_single_model
        imgs_v1_224 = np.ones((1, 3, 224, 224))
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        rest_url = 'http://localhost:{}/v1/models/resnet:predict'.format(
                    ports["rest_port"])
        output = infer_rest(imgs_v1_224, input_tensor=in_name,
                            rest_url=rest_url,
                            output_tensors=[out_name],
                            request_format=request_format)
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1001), ERROR_SHAPE

    def test_get_model_metadata_rest(self, resnet_multiple_batch_sizes,
                                     start_server_single_model):
        print("Downloaded model files:", resnet_multiple_batch_sizes)

        _, ports = start_server_single_model
        model_name = 'resnet'
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        expected_input_metadata = {in_name: {'dtype': 1,
                                             'shape': [1, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [1, 1001]}}
        rest_url = 'http://localhost:{}/v1/models/resnet/metadata'.format(
                    ports["rest_port"])
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    def test_get_model_status_rest(self, resnet_multiple_batch_sizes,
                                   start_server_single_model):
        print("Downloaded model files:", resnet_multiple_batch_sizes)

        _, ports = start_server_single_model
        rest_url = 'http://localhost:{}/v1/models/resnet'.format(
                    ports["rest_port"])
        response = get_model_status_response_rest(rest_url)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]
"""
