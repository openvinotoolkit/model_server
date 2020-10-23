#
# Copyright (c) 2020 Intel Corporation
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


import os
import numpy as np

import config

MODEL_REPOSITORY_SERVER = "https://download.01.org"
OPENCV_OPENVINO_TOOLKIT = "opencv/2020/openvinotoolkit"
OPENCV_PUBLIC = "opencv/public_models"
OPENVINO_VERSION = "2020.2"
OPEN_MODEL_ZOO_BIN = "open_model_zoo/models_bin"
BUILD_DIR = "1"
BUILD_012020 = "012020"
PRECISION = "FP32"
FACE_DETECTION_MODEL = "face-detection-retail-0004"
AGE_GENDER_RECOGNITION_MODEL = "age-gender-recognition-retail-0013"
PERSON_VEHICLE_BIKE_DETECTION_MODEL = "person-vehicle-bike-detection-crossroad-0078"
RESNET_50 = "resnet-50-tf"
RESNET_V1_50 = "resnet_v1-50"
OPEN_MODEL_ZOO_MODELS_LOCATION = "{repo}/{opencv}/{version}/{bin}/{build}".format(repo=MODEL_REPOSITORY_SERVER,
                                                                                  opencv=OPENCV_OPENVINO_TOOLKIT,
                                                                                  version=OPENVINO_VERSION,
                                                                                  bin=OPEN_MODEL_ZOO_BIN,
                                                                                  build=BUILD_DIR)
URL_OPEN_MODEL_ZOO_FORMAT = "{model_location}/{model}/{precision}/{model}"
URL_PUBLIC_MODEL_FORMAT = "{repo}/{opencv}/{build}/{model}/{model_version}"


class AgeGender:
    name = "age_gender"
    dtype = np.float32
    input_name = "new_key"
    input_shape = (1, 3, 62, 62)
    output_name = ['age', 'gender']
    output_shape = {'age': (1, 1, 1, 1),
                    'gender': (1, 2, 1, 1)}
    rest_request_format = 'column_name'
    url = URL_OPEN_MODEL_ZOO_FORMAT.format(model_location=OPEN_MODEL_ZOO_MODELS_LOCATION,
                                           model=AGE_GENDER_RECOGNITION_MODEL, precision=PRECISION)  # noqa
    version = 1
    download_extensions = [".xml", ".bin"]
    model_path = os.path.join(config.models_path, name)


class PVBDetection:
    name = "pvb_detection"
    dtype = np.float32
    input_name = "data"
    input_shape = (1, 3, 300, 300)
    output_name = "detection_out"
    output_shape = (1, 1, 200, 7)
    rest_request_format = 'column_name'
    url = URL_OPEN_MODEL_ZOO_FORMAT.format(model_location=OPEN_MODEL_ZOO_MODELS_LOCATION,
                                           model=PERSON_VEHICLE_BIKE_DETECTION_MODEL, precision=PRECISION)  # noqa
    version = 1
    download_extensions = [".xml", ".bin"]


class FaceDetection:
    name = "face_detection"
    dtype = np.float32
    input_name = "data"
    input_shape = (1, 3, 300, 300)
    output_name = "detection_out"
    output_shape = (1, 1, 200, 7)
    rest_request_format = 'column_name'
    url = URL_OPEN_MODEL_ZOO_FORMAT.format(model_location=OPEN_MODEL_ZOO_MODELS_LOCATION,
                                           model=FACE_DETECTION_MODEL, precision=PRECISION)  # noqa
    version = 1
    download_extensions = [".xml", ".bin"]
    model_path = os.path.join(config.models_path, name)


class PVBFaceDetectionV1(FaceDetection):
    name = "pvb_face_multi_version"


class PVBFaceDetectionV2(PVBDetection):
    name = "pvb_face_multi_version"
    input_shape = (1, 3, 1024, 1024)
    version = 2


PVBFaceDetection = [PVBFaceDetectionV1, PVBFaceDetectionV2]


class Resnet:
    name = "resnet"
    dtype = np.float32
    input_name = "map/TensorArrayStack/TensorArrayGatherV3"
    input_shape = (1, 3, 224, 224)
    output_name = "softmax_tensor"
    output_shape = (1, 1001)
    rest_request_format = 'column_name'
    model_path = os.path.join(config.models_path, name)
    url = URL_PUBLIC_MODEL_FORMAT.format(repo=MODEL_REPOSITORY_SERVER, opencv=OPENCV_PUBLIC, build=BUILD_012020,
                                         model=RESNET_50, model_version=RESNET_V1_50)
    local_conversion_dir = "tensorflow_format"
    download_extensions = [".pb"]
    version = 1


class ResnetBS4:
    name = "resnet_bs4"
    dtype = np.float32
    input_name = "map/TensorArrayStack/TensorArrayGatherV3"
    input_shape = (4, 3, 224, 224)
    output_name = "softmax_tensor"
    output_shape = (4, 1001)
    rest_request_format = 'row_noname'
    version = 1


class ResnetBS8:
    name = "resnet_bs8"
    dtype = np.float32
    input_name = "map/TensorArrayStack/TensorArrayGatherV3"
    input_shape = (8, 3, 224, 224)
    output_name = "softmax_tensor"
    output_shape = (8, 1001)
    rest_request_format = 'row_noname'
    model_path = os.path.join(config.models_path, name)
    version = 1


class ResnetS3:
    name = "resnet_s3"
    dtype = np.float32
    input_name = "map/TensorArrayStack/TensorArrayGatherV3"
    input_shape = (1, 3, 224, 224)
    output_name = "softmax_tensor"
    output_shape = (1, 1001)
    rest_request_format = 'row_name'
    model_path = "s3://inference/resnet"


class ResnetGS:
    name = "resnet_gs"
    dtype = np.float32
    input_name = "0"
    input_shape = (1, 3, 224, 224)
    output_name = "1463"
    output_shape = (1, 1000)
    rest_request_format = 'row_name'
    model_path = "gs://ovms-public-eu/resnet50-binary"

class ResnetONNX:
    name = "resnet_onnx"
    dtype = np.float32
    input_name = "data"
    input_shape = (1, 3, 224, 224)
    output_name = "resnetv24_dense0_fwd"
    output_shape = (1, 1000)
    rest_request_format = 'row_name'
    url = "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7"
    download_extensions = [".onnx"]
    version = 1
    model_path = os.path.join(config.models_path, name)
