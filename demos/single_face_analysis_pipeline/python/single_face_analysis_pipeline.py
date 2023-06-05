#
# Copyright (c) 2022 Intel Corporation
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

import cv2
from tensorflow import make_tensor_proto, make_ndarray
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import argparse

parser = argparse.ArgumentParser(description='Client for single face analysis pipeline')
parser.add_argument('--grpc_address', required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--pipeline_name', required=False, default='single_face_analysis', help='Pipeline name to request. default: single_face_analysis')
parser.add_argument('--image_path', required=True, help='Path to the file with the input image')

args = vars(parser.parse_args())

def getJpeg(path, size):

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # retrieved array has BGR format and 0-255 normalization
    img = cv2.resize(img, (size, size))
    img = img.astype('float32')
    img = img.transpose(2,0,1).reshape(1,3,size,size)
    return img

my_image = getJpeg(args["image_path"],64)

channel = grpc.insecure_channel("{}:{}".format(args["grpc_address"], args["grpc_port"]))
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = args["pipeline_name"]
request.inputs["image"].CopyFrom(make_tensor_proto(my_image, shape=(my_image.shape)))
result = stub.Predict(request, 10.0)
age_results = make_ndarray(result.outputs["age"])
gender_results = make_ndarray(result.outputs["gender"])
emotion_results = make_ndarray(result.outputs["emotion"])

print("Age results:", age_results[0]*100)
print("Gender results: Female:", gender_results[0,0,0,0], "; Male:", gender_results[0,1,0,0] )
print("Emotion results: Natural:", emotion_results[0,0,0,0], "; Happy:", emotion_results[0,1,0,0], "; Sad:", emotion_results[0,2,0,0], "; Surprise:", emotion_results[0,3,0,0], "; Angry:", emotion_results[0,4,0,0] )