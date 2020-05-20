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

import json

import cv2
import numpy as np
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import requests

from extras.ams_wrapper.src.preprocessing.preprocess_image import preprocess_binary_image

test_image = "tests/functional/fixtures/test_images/1car_fit.jpeg"

img_out = cv2.imread(test_image)


def load_cv_image(file_path):
    # OpenVINO standard format
    img = cv2.imread(file_path)  # BGR color format, shape HWC
    img = img.transpose(2, 0, 1)
    return img


def load_tf_image(file_path):
    #
    with open(file_path, mode='rb') as image_file:
        image_bytes = image_file.read()
        image = preprocess_binary_image(
            image_bytes, reverse_input_channels=True)
    return image


def send_image_ovms(file_path):
    channel = grpc.insecure_channel("{}:{}".format("localhost", "9000"))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # when the same preprocessing is used, results would be identical
    img = load_cv_image(file_path).reshape(1, 3, 384, 672)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "vehicle_detection_adas"
    request.inputs["data"].CopyFrom(make_tensor_proto(img, shape=(img.shape)))
    result = stub.Predict(request, 10.0)
    output = make_ndarray(result.outputs["detection_out"])
    return output


def interpret_array(output, img_out):
    results = np.zeros((200, 5), np.float32)
    for i in range(0, 199):  # there is returned 200 detections for each image in the batch
        detection = output[:, :, i, :]
        # each detection has shape 1,1,7 where last dimension represent:
        # image_id - ID of the image in the batch
        # label - predicted class ID
        # conf - confidence for the predicted class
        # (x_min, y_min) - coordinates of the top left bounding box corner
        # (x_max, y_max) - coordinates of the bottom right bounding box corner.

        x_min = detection[0, 0, 3]
        y_min = detection[0, 0, 4]
        x_max = detection[0, 0, 5]
        y_max = detection[0, 0, 6]
        results[[i], :] = [detection[0, 0, 2],
                           x_min, y_min, x_max-x_min, y_max-y_min]
        if detection[0, 0, 2] > 0.5:  # ignore detections confidence <0.5
            print("detection", i)
            x_min_s = int(detection[0, 0, 3]*672)
            y_min_s = int(detection[0, 0, 4]*384)
            x_max_s = int(detection[0, 0, 5]*672)
            y_max_s = int(detection[0, 0, 6]*384)
        # box coordinates are proportional to the image size
            print("confidence", detection[0, 0, 2])
            print("x_min", x_min, x_min_s)
            print("y_min", y_min, y_min_s)
            print("x_max", x_max, x_max_s)
            print("y_max", y_max, y_max_s)
            img_out = cv2.rectangle(
                img_out, (x_min_s, y_min_s), (x_max_s, y_max_s), (0, 0, 255), 1)
            # draw each detected box on the input image
    cv2.imwrite("results_ovms.jpeg", img_out)
    return results


def interpret_ams(json_txt, img_out):
    y = json.loads(json_txt)
    results = np.zeros((200, 5), np.float32)
    detections = y["entities"]
    i = 0
    for detection in detections:
        confidence = detection["tag"]["confidence"]
        print("confidence", confidence)
        box = detection["box"]
        print("l:", box["l"], " t:", box["t"],
              " w:", box["w"], " h:", box["h"])
        if detection["tag"]["confidence"] > 0.5:
            print(box["l"]+box["w"])
            img_out = cv2.rectangle(img_out,
                                    (int(box["l"]*672), int(box["t"]*384)),
                                    (int((box["l"]+box["w"])*672),
                                     int((box["t"]+box["h"])*384)),
                                    (255, 0, 0), 1)
        results[[i], :] = [confidence, box["l"], box["t"], box["w"], box["h"]]
        i = i+1
    cv2.imwrite("results_combined.jpeg", img_out)

    return results


def send_image_ams(file_path):
    with open(file_path, mode='rb') as image_file:
        image_bytes = image_file.read()
        response = requests.post("http://localhost:5000/vehicleDetection",
                                 headers={'Content-Type': 'image/png',
                                          'Content-Length': str(len(image_bytes))},
                                 data=image_bytes)
    return response.content


i1 = load_cv_image(test_image)
print("loading image via opencv", i1.shape,
      " sum:", np.sum(i1), " dtype:", i1.dtype)

i2 = load_tf_image(test_image)
print("loading image via tensorflow", i2.shape,
      " sum:", np.sum(i2), " dtype:", i2.dtype)

diff = np.absolute(i1.astype(np.float32) -
                   i2.astype(np.float32)).transpose(1, 2, 0)
print("diff", np.sum(np.absolute(i1.astype(np.float32) - i2.astype(np.float32))))

print("Average normalized difference in volume:",
      np.sum(np.absolute(i1.astype(np.float32) -
                         i2.astype(np.float32)))/np.sum(i1),
      "should be < 0.02")

cv2.imwrite("cv_import_test.jpeg", i1.transpose(1, 2, 0))
cv2.imwrite("tf_import_test.jpeg", i2.transpose(1, 2, 0))

output = send_image_ovms(test_image)
t1 = interpret_array(output, img_out)
# array with all detection results from OVMS endpoint


json_output = send_image_ams(test_image)
print("AMS json response", json_output)

t2 = interpret_ams(json_output, img_out)
# array with all detection results from AMS endpoint
np.printoptions(precision=4, suppress=True)

print("OVMS results", t1, t1.dtype)
print("AMS results", t2, t2.dtype)
