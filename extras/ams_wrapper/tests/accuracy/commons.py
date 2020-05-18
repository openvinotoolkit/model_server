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

import cv2
import numpy as np
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import json
import requests
import os


def load_ov_standard_image(file_path, height, width):
    # OpenVINO standard format
    image = cv2.imread(file_path)  # BGR color format, shape HWC
    if image.shape[:-1] != (height, width):
        image = cv2.resize(image, (width, height))
    image = image.transpose(2, 0, 1)
    return image


def load_ams_image(file_path):
    with open(file_path, mode='rb') as image_file:
        image_bytes = image_file.read()
    return image_bytes


def send_image_ovms(address, port, image, model_name, input_name, output_name, channels, height, width):
    channel = grpc.insecure_channel("{}:{}".format(address, port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    shape = (1, channels, height, width)
    image.reshape(shape)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs[input_name].CopyFrom(
        make_tensor_proto(image, shape=shape))
    result = stub.Predict(request, 10.0)
    output = {}
    for out in output_name:
        output[out] = make_ndarray(result.outputs[out])
    return output


def extract_content_type(file_path):
    _, ext = os.path.splitext(file_path)
    if ext == ".bmp":
        return 'image/bmp'
    elif ext == ".png":
        return 'image/png'
    elif ext == ".jpg":
        return 'image/jpg'
    else:
        return None


def send_image_ams(image, address, port, endpoint, content_type):
    response = requests.post("http://{}:{}/{}".format(address, port, endpoint),
                             headers={'Content-Type': content_type,
                                      'Content-Length': str(len(image))},
                             data=image)
    return response.content


def detection_array(output, img_out, height, width):
    results = {}
    for out in output:
        results[out] = []
        # there is returned 200 detections for each image in the batch
        for i in range(0, len(output[out][0][0])):
            detection = output[out][0][0][i]

            x_min = detection[3]
            y_min = detection[4]
            x_max = detection[5]
            y_max = detection[6]
            if detection[2] > 0.5:  # ignore detections confidence <0.5
                x_min_s = int(x_min*width)
                y_min_s = int(y_min*height)
                x_max_s = int(x_max*width)
                y_max_s = int(y_max*height)
                img_out = cv2.rectangle(
                    img_out, (x_min_s, y_min_s), (x_max_s, y_max_s), (0, 0, 255), 1)
                # draw each detected box on the input image
                results[out].append([detection[2], x_min, y_min, x_max, y_max])
            cv2.imwrite("results_ovms.jpg", img_out)
    return results


def classification_array(output, output_names, classes):
    results = {}
    tag_name = ""
    for output_name in output_names:
        highest_prob = 0.0
        for class_name in classes[output_name]:
            class_id = classes[output_name][class_name]
            probability = output[output_name][0,int(float(class_id)),0,0].item()
            if probability > highest_prob:
                tag_name = class_name
                highest_prob = probability
        results[output_name] = {"value": tag_name, "confidence": highest_prob}
    return results



def detection_json(json_txt, img_out, height, width):
    ams_detection_json = json.loads(json_txt)
    results = []
    detections = ams_detection_json["entities"]
    i = 0
    for detection in detections:
        confidence = detection["tag"]["confidence"]
        box = detection["box"]
        if detection["tag"]["confidence"] > 0.5:
            img_out = cv2.rectangle(img_out, (int(box["l"]*width), int(box["t"]*height)), (int(
                (box["l"]+box["w"])*width), int((box["t"]+box["h"])*height)), (255, 0, 0), 1)
            results.append([confidence, box["l"], box["t"], box["w"], box["h"]])
        i = i+1
    cv2.imwrite("results_ams.jpg", img_out)

    return results


def classification_json(json_txt):
    ams_classification_json = json.loads(json_txt)
    classifications = ams_classification_json["classifications"]
    results = {}
    for classification in classifications:
        attributes = classification["attributes"]
        highest_probability = {"value": "", "confidence": 0.0}
        for attribute in attributes:
            if attribute["confidence"] > highest_probability["confidence"]:
                highest_probability["confidence"] = attribute["confidence"]
                highest_probability["value"] = attribute["value"]
        results[attributes[0]["name"]] = highest_probability
    return results
