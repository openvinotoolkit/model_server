#
# Copyright (c) 2021 Intel Corporation
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
import os


def vehicle_postprocess(response, image_path, output_name, output_save_path):
    img_name = image_path.split('/')[-1]
    if isinstance(response, dict):
        output = response[output_name]
    else:
        output = response
    image = cv2.imread(image_path)
    width = image.shape[1]
    height = image.shape[0]
    confidence_threshold = 0.5
    for i in range(output.shape[2]):
        image = draw_bounding_box(output[0, 0, i, ...], image, width, height, confidence_threshold)
    if not os.path.exists(output_save_path):
        print("Making directory for output: " + output_save_path)
        os.makedirs(output_save_path)
    output_file = os.path.join(output_save_path, img_name)
    cv2.imwrite(output_file, image)
    print("Detection results in file: ", output_file)


def draw_bounding_box(prediction_result, image, width, height, confidence_threshold):
    if prediction_result[2] > confidence_threshold:   # if confidence level is greater than confidence_threshold
        x_min = int(prediction_result[3] * width)
        y_min = int(prediction_result[4] * height)
        x_max = int(prediction_result[5] * width)
        y_max = int(prediction_result[6] * height)
        return cv2.rectangle(cv2.UMat(image), (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
    return image
