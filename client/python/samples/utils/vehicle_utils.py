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
    response_dict = response.to_dict()
    output = response_dict[output_name]
    image = cv2.imread(image_path)
    width = image.shape[1]
    height = image.shape[0]
    for i in range(output.shape[2]):
        image = draw_vehicle(output[0, 0, i, ...], image, width, height)
    cv2.imwrite(os.path.join(output_save_path, 'vehicle-detection' + '.jpg'), image)


def draw_vehicle(vehicle_output, image, width, height):
    if vehicle_output[2] > 0.5:   # if confidence level is greater than 50%
        x_min = int(vehicle_output[3] * width)
        y_min = int(vehicle_output[4] * height)
        x_max = int(vehicle_output[5] * width)
        y_max = int(vehicle_output[6] * height)
        return cv2.rectangle(cv2.UMat(image), (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
    return image
