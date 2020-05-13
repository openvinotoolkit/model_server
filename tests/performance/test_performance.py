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

# Can be run locally with: py.test tests/test_performance.py -v -s

import os
import subprocess
import datetime
import urllib.request

from tabulate import tabulate

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASET = os.path.join(ROOT_PATH, "tests", "functional", "fixtures", "test_images", "performance")
OVMS_CLIENT = os.path.join(ROOT_PATH, "example_client", "face_detection_performance.py")
OVMS_DATASET = os.path.join(DATASET, "test_dir")

ITERATIONS = 1000
SERVICES = []


class TestPerformance:

    @staticmethod
    def inference(image, iterations: int):
        responses = []
        with open(os.path.join(DATASET, image), mode='rb') as image_file:
            image_bytes = image_file.read()

        for num in range(iterations):
            start_time = datetime.datetime.now()
            url = 'http://localhost:5000/vehicleDetection'

            headers = {'Content-Type': 'image/png'}
            req = urllib.request.Request(url, image_bytes, headers=headers)
            response = urllib.request.urlopen(req)
            assert response.getcode() == 200, "Not expected response code: {}".format(response.getcode)
            stop_time = datetime.datetime.now()
            duration = (stop_time - start_time).total_seconds() * 1000
            responses.append({"response": response,
                              "duration": duration})
        return responses

    @staticmethod
    def inference_ovms(image_width, image_height):
        cmd = ["python", OVMS_CLIENT, "--grpc_port", "9002", "--model_name", "vehicle_detection",
               "--input_images_dir", OVMS_DATASET, "--width", str(image_width), "--height", str(image_height)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output = proc.communicate()[0].decode('utf-8')
        metrics = [line for line in output.splitlines() if "average time" in line][0]
        average_proccessing_time = metrics.split(";")[0].split(":")[1]
        average_speed = metrics.split(";")[1].split(":")[1]
        return average_proccessing_time, average_speed

    def test_performance_simple_for_given_model_ams(self, ams):
        """
        <b>Description:</b>
        Checks AMS performance - latency

        <b>Assumption:</b>
        - AMS image present locally - ams

        <b>Input data:</b>
        - AMS
        - model
        - configuration

        <b>Expected results:</b>
        Test passes when AMS has results close to OVMS and OpenVino benchmark app.

        <b>Steps:</b>
        1. Run AMS and get response time.
        """

        responses = self.inference(image="single_car_small_reshaped.png", iterations=ITERATIONS)
        total_processing_time = 0
        total_speed = 0
        for rsp in responses:
            total_processing_time = total_processing_time + round(rsp["duration"], 2)
            total_speed = total_speed + round(1000/rsp["duration"], 2)
        average_proccessing_time = "{} ms".format(round(total_processing_time / len(responses), 2))
        average_speed = "{} fps".format(round(total_speed / len(responses), 2))
        ams = ["AMS", average_proccessing_time, average_speed]

        SERVICES.append(ams)

    def test_performance_simple_for_given_model_ovms(self, ovms, prepare_dataset_for_ovms):
        """
        <b>Description:</b>
        Checks OVMS performance - latency

        <b>Assumption:</b>
        - OVMS image present locally - ie-serving-py:latest

        <b>Input data:</b>
        - OVMS
        - model
        - configuration

        <b>Expected results:</b>
        Test passes when AMS has results close to OVMS and OpenVino benchmark app.

        <b>Steps:</b>
        1. Run OVMS and get response time.
        """

        average_proccessing_time, average_speed = self.inference_ovms(image_width=672, image_height=384)
        ovms = ["OVMS", average_proccessing_time, average_speed]
        SERVICES.append(ovms)
        print(tabulate(SERVICES, headers=['Average processing time', 'Average speed']))
