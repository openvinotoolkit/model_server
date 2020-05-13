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
import pytest

from tabulate import tabulate
from data.performance_constants import DATASET, OVMS_CLIENT, OVMS_DATASET, ITERATIONS, AMS_PORT, \
    AMS_ADDRESS, OVMS_PORT, OVMS_CLIENT_PATH, AMS_CLIENT_PATH, MODELS, PARAMS, OVMS_DATASET

from fixtures.performance import run_ams, cleanup_ams, run_ovms, cleanup_ovms, prepare_dataset_for_ovms, cleanup_dataset

LATENCY = []
THROUGHPUT = []


class TestPerformance:

    @staticmethod
    def inference(model_name, image, iterations: int):
        responses = []
        with open(os.path.join(DATASET, image), mode='rb') as image_file:
            image_bytes = image_file.read()

        for num in range(iterations):
            start_time = datetime.datetime.now()
            url = 'http://{}:{}/{}'.format(AMS_ADDRESS, AMS_PORT, model_name)

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
    def inference_ovms(model_name, image_width, image_height):
        cmd = ["python", OVMS_CLIENT, "--grpc_port", OVMS_PORT, "--model_name", model_name,
               "--input_images_dir", OVMS_DATASET, "--width", str(image_width), "--height",
               str(image_height), "--model_name", model_name]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output = proc.communicate()[0].decode('utf-8')
        metrics = [line for line in output.splitlines() if "average time" in line][0]
        average_proccessing_time = metrics.split(";")[0].split(":")[1]
        average_speed = metrics.split(";")[1].split(":")[1]
        return average_proccessing_time, average_speed

    @staticmethod
    def return_metrics(cmd):
        ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = ps.communicate()[0].decode()
        print(output)
        print(cmd)
        times = [line for line in output.splitlines() if "real" in line][0]
        time = times.split("\t")[1].split("m")
        final_time = float(time[0]) * 60 + float(time[1].replace("s", ""))
        throughput = 10 * 1 * (ITERATIONS / final_time)
        subprocess.call("rm -rf metrics", shell=True)
        return final_time, throughput

    @staticmethod
    def measure_throughput_ams(model_name, image):
        cmd = " ".join(["time printf '%s\\n' {1..10} | xargs -n 1 -P 10 python",
                        AMS_CLIENT_PATH, model_name, image])
        return TestPerformance.return_metrics(cmd)

    @staticmethod
    def measure_throughput_ovms(model_name, width, height):
        cmd = " ".join(["time printf '%s\\n' {1..10} | xargs -n 1 -P 10 sh",
                        OVMS_CLIENT_PATH, model_name, width, height, OVMS_PORT, OVMS_DATASET])
        return TestPerformance.return_metrics(cmd)

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("params", PARAMS)
    def test_performance_latency_one_client_ams(self, model, params):
        """
        <b>Description:</b>
        Checks AMS performance - latency

        <b>Input data:</b>
        - AMS and OVMS
        - model
        - configuration

        <b>Expected results:</b>
        Test passes when AMS has results close to OVMS and OpenVino benchmark app.

        <b>Steps:</b>
        1. Run AMS and get response time.
        2. Run OVMS and get response time.
        """
        # running ams
        container = run_ams(params)

        # measure latency for ams
        responses = self.inference(image=model["dataset"], iterations=ITERATIONS,
                                   model_name=model["model_name"])
        total_processing_time = 0
        total_speed = 0
        for rsp in responses:
            total_processing_time = total_processing_time + round(rsp["duration"], 2)
            total_speed = total_speed + round(1000/rsp["duration"], 2)
        average_proccessing_time = "{} ms".format(round(total_processing_time / len(responses), 2))
        average_speed = "{} fps".format(round(total_speed / len(responses), 2))
        plugin_config = "single stream" if not params["plugin_config"] else "multi stream"
        ams = [model["model_name"], params["cores"],
               "ams", params["nireq"], params["grpc_workers"], plugin_config,
               average_proccessing_time, average_speed]
        LATENCY.append(ams)

        # remove ovms
        cleanup_ams(container)

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("params", PARAMS)
    def test_performance_latency_one_client_ovms(self, model, params):
        """
        <b>Description:</b>
        Checks AMS performance - latency

        <b>Input data:</b>
        - AMS and OVMS
        - model
        - configuration

        <b>Expected results:</b>
        Test passes when AMS has results close to OVMS and OpenVino benchmark app.

        <b>Steps:</b>
        1. Run AMS and get response time.
        2. Run OVMS and get response time.
        """

        # prepare dataset
        prepare_dataset_for_ovms(params["dataset"])

        # running ams
        container = run_ovms(params)

        # measure latency for ovms
        plugin_config = "single stream" if not params["plugin_config"] else "multi stream"
        average_proccessing_time, average_speed = self.inference_ovms(model_name=model["model_name_ovms"],
                                                                      image_width=model["width"],
                                                                      image_height=model["height"])
        ovms = [model["model_name"], params["cores"], "ovms", params["nireq"],
                params["grpc_workers"], plugin_config, average_proccessing_time, average_speed]
        LATENCY.append(ovms)

        # remove ovms
        cleanup_ovms(container)

        # remove dataset
        cleanup_dataset()

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("params", PARAMS)
    def test_performance_measure_throughput_ten_clients_ams(self, model, params):
        """
        <b>Description:</b>
        Measures OVMS and AMS performance - throughput

        <b>Input data:</b>
        - OVMS and AMS
        - model
        - configuration

        <b>Expected results:</b>
        Test passes when AMS results are close to OVMS results - throughput

        <b>Steps:</b>
        1. Measure OVMS and AMS results
        """
        # run ams
        container = run_ams(params)

        # measure throughput for ams
        plugin_config = "single stream" if not params["plugin_config"] else "multi stream"

        final_time, throughput = self.measure_throughput_ams(model_name=model["model_name"], image=model["dataset"])
        ams = [model["model_name"], params["cores"], "ams", params["nireq"],
               params["grpc_workers"], plugin_config, final_time, throughput]
        THROUGHPUT.append(ams)

        # remove ams
        cleanup_ams(container)

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("params", PARAMS)
    def test_performance_measure_throughput_ten_clients_ovms(self, model, params):
        """
        <b>Description:</b>
        Measures OVMS and AMS performance - throughput

        <b>Input data:</b>
        - OVMS and AMS
        - model
        - configuration

        <b>Expected results:</b>
        Test passes when AMS results are close to OVMS results - throughput

        <b>Steps:</b>
        1. Measure OVMS and AMS results
        """
        # prepare dataset
        prepare_dataset_for_ovms(params["dataset"])

        # run ovms
        container = run_ovms(params)

        # measure throughput for ovms
        final_time, throughput = self.measure_throughput_ovms(model_name=model["model_name_ovms"], width=model["width"],
                                                              height=model["height"])
        plugin_config = "single stream" if not params["plugin_config"] else "multi stream"
        ovms = [model["model_name"], params["cores"], "ovms", params["nireq"],
                params["grpc_workers"], plugin_config, final_time, throughput]
        THROUGHPUT.append(ovms)

        # remove ovms
        cleanup_ovms(container)

        # remove dataset
        cleanup_dataset()

    def test_performance_compare_latency_one_client(self):
        """
        <b>Description:</b>
        Compares OVMS and AMS performance - latency

        <b>Input data:</b>
        - OVMS performance - latency results
        - AMS performance - latency results

        <b>Expected results:</b>
        Test passes when AMS results are close to OVMS results - latency

        <b>Steps:</b>
        1. Compare OVMS and AMS results
        """
        print("\n")
        print(tabulate(LATENCY, headers=["Model", "Cores", "Service", "Nireq", "Grcp_workers",
                                         "Singlestream/Multistream", "Average processing time", "Average speed"]))

    def test_performance_compare_throughput_ten_clients(self):
        """
        <b>Description:</b>
        Compares OVMS and AMS performance - throughput

        <b>Input data:</b>
        - OVMS performance - throughput results
        - AMS performance - throughput results

        <b>Expected results:</b>
        Test passes when AMS results are close to OVMS results - throughput

        <b>Steps:</b>
        1. Compare OVMS and AMS results
        """
        print("\n")
        print(tabulate(THROUGHPUT, headers=["Model", "Cores", "Service", "Nireq", "Grcp_workers",
                                            "Singlestream/Multistream", "Final time", "Throughput"]))
