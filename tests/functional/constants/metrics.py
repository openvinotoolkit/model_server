#
# Copyright (c) 2026 Intel Corporation
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

# Metrics feature API
# Documentation URL: https://docs.openvino.ai/latest/ovms_docs_metrics.html
# CLI parameters:
#   metrics_enabled: Flag indicating if metrics are enabled.
#   metrics_list: List of metrics families to be enabled.
#       If empty the default metrics are enabled
#           ovms_requests_success, ovms_requests_fail, ovms_request_time_us,
#           ovms_streams, ovms_inference_time_us, ovms_wait_for_infer_req_time_us.
#       Additional families:
#           ovms_infer_req_queue_size, ovms_infer_req_active
# Config:
#   "monitoring": {"metrics": {"enable": true, "metrics_list": [...]}}

import re
from enum import Enum, auto

from tests.functional.utils.logger import get_logger
from tests.functional.constants.ovms_messages import OvmsMessages

logger = get_logger(__name__)


class MetricsPolicy(str, Enum):
    Disabled = auto()
    NotDefined = auto()
    EnabledInConfig = auto()
    EnabledInCli = auto()
    EnabledMetricsList = auto()

    @staticmethod
    def get_policy(enable_metrics, use_config, metrics_list=None):
        if enable_metrics:
            metrics_policy = MetricsPolicy.EnabledInConfig if use_config else MetricsPolicy.EnabledInCli
        elif not enable_metrics and metrics_list is not None:
            # Negative scenario
            metrics_policy = MetricsPolicy.EnabledMetricsList
        else:
            metrics_policy = MetricsPolicy.Disabled

        return metrics_policy


class Metric:
    Method_success = "ovms_requests_success"
    Method_fail = "ovms_requests_fail"
    Stream = "ovms_streams"
    Infer_req_active = "ovms_infer_req_active"
    Infer_req_queue_size = "ovms_infer_req_queue_size"
    Current_request = "ovms_current_requests"
    Request_histogram = "ovms_request_time_us"
    Inference_histogram = "ovms_inference_time_us"
    Wait_for_inference_histogram = "ovms_wait_for_infer_req_time_us"

    Method_infer = "ModelInfer"

    Method_getmodelstatus = "GetModelStatus"
    Method_getmodelmetadata = "GetModelMetadata"
    Method_predict = "Predict"
    Method_modelready = "ModelReady"
    Method_modelmetadata = "ModelMetadata"

    TensorFlowServing = "TensorFlowServing"
    KServe = "KServe"

    Type_counter = "counter"
    Type_gauge = "gauge"
    Type_histogram = "histogram"

    Default = {
        Method_success: Type_counter,
        Method_fail: Type_counter,
        Stream: Type_gauge,
        Current_request: Type_gauge,
        Request_histogram: Type_histogram,
        Inference_histogram: Type_histogram,
        Wait_for_inference_histogram: Type_histogram,
    }

    Default_names = [x for x in Default.keys()]

    Additional = {"ovms_infer_req_queue_size": Type_gauge, "ovms_infer_req_active": Type_gauge}

    Additional_names = [x for x in Additional.keys()]

    Models_only = [
        Current_request,
        Stream,
        Inference_histogram,
        Wait_for_inference_histogram,
        "ovms_infer_req_queue_size",
        "ovms_infer_req_active",
    ]

    All = Default_names + Additional_names

    Interface_gRPC = "gRPC"
    Interface_REST = "REST"

    Interface = [Interface_REST, Interface_gRPC]

    Type = [Type_counter, Type_gauge, Type_histogram]

    DescType = {"requests_success", "counter", "request_fail", "counter"}

    Protocol = [KServe, TensorFlowServing]

    Methods = [
        Method_getmodelstatus,
        Method_getmodelmetadata,
        Method_predict,
        Method_infer,
        Method_modelready,
        Method_modelmetadata,
    ]

    MethodsProtocol = {
        Method_modelready: KServe,
        Method_modelmetadata: KServe,
        Method_infer: KServe,
        Method_getmodelstatus: TensorFlowServing,
        Method_getmodelmetadata: TensorFlowServing,
        Method_predict: TensorFlowServing,
    }

    Histogram_bucket_len_list = [
        "10",
        "18",
        "32",
        "58",
        "104",
        "188",
        "340",
        "612",
        "1101",
        "1983",
        "3570",
        "6426",
        "11568",
        "20822",
        "37481",
        "67466",
        "121439",
        "218591",
        "393464",
        "708235",
        "1274823",
        "2294682",
        "4130428",
        "7434771",
        "13382588",
        "24088659",
        "43359586",
        "78047255",
        "140485060",
        "252873108",
        "455171596",
        "819308872",
        "1474755971",
        "+Inf",
    ]

    @staticmethod
    def api_type(interface):
        arg_interface = interface.lower()
        result = list(filter(lambda x: x.lower() == arg_interface, Metric.Interface))
        assert len(result) > 0, f"Unable to match interface: {interface} to available values: {Metric.Interface}"
        return result[0]

    @staticmethod
    def create_method_metrics(model, base_name):
        result = []
        for method in Metric.Methods:
            for interface in Metric.Interface:
                content = {
                    "api": Metric.MethodsProtocol[method],
                    "interface": interface,
                    "method": method,
                    "name": model.name,
                }

                if method not in [Metric.Method_getmodelstatus, Metric.Method_modelready]:
                    # Defects in documentation:
                    # https://jira.devtools.intel.com/browse/CVS-92936
                    # https://jira.devtools.intel.com/browse/CVS-92935
                    content["version"] = str(model.version)

                result.append(Metric(metric_name=base_name, content=content))
        return result

    @staticmethod
    def create_successful_method_metrics(model, ovms_run=None):
        return Metric.create_method_metrics(model, Metric.Method_success)

    @staticmethod
    def create_fail_method_metrics(model, ovms_run=None):
        return Metric.create_method_metrics(model, Metric.Method_fail)

    @staticmethod
    def create_stream_metrics(model, ovms_run=None):
        value = None
        if ovms_run is not None:
            ovms_log_monitor = ovms_run.ovms.create_log(True)
            value = ovms_log_monitor.get_log_value(msg_to_found=OvmsMessages.OV_NUMBER_STREAMS)
        return [
            Metric(metric_name=Metric.Stream, content={"name": model.name, "version": str(model.version)}, value=value)
        ]

    @staticmethod
    def create_infer_request_active_metrics(model, ovms_run=None):
        return [
            Metric(
                metric_name=Metric.Infer_req_active,
                content={"name": model.name, "version": str(model.version)},
                value=0,
            )
        ]

    @staticmethod
    def create_infer_request_queue_size_metrics(model, ovms_run=None):
        value = None
        if ovms_run is not None:
            ovms_log_monitor = ovms_run.ovms.create_log(True)
            value = ovms_log_monitor.get_log_value(msg_to_found=OvmsMessages.OV_NIREQ)
        return [
            Metric(
                metric_name=Metric.Infer_req_queue_size,
                content={"name": model.name, "version": str(model.version)},
                value=value,
            )
        ]

    @staticmethod
    def create_current_request_metrics(model, ovms_run=None):
        return [Metric(metric_name=Metric.Current_request, content={"name": model.name, "version": str(model.version)})]

    @staticmethod
    def create_histogram_metrics(model, base_name, ovms_run=None):
        result = []
        result.append(
            Metric(metric_name=f"{base_name}_count", content={"name": model.name, "version": str(model.version)})
        )
        result.append(
            Metric(metric_name=f"{base_name}_sum", content={"name": model.name, "version": str(model.version)})
        )
        for bucket_len in Metric.Histogram_bucket_len_list:
            result.append(
                Metric(
                    metric_name=f"{base_name}_bucket",
                    content={"name": model.name, "version": str(model.version), "le": bucket_len},
                )
            )
        return result

    @staticmethod
    def create_request_histogram_metrics(model, ovms_run=None):
        base_name = Metric.Request_histogram
        result = []
        for interface in Metric.Interface:
            result.append(
                Metric(
                    metric_name=f"{base_name}_count",
                    content={"interface": interface, "name": model.name, "version": str(model.version)},
                )
            )
            result.append(
                Metric(
                    metric_name=f"{base_name}_sum",
                    content={"interface": interface, "name": model.name, "version": str(model.version)},
                )
            )
            for bucket_len in Metric.Histogram_bucket_len_list:
                result.append(
                    Metric(
                        metric_name=f"{base_name}_bucket",
                        content={
                            "interface": interface,
                            "name": model.name,
                            "version": str(model.version),
                            "le": bucket_len,
                        },
                    )
                )
        return result

    @staticmethod
    def create_wait_for_infer_histogram(model, ovms_run=None):
        return Metric.create_histogram_metrics(model, Metric.Wait_for_inference_histogram)

    @staticmethod
    def create_infer_histogram_metrics(model, ovms_run=None):
        return Metric.create_histogram_metrics(model, Metric.Inference_histogram)

    def __init__(self, metric_name, content: dict, value=0):
        self.name = metric_name
        self.content = content
        self.keys = [x for x in content]
        self.value = value

    def get_type(self):
        result = None
        if self.name in Metric.Default:
            result = Metric.Default[self.name]
        elif self.name in Metric.Additional:
            result = Metric.Additional[self.name]
        else:
            histogram_metrics = [
                Metric.Request_histogram,
                Metric.Inference_histogram,
                Metric.Wait_for_inference_histogram,
            ]
            if any([x in self.name for x in histogram_metrics]):
                result = Metric.Type_histogram
        return result

    def to_str(self):
        return f"{self.name}[{self.content}] {self.value}"

    def __str__(self):
        return self.to_str()

    def compare(self, ref_metric):
        return int(self.value) == int(ref_metric.value)


class Metrics:
    std_regexp = re.compile(r"(\w+)\{([^\}]+)\}\s+(\d+)")
    properties_regexp = re.compile(r"(\w+)=\"([\w\+-]+)\"")

    @staticmethod
    def create_from_request(request_output):
        lines = request_output.splitlines()
        metrics = []
        for line in lines:
            match = Metrics.std_regexp.search(line)
            if match:
                name, properties, value = match.groups()
                content = {}
                for key_val in properties.split(","):
                    key, v = Metrics.properties_regexp.search(key_val).groups()
                    content[key] = v
                metrics.append(Metric(metric_name=name, content=content, value=value))
        result = Metrics()
        result.list = metrics
        return result

    _fill_method = {
        Metric.Method_success: Metric.create_successful_method_metrics,
        Metric.Method_fail: Metric.create_fail_method_metrics,
        Metric.Stream: Metric.create_stream_metrics,
        Metric.Infer_req_active: Metric.create_infer_request_active_metrics,
        Metric.Infer_req_queue_size: Metric.create_infer_request_queue_size_metrics,
        Metric.Current_request: Metric.create_current_request_metrics,
        Metric.Request_histogram: Metric.create_request_histogram_metrics,
        Metric.Inference_histogram: Metric.create_infer_histogram_metrics,
        Metric.Wait_for_inference_histogram: Metric.create_wait_for_infer_histogram,
    }

    @staticmethod
    def create_from_model_list(model_list, ovms_run=None, metrics=None):
        metric_list = []
        for model in model_list:
            for metric in metrics:
                if model.is_pipeline():
                    for pipeline_model in model.get_models():
                        metric_list += Metrics._fill_method[metric](pipeline_model)

                    if metric not in Metric.Models_only:
                        metric_list += Metrics._fill_method[metric](model)
                else:
                    metric_list += Metrics._fill_method[metric](model=model, ovms_run=ovms_run)

        """ 
        The following metrics are not multiplied for each model version (should occur once for single model name)
        ovms_requests_success[{'api': 'TensorFlowServing', 'interface': 'gRPC', 'method': 'GetModelStatus', 'name': 'resnet-50-tf'}] 0
        ovms_requests_success[{'api': 'TensorFlowServing', 'interface': 'REST', 'method': 'GetModelStatus', 'name': 'resnet-50-tf'}] 0
        ovms_requests_success[{'api': 'KServe', 'interface': 'gRPC', 'method': 'ModelReady', 'name': 'resnet-50-tf'}] 0
        ovms_requests_success[{'api': 'KServe', 'interface': 'REST', 'method': 'ModelReady', 'name': 'resnet-50-tf'}] 0
        ovms_requests_fail[{'api': 'TensorFlowServing', 'interface': 'gRPC', 'method': 'GetModelStatus', 'name': 'resnet-50-tf'}] 0
        ovms_requests_fail[{'api': 'TensorFlowServing', 'interface': 'REST', 'method': 'GetModelStatus', 'name': 'resnet-50-tf'}] 0
        ovms_requests_fail[{'api': 'KServe', 'interface': 'gRPC', 'method': 'ModelReady', 'name': 'resnet-50-tf'}] 0
        ovms_requests_fail[{'api': 'KServe', 'interface': 'REST', 'method': 'ModelReady', 'name': 'resnet-50-tf'}] 0
        """
        metrics_to_remove = []
        model_unique_metrics = []
        for metric in metric_list:
            if metric.content.get("method", None) in [Metric.Method_getmodelstatus, Metric.Method_modelready]:
                if metric.to_str() in model_unique_metrics:
                    metrics_to_remove.append(metric)
                else:
                    model_unique_metrics.append(metric.to_str())

        for metric_to_remove in metrics_to_remove:
            metric_list.remove(metric_to_remove)

        result = Metrics()
        result.list = metric_list
        return result

    def __init__(self):
        self.list = []

    def _search_metrics(self, key):
        if isinstance(key, str):
            key = [key]

        search_keys = [key]
        if isinstance(key, tuple):
            search_keys = list(key)

        metric_name = key[0]
        search_keys = key[1:]

        result_metric_list = [x for x in self.list if x.name == metric_name]
        for item in search_keys:
            result_metric_list = [x for x in result_metric_list if item in x.to_str()]
        return result_metric_list

    def __getitem__(self, key):
        result_metric_list = self._search_metrics(key)

        result = Metrics()
        result.list = result_metric_list
        return result

    def __setitem__(self, key, value):
        result_metric_list = self._search_metrics(key)
        assert len(result_metric_list) > 0
        for metric in result_metric_list:
            metric.value = value

    def to_str(self):
        result = ""
        for metric in self.list:
            result += f"{metric.to_str()}\n"

        return result

    def search(self, metric):
        metric_list = [x for x in self.list if x.name == metric.name and x.content == metric.content]

        if len(metric_list) == 0:
            metric_list = [x for x in self.list if x.name == metric.name]
            for key, value in metric.content.items():
                tmp_metrics = [x for x in metric_list if key in x.content and x.content[key] == value]
                assert (
                    len(tmp_metrics) > 0
                ), f"Unable to find metric: {metric.to_str()} (closes: {[x.to_str() for x in metric_list]})"
                metric_list = tmp_metrics
        return metric_list[0]

    def compare(self, ref_metrics, metrics_title=None, ref_metrics_title=None):
        logger.info(f"Metrics::compare - nr of items: {len(self.list)}")
        metrics_not_equal = []
        for metric in self.list:
            logger.debug(f"\t{metric.to_str()}")
            found_metric = ref_metrics.search(metric)
            if found_metric.get_type() not in [Metric.Type_histogram] and metric.name not in [Metric.Stream]:
                if not metric.compare(found_metric):
                    error_message = f"Metrics are not equal: {metric.to_str()} != {found_metric.to_str()}"
                    logger.info(f"Metrics are not equal: {metric.to_str()} != {found_metric.to_str()}")
                    metrics_not_equal.append(error_message)

        assert len(metrics_not_equal) == 0
        assert len(self.list) == len(
            ref_metrics.list
        ), (f"Length of {metrics_title} ({len(self.list)}) "
            f"and length of {ref_metrics_title} ({len(ref_metrics.list)}) are not equal!")

    def find_metric_specific_value_content(self, metric_name, api, interface, method, model_name, model_version, value):
        expected_content = {
            "api": api,
            "interface": interface,
            "method": method,
            "name": model_name,
            "version": str(model_version),
        }
        metric_found = False
        for metric in self.list:
            if metric.content == expected_content and metric.name == metric_name:
                if metric.value == str(value):
                    logger.debug(f"Found expected value={value} in metric {metric_name} for method {method}")
                    metric_found = True
                    break
        assert metric_found, f"No metric found"

    def verify_metric_values(self, value):
        for metric in self.list:
            assert metric.value == value


class DefaultMetrics(Metrics):

    def __init__(self):
        super().__init__()

    @staticmethod
    def create_from_model_list(model_list):
        return Metrics.create_from_model_list(model_list, metrics=Metric.Default_names)


class AdditionalMetrics(Metrics):
    Names = ["ovms_infer_req_queue_size", "ovms_infer_req_active"]

    def __init__(self):
        super().__init__()


# Output example
#
# HELP ovms_requests_success Number of successful requests to a model or a DAG.
# TYPE ovms_requests_success counter
#
# ovms_requests_success{
#    interface="rest",
#    method="modelinfer",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="kserve",
#    version="1"} 0
# ovms_requests_success{
#    interface="grpc",
#    method="modelready",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="kserve"} 0
# ovms_requests_success{
#    interface="grpc",
#    method="modelmetadata",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="kserve",
#    version="1"} 0
# ovms_requests_success{
#    interface="rest",
#    method="getmodelstatus",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="tensorflowserving"} 0
# ovms_requests_success{
#    interface="rest",
#    method="getmodelmetadata",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="tensorflowserving",
#    version="1"} 0
# ovms_requests_success{
#    interface="rest",
#    method="predict",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="tensorflowserving",
#    version="1"} 0
# ovms_requests_success{
#    interface="grpc",
#    method="modelinfer",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="kserve",
#    version="1"} 0
# ovms_requests_success{
#    interface="grpc",
#    method="getmodelstatus",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="tensorflowserving"} 0
# ovms_requests_success{
#    interface="grpc",
#    method="getmodelmetadata",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="tensorflowserving",
#    version="1"} 0
# ovms_requests_success{
#    interface="rest",
#    method="modelready",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="kserve"} 0
# ovms_requests_success{
#    interface="rest",
#    method="modelmetadata",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="kserve",
#    version="1"} 0
# ovms_requests_success{
#    interface="grpc",
#    method="predict",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="tensorflowserving",
#    version="1"} 0
# # HELP ovms_requests_fail Number of failed requests to a model or a DAG.
# # TYPE ovms_requests_fail counter
# ovms_requests_fail{
#    interface="rest",
#    method="modelready",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="kserve",
#    version="1"} 0
# ovms_requests_fail{
#    interface="grpc",
#    method="modelready",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="kserve",
#    version="1"} 0
# ovms_requests_fail{
#    interface="grpc",
#    method="modelinfer",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="kserve",
#    version="1"} 0
# ovms_requests_fail{
#    interface="rest",
#    method="modelinfer",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="kserve",
#    version="1"} 0
# ovms_requests_fail{
#    interface="rest",
#    method="getmodelstatus",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="tensorflowserving",
#    version="1"} 0
# ovms_requests_fail{
#    interface="rest",
#    method="getmodelmetadata",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="tensorflowserving",
#    version="1"} 0
# ovms_requests_fail{
#    interface="rest",
#    method="predict",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="tensorflowserving",
#    version="1"} 0
# ovms_requests_fail{
#    interface="grpc",
#    method="modelmetadata",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="kserve",
#    version="1"} 0
# ovms_requests_fail{
#    interface="grpc",
#    method="getmodelstatus",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="tensorflowserving",
#    version="1"} 0
# ovms_requests_fail{
#    interface="grpc",
#    method="getmodelmetadata",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="tensorflowserving",
#    version="1"} 0
# ovms_requests_fail{
#    interface="rest",
#    method="modelmetadata",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="kserve",
#    version="1"} 0
# ovms_requests_fail{
#    interface="grpc",
#    method="predict",
#    name="ssdlite_mobilenet_v2_ov",
#    protocol="tensorflowserving",
#    version="1"} 0
# # HELP ovms_streams Number of OpenVINO execution streams.
# # TYPE ovms_streams gauge
# ovms_streams{name="ssdlite_mobilenet_v2_ov",version="1"} 4
# # HELP ovms_infer_req_queue_size Inference request queue size (nireq).
# # TYPE ovms_infer_req_queue_size gauge
# ovms_infer_req_queue_size{name="ssdlite_mobilenet_v2_ov",version="1"} 4
# # HELP ovms_infer_req_active Number of currently consumed inference request from the processing queue.
# # TYPE ovms_infer_req_active gauge
# ovms_infer_req_active{name="ssdlite_mobilenet_v2_ov",version="1"} 0
# # HELP ovms_current_requests Number of inference requests currently in process.
# # TYPE ovms_current_requests gauge
# ovms_current_requests{name="ssdlite_mobilenet_v2_ov",version="1"} 0
# # HELP ovms_request_time_us Processing time of requests to a model or a DAG.
# # TYPE ovms_request_time_us histogram
# ovms_request_time_us_count{interface="rest",name="ssdlite_mobilenet_v2_ov",version="1"} 0
# ovms_request_time_us_sum{interface="rest",name="ssdlite_mobilenet_v2_ov",version="1"} 0
# ovms_request_time_us_bucket{interface="rest",name="ssdlite_mobilenet_v2_ov",version="1",le="10"} 0
# ovms_request_time_us_bucket{interface="rest",name="ssdlite_mobilenet_v2_ov",version="1",le="18"} 0
# ...
# ovms_request_time_us_bucket{interface="rest",name="ssdlite_mobilenet_v2_ov",version="1",le="1474755971"} 0
# ovms_request_time_us_bucket{interface="rest",name="ssdlite_mobilenet_v2_ov",version="1",le="+Inf"} 0
# ovms_request_time_us_count{interface="grpc",name="ssdlite_mobilenet_v2_ov",version="1"} 0
# ovms_request_time_us_sum{interface="grpc",name="ssdlite_mobilenet_v2_ov",version="1"} 0
# ovms_request_time_us_bucket{interface="grpc",name="ssdlite_mobilenet_v2_ov",version="1",le="10"} 0
# ...
# ovms_request_time_us_bucket{interface="grpc",name="ssdlite_mobilenet_v2_ov",version="1",le="+Inf"} 0
# # HELP ovms_inference_time_us Inference execution time in the OpenVINO backend.
# # TYPE ovms_inference_time_us histogram
# ovms_inference_time_us_count{name="ssdlite_mobilenet_v2_ov",version="1"} 0
# ovms_inference_time_us_sum{name="ssdlite_mobilenet_v2_ov",version="1"} 0
# ovms_inference_time_us_bucket{name="ssdlite_mobilenet_v2_ov",version="1",le="10"} 0
# ...
# ovms_inference_time_us_bucket{name="ssdlite_mobilenet_v2_ov",version="1",le="+Inf"} 0
# # HELP ovms_wait_for_infer_req_time_us Request waiting time in the scheduling queue.
# # TYPE ovms_wait_for_infer_req_time_us histogram
# ovms_wait_for_infer_req_time_us_count{name="ssdlite_mobilenet_v2_ov",version="1"} 0
# ovms_wait_for_infer_req_time_us_sum{name="ssdlite_mobilenet_v2_ov",version="1"} 0
# ovms_wait_for_infer_req_time_us_bucket{name="ssdlite_mobilenet_v2_ov",version="1",le="10"} 0
# ...
# ovms_wait_for_infer_req_time_us_bucket{name="ssdlite_mobilenet_v2_ov",version="1",le="+Inf"} 0
