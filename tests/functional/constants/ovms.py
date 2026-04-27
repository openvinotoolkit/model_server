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

import os
import re

from enum import Enum
from pathlib import Path
from tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus

from tests.functional.constants.os_type import OsType

from ovms.config import ovms_c_repo_path, ovms_operator_repo_path
from ovms.constants.ov import OV
from tests.functional.constants.target_device import TargetDevice
from tests.functional.constants.ovms_type import OvmsType


class Ovms:

    # OPENVINO
    OPENVINO = "OpenVINO"
    OPENVINO_MODEL_SERVER = "OpenVINO Model Server"

    INFERENCE_PRECISION_HINT = "INFERENCE_PRECISION_HINT"

    # OVMS INTERNAL PARAMS
    CPU_THROUGHPUT_AUTO = "CPU_THROUGHPUT_AUTO"
    PLUGIN_CONFIG_CPU_STREAMS_THROUGHPUT_AUTO = {"CPU_THROUGHPUT_STREAMS": CPU_THROUGHPUT_AUTO}

    PLUGIN_CONFIG_AUTO = {"PERFORMANCE_HINT": "LATENCY"}
    PLUGIN_CONFIG_CPU = PLUGIN_CONFIG_AUTO
    PLUGIN_CONFIG_GPU = PLUGIN_CONFIG_AUTO
    PLUGIN_CONFIG_NPU = PLUGIN_CONFIG_AUTO
    PLUGIN_CONFIG_AUTO_CUMULATIVE_THROUGHPUT = {"PERFORMANCE_HINT": "CUMULATIVE_THROUGHPUT"}
    PLUGIN_CONFIG_HETERO = {"MULTI_DEVICE_PRIORITIES": "GPU,CPU"}

    PLUGIN_CONFIG = {
        TargetDevice.CPU: PLUGIN_CONFIG_CPU,
        TargetDevice.GPU: PLUGIN_CONFIG_GPU,
        TargetDevice.NPU: PLUGIN_CONFIG_NPU,
        TargetDevice.AUTO: PLUGIN_CONFIG_AUTO,
        TargetDevice.HETERO: PLUGIN_CONFIG_HETERO,
    }

    PLUGIN_CONFIG_CPU_PARAMS_LIST = [
        PLUGIN_CONFIG_CPU,
        {"NUM_STREAMS": "AUTO"},
        {"NUM_STREAMS": "1"},
        {"NUM_STREAMS": "32"},
        {"INFERENCE_NUM_THREADS": "1"},
        {"INFERENCE_NUM_THREADS": "24"},
        {"ENABLE_CPU_PINNING": "false"},
    ]

    PLUGIN_CONFIG_GPU_PARAMS_LIST = [
        PLUGIN_CONFIG_GPU,
        {"NUM_STREAMS": "AUTO"},
        {"NUM_STREAMS": "1"},
        {"NUM_STREAMS": "32"},
        {"PERFORMANCE_HINT": "THROUGHPUT"},
    ]

    PLUGIN_CONFIG_NPU_PARAMS_LIST = [
        PLUGIN_CONFIG_NPU,
        {"PERFORMANCE_HINT": "THROUGHPUT"},
    ]

    PLUGIN_CONFIG_AUTO_PARAMS_LIST = [
        PLUGIN_CONFIG_AUTO,
        {"PERFORMANCE_HINT": "THROUGHPUT"},
        PLUGIN_CONFIG_AUTO_CUMULATIVE_THROUGHPUT
    ]

    PLUGIN_CONFIG_HETERO_PARAMS_LIST = [
        # https://docs.openvino.ai/latest/openvino_docs_OV_UG_Hetero_execution.html
        PLUGIN_CONFIG_HETERO,
    ]

    PLUGIN_CONFIG_PARAMS = {
        TargetDevice.CPU: PLUGIN_CONFIG_CPU_PARAMS_LIST,
        TargetDevice.GPU: PLUGIN_CONFIG_GPU_PARAMS_LIST,
        TargetDevice.NPU: PLUGIN_CONFIG_NPU_PARAMS_LIST,
        TargetDevice.AUTO: PLUGIN_CONFIG_AUTO_PARAMS_LIST,
        TargetDevice.HETERO: PLUGIN_CONFIG_HETERO_PARAMS_LIST,
    }

    PLUGIN_CONFIG_WINDOWS = {"ENABLE_MMAP": "NO"}

    PERFORMANCE_HINT = "PERFORMANCE_HINT"
    PERFORMANCE_HINT_VALUES = ["LATENCY", "THROUGHPUT"]

    OVMS_REQUEST_TIMEOUT: int = 240

    V2_OPERATION_HEALTHY = "healthy"
    V2_OPERATION_METADATA = "metadata"
    V2_OPERATIONS = [V2_OPERATION_HEALTHY, V2_OPERATION_METADATA]

    OVMS_CONTAINER_NAME_DEFAULT = "ovms_cpp"
    ALL_MODEL_VERSION_POLICY = '{"all":{}}'
    TARGET_DEVICE_CPU = "CPU"
    TARGET_DEVICE_GPU = "GPU"
    TARGET_DEVICE_NPU = "NPU"
    BATCHSIZE = "1"
    BATCHSIZE_2 = "2"
    BATCHSIZE_3 = "3"
    BATCHSIZE_HUGE = "150"
    BATCHSIZE_TOO_LARGE = "1800"
    NIREQ = 2
    WINDOWS_GRPC_WORKERS = 1
    GRPC_WORKERS = 4
    REST_WORKERS = 4
    LOG_LEVEL_TRACE = "TRACE"
    LOG_LEVEL_DEBUG = "DEBUG"
    LOG_LEVEL_INFO = "INFO"
    LOG_LEVEL_ERROR = "ERROR"
    LOG_LEVEL_WARNING = "WARNING"
    DYNAMIC_AUTO_SIZE = "auto"

    SCALAR_BATCH_SIZE = "none"
    OUTPUT_FILLER = "\00" * 7

    # The time in seconds of OVMS verification if models have been changed on disk
    OVMS_MODELS_REFRESH_TIMEOUT = 2
    OVMS_DEFAULT_FILE_SYSTEM_POLL_WAIT_SECONDS = 1

    class ModelStatus(Enum):
        UNDEFINED = None
        UNKNOWN = ModelVersionStatus.UNKNOWN
        START = ModelVersionStatus.START
        LOADING = ModelVersionStatus.LOADING
        AVAILABLE = ModelVersionStatus.AVAILABLE
        UNLOADING = ModelVersionStatus.UNLOADING
        END = ModelVersionStatus.END

    LAYOUT_NHWC = "NHWC:NCHW"
    LAYOUT_NCHW = "NCHW:NCHW"

    IMAGE_CHANNEL_FORMAT_RGB = "RGB"

    BINARY_IO_LAYOUT_ROW_NAME = "row_name"
    BINARY_IO_LAYOUT_COLUMN_NAME = "column_name"
    BINARY_IO_LAYOUT_ROW_NONAME = "row_noname"
    BINARY_IO_LAYOUT_COLUMN_NONAME = "column_noname"

    TFS_REST_LAYOUT_TYPES = [
        BINARY_IO_LAYOUT_ROW_NAME,
        BINARY_IO_LAYOUT_COLUMN_NAME,
        BINARY_IO_LAYOUT_ROW_NONAME,
        BINARY_IO_LAYOUT_COLUMN_NONAME,
    ]

    GRPC_PROTOCOL_NAME = "gRPC"
    REST_PROTOCOL_NAME = "REST"

    JPG_IMAGE_FORMAT = "JPEG"
    PNG_IMAGE_FORMAT = "PNG"

    # For details take a peek:
    # model_server/docs/ovms_docs_streaming_endpoints.html#manual-timestamping
    TIMESTAMP_PARAM_NAME = "OVMS_MP_TIMESTAMP"

    SIGTERM_SIGNAL = "SIGTERM"
    SIGKILL_SIGNAL = "SIGKILL"
    SIGINT_SIGNAL = "SIGINT"
    KILL_SIGNAL = "KILL"
    TERM_SIGNAL = "TERM"

    STOP_METHOD = "stop"
    KILL_METHOD = "kill"

    MAX_THREADS_VALGRIND = 96 * 4

    GET_WEATHER_TOOLS = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia"
                    }
                },
                "required": [
                    "location"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    }]
    GET_WEATHER_TOOL_CHOICE = {"type": "function", "function": {"name": "get_weather"}}

    GET_POLLUTIONS_TOOLS = [{
        "type": "function",
        "function": {
            "name": "get_pollutions",
            "description": "Get current level of air pollutions for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia"
                    }
                },
                "required": [
                    "location"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    }]

    WEATHER_AND_POLLUTIONS_TOOLS = [GET_WEATHER_TOOLS[0], GET_POLLUTIONS_TOOLS[0]]

    @staticmethod
    def get_ovms_binary_paths(ovms_type, base_os=None):
        if base_os == OsType.Windows:
            ovms_binary_path = "ovms\\ovms.exe"
            ovms_lib_binary_path = ""
        else:
            ovms_binary_path = "ovms/bin/ovms" if ovms_type == OvmsType.BINARY else "/ovms/bin/ovms"
            ovms_lib_binary_path = "ovms/lib" if ovms_type == OvmsType.BINARY else "/ovms/lib/"
        return ovms_binary_path, ovms_lib_binary_path


class CurrentTarget:
    target_device = None

    is_auto_target = lambda: CurrentTarget.target_device in [TargetDevice.AUTO]
    is_hetero_target = lambda: CurrentTarget.target_device in [TargetDevice.HETERO]
    is_gpu_target = lambda: CurrentTarget.target_device in [TargetDevice.GPU]
    is_npu_target = lambda: CurrentTarget.target_device in [TargetDevice.NPU]
    is_cpu_target = lambda: CurrentTarget.target_device in [TargetDevice.CPU]

    @classmethod
    def is_plugin_target(cls):
        return any([
            cls.is_auto_target(),
            cls.is_hetero_target(),
            cls.is_gpu_target(),
        ])

    @staticmethod
    def is_gpu_based_target(target_device):
        return target_device in [
            TargetDevice.GPU,
            TargetDevice.NPU,
            TargetDevice.AUTO,
            TargetDevice.AUTO_CPU_GPU,
            TargetDevice.HETERO,
        ]


class CurrentOvmsType:
    ovms_type = None

    is_docker_type = lambda: CurrentOvmsType.ovms_type in [OvmsType.DOCKER]
    is_binary_type = lambda: CurrentOvmsType.ovms_type in [OvmsType.BINARY]
    is_binary_docker_type = lambda: CurrentOvmsType.ovms_type in [OvmsType.BINARY_DOCKER]
    is_kubernetes_type = lambda: CurrentOvmsType.ovms_type in [OvmsType.KUBERNETES]
    is_docker_cmd_line_type = lambda: CurrentOvmsType.ovms_type in [OvmsType.DOCKER_CMD_LINE]
    is_none_type = lambda: CurrentOvmsType.ovms_type in [OvmsType.NONE]


is_ovms_c_repo_absent = not Path(ovms_c_repo_path).exists()
is_ovms_operator_repo_absent = not Path(ovms_operator_repo_path).exists()

TARGET_DEVICE_PARAM_NAME = "target_device"
OVMS_TYPE_PARAM_NAME = "ovms_type"
API_TYPE_PARAM_NAME = "api_type"
GRPC_API_TYPE_PARAM_NAME = "grpc_api_type"
REST_API_TYPE_PARAM_NAME = "rest_api_type"
MODEL_TYPE_PARAM_NAME = "model_type"
CLOUD_TYPE_PARAM_NAME = "cloud_type"
USES_MAPPING_PARAM_NAME = "use_mapping"
USES_CONFIG_PARAM_NAME = "use_config"
BASE_OS_PARAM_NAME = "base_os"
TEST_RUN_WORKER_ARGUMENT = "test_run_reporters"
TMP_REPOS_DIR_ARGUMENT = "tmp_repos_dir"
CURRENT_TARGET_DEVICE_DICT_ARGUMENT = "current_target_device_dict"


class Config:
    MODEL_CONFIG_LIST = "model_config_list"
    MEDIAPIPE_CONFIG_LIST = "mediapipe_config_list"
    PIPELINE_CONFIG_LIST = "pipeline_config_list"
    CUSTOM_LOADER_CONFIG_LIST = "custom_loader_config_list"
    CUSTOM_NODE_LIBRARY_CONFIG_LIST = "custom_node_library_config_list"
    MONITORING = "monitoring"
    CONFIG = "config"
    PLUGIN_CONFIG = "plugin_config"


class MediaPipeConstants:
    DEFAULT_INPUT_STREAM = "in"
    DEFAULT_OUTPUT_STREAM = "out"

    IMAGE_URL_ZEBRA_JPEG = ("https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/"
                            "common/static/images/zebra.jpeg")
    IMAGE_URL_VEHICLES_PNG = ("https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/"
                           "vehicle_analysis_pipeline/python/vehicles_analysis.png")
    IMAGE_URL_INVALID_ZEBRA_PATH = ("https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/"
                                    "demos/common/static/images/zebra")
    IMAGE_URL_IMAGES_DIR = "https://github.com/openvinotoolkit/model_server/tree/main/demos/common/static/images"
    IMAGE_URL_REQUIREMENTS_FILE = ("https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/"
                                   "demos/common/export_models/requirements.txt")
    IMAGE_URL_RESIZED_ZEBRA_JPEG = ("https://raw.githubusercontent.com/intel-innersource/"
                                    "frameworks.ai.openvino.model-server.tests/refs/heads/"
                                    "main/data/llm/images/resized_zebra.jpeg")
    IMAGE_URL_HTTP_ZEBRA_JPEG = ("http://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/"
                                 "common/static/images/zebra.jpeg")
    IMAGE_FILESYSTEM_ZEBRA_JPEG = os.path.join(ovms_c_repo_path, "demos", "common", "static", "images", "zebra.jpeg")
    IMAGE_FILESYSTEM_INVALID_JPEG_PATH = os.path.join(ovms_c_repo_path, "demos", "common", "static", "images", "x.jpg")
    IMAGE_FILESYSTEM_DIRECTORY = os.path.join(ovms_c_repo_path, "demos", "common", "static", "images")


class MediapipeIntermediatePacket(Enum):
    # In case of whole graph input/output stream packet types accepted tags are:
    TENSOR = "TENSOR"
    TFTENSOR = "TFTENSOR"
    TFLITE_TENSOR = "TFLITE_TENSOR"
    OVTENSOR = "OVTENSOR"
    IMAGE = "IMAGE"


def set_plugin_config_boolean_value(plugin_config_str, config_file=False):
    # remove quotation marks for bool plugin_config values
    if config_file:
        plugin_config_pattern = re.compile(r"(\"plugin_config\":\s\{[\s\"\w]+\:\s)(\"(false|true)\")([\s\"\w]+\})")
        match_plugin_config = plugin_config_pattern.search(plugin_config_str)
        if match_plugin_config:
            return re.sub(
                plugin_config_pattern,
                f"{match_plugin_config[1]}{match_plugin_config[3]}{match_plugin_config[4]}",
                plugin_config_str,
            )
        return plugin_config_str
    else:
        return plugin_config_str.replace('\\"false\\"', "false").replace('\\"true\\"', "true")


def get_model_base_path(model_base_path, context, ovms_run):
    new_model_base_path = model_base_path if OvmsType.DOCKER in context.ovms_type \
        else os.path.join(ovms_run.ovms.container_folder, *model_base_path.split(os.path.sep)[1:])
    return new_model_base_path


class HfImportParams:
    PULL = "pull"
    LIST_MODELS = "list_models"
    ADD_TO_CONFIG = "add_to_config"
    REMOVE_FROM_CONFIG = "remove_from_config"
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    EMBEDDINGS = "embeddings"
    RERANK = "rerank"
