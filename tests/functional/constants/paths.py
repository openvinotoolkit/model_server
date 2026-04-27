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
import performance

from tests.functional.constants.os_type import OsType
from ovms import config
from ovms.constants.ov import OV
from tests.functional.constants.target_device import TargetDevice


class Paths:

    MODELS_PATH_NAME = "models"
    CUSTOM_NODE_PATH_NAME = "custom_nodes"
    CUSTOM_LOADER_PATH_NAME = "custom_loader"
    CPU_EXTENSIONS = "cpu_extensions"
    CONFIG_FILE_NAME = "config.json"
    SUBCONFIG_FILE_NAME = "subconfig.json"
    GRAPH_NAME = "graph.pbtxt"
    IMAGES = "images"
    OVMS_PATH_INTERNAL = os.path.join("/") if OsType.Windows not in config.base_os else os.path.join("\\")
    MODELS_PATH_INTERNAL = os.path.join(OVMS_PATH_INTERNAL, MODELS_PATH_NAME)
    CONFIG_PATH_INTERNAL = os.path.join(MODELS_PATH_INTERNAL, CONFIG_FILE_NAME)
    CUSTOM_NODE_LIBRARIES_PATH_INTERNAL = os.path.join(OVMS_PATH_INTERNAL, CUSTOM_NODE_PATH_NAME)
    CUSTOM_LOADER_LIBRARIES_PATH_INTERNAL = os.path.join(OVMS_PATH_INTERNAL, CUSTOM_LOADER_PATH_NAME)
    ROOT_PATH_CPU_EXTENSIONS = os.path.join(OVMS_PATH_INTERNAL, CPU_EXTENSIONS)
    IMAGES_PATH_INTERNAL = os.path.join(OVMS_PATH_INTERNAL, IMAGES)
    ZEBRA_PATH_INTERNAL = os.path.join(IMAGES_PATH_INTERNAL, "zebra.jpeg")
    CACHE_INTERNAL = os.path.join(OVMS_PATH_INTERNAL, "opt", "cache")

    # CAPI OVMS-C
    OVMS_C_CAPI_MIMIMAL_APP_DIR = os.path.join(config.ovms_c_repo_path, "demos", "c_api_minimal_app")
    OVMS_C_CAPI_MIMIMAL_APP_MAKEFILE = os.path.join(OVMS_C_CAPI_MIMIMAL_APP_DIR, "Makefile")
    OVMS_C_CAPI_MIMIMAL_APP_FILES_DIR = os.path.join(OVMS_C_CAPI_MIMIMAL_APP_DIR, "capi_files")
    OVMS_C_CAPI_SRC_TEST_DUMMY = os.path.join(config.ovms_c_repo_path, "src", "test", "dummy")
    OVMS_C_CAPI_MIMIMAL_APP_BENCHMARK_FILE = os.path.join(
        OVMS_C_CAPI_MIMIMAL_APP_DIR, "..", "..", "src", "main_benchmark.cpp"
    )

    # CAPI OVMS-TEST
    OVMS_TEST_CAPI_WRAPPER_DIR = os.path.join(config.ovms_test_repo_path, "data", "ovms_capi_wrapper")
    OVMS_TEST_CAPI_PXD = os.path.join(OVMS_TEST_CAPI_WRAPPER_DIR, "ovms_capi.pxd")
    OVMS_TEST_CAPI_AUTOPXD_PY = os.path.join(OVMS_TEST_CAPI_WRAPPER_DIR, "ovms_autopxd.py")
    OVMS_TEST_CAPI_WRAPPER_PYX = os.path.join(OVMS_TEST_CAPI_WRAPPER_DIR, "ovms_capi_wrapper.pyx")
    OVMS_TEST_CAPI_WRAPPER_MAKEFILE = os.path.join(OVMS_TEST_CAPI_WRAPPER_DIR, "Makefile")
    OVMS_TEST_CAPI_WRAPPER_SETUP = os.path.join(OVMS_TEST_CAPI_WRAPPER_DIR, "setup.py")

    OVMS_TEST_CAPI_SCENARIOS_DIR = os.path.join(config.ovms_test_repo_path, "data", "capi_scenarios")
    OVMS_TEST_CAPI_NEGATIVE_SCENARIOS_DIR = os.path.join(OVMS_TEST_CAPI_SCENARIOS_DIR, "negative")
    OVMS_TEST_CAPI_POSITIVE_SCENARIOS_DIR = os.path.join(OVMS_TEST_CAPI_SCENARIOS_DIR, "positive")

    # OVMS-C PYTHON CUSTOM NODES
    OMVS_C_PYTHON_CUSTOM_NODES_TEST_SCRIPTS_DIR = os.path.join(
        config.ovms_c_repo_path, "src", "test", "mediapipe", "python", "scripts"
    )

    # performance
    PERFORMANCE_DIR = os.path.dirname(performance.__file__)
    ROOT_DIR = os.path.dirname(PERFORMANCE_DIR)
    PARENT_ROOT_DIR = os.path.dirname(ROOT_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, "data")

    CLIENT_CONF_DIR = os.path.join(DATA_DIR, "client")
    CLIENTS_DIR = os.path.join(ROOT_DIR, "clients")
    CLIENT_METRICS_CONFIG = os.path.join(CLIENTS_DIR, "metrics", "config", "haproxy.yaml")

    # DATASET
    DATASET_MAIN_PATH = os.path.join("/", "opt", "test_data")

    # CONFIGURATIONS FILES
    HAPROXY_DIR = os.path.join(CLIENT_CONF_DIR, "haproxy")
    HAPROXY_CONF = os.path.join(HAPROXY_DIR, "haproxy.cfg")
    HAPROXY_CONF_TEMP = os.path.join(config.tmp_dir, "haproxy.cfg")

    NGINX_DIR = os.path.join(CLIENT_CONF_DIR, "nginx")
    NGINX_CONF = os.path.join(NGINX_DIR, "conf.d", "default.conf")

    NGINX_EXPORTER_DIR = os.path.join(CLIENT_CONF_DIR, "nginx-exporter")
    NGINX_EXPORTER_CONF = os.path.join(NGINX_EXPORTER_DIR, "config.hcl")

    # SERVER REMOTE PATHS
    REMOTE_ROOT_DIR = os.path.join("/", "root")
    OVMS_TEST = os.path.join("ovms-test")
    OVMS_TEST_REMOTE = os.path.join(REMOTE_ROOT_DIR, OVMS_TEST)
    HAPROXY_DIR_REMOTE = os.path.join(OVMS_TEST_REMOTE, "haproxy")
    HAPROXY_CONFIG_REMOTE = os.path.join(HAPROXY_DIR_REMOTE, "haproxy.cfg")

    # PERFORMANCE
    PERFORMANCE_RESULTS_FILE = "performance_results.json"
    PERFORMANCE_RESULTS_PATH = os.path.dirname(config.artifacts_dir)

    DESCRIPTION_JSON_DIR = os.path.join("data", "models_descriptions")

    TEST_DATA = "/opt/test_data"

    test_root_directory = os.path.join("tests")
    test_directories = {
        "test_smoke": os.path.join(test_root_directory, "test_smoke"),
        "test_components": os.path.join(test_root_directory, "test_components"),
        "test_cli": os.path.join(test_root_directory, "test_cli"),
        "test_functional": os.path.join(test_root_directory, "test_functional"),
    }

    COMMON_GIT_CLONE_LOCK_FILE = os.path.join(config.ovms_file_locks_dir, "common_git_clone.lock")

    COMMON_BUILD_LOCK_FILE = os.path.join(config.ovms_file_locks_dir, "common_build.lock")

    # Use single shared lock file until ensure that those builds can be done concurrently.
    DOCKER_BUILD_LOCK_FILE = COMMON_BUILD_LOCK_FILE
    CUSTOM_NODE_BUILD_LOCK_FILE = COMMON_BUILD_LOCK_FILE
    CPU_EXTENSION_BUILD_LOCK_FILE = COMMON_BUILD_LOCK_FILE

    COMMON_DOWNLOAD_LOCK_FILE = os.path.join(config.ovms_file_locks_dir, "common_download.lock")

    # OVMS-C FILES
    LLM_EXPORT_MODELS_DIR = os.path.join(config.ovms_c_repo_path, "demos", "common", "export_models")
    LLM_EXPORT_MODELS_REQUIREMENTS = os.path.join(LLM_EXPORT_MODELS_DIR, "requirements.txt")
    LLM_EXPORT_MODELS_SCRIPT = os.path.join(LLM_EXPORT_MODELS_DIR, "export_model.py")

    # ACCURACY
    ACCURACY_REQUIREMENTS_DIR = os.path.join(config.ovms_test_repo_path, "llm", "accuracy")
    LM_EVAL_HARNESS_WORK_DIR = os.path.join(config.tmp_dir, "lm_evaluation_harness")
    LM_EVAL_HARNESS_VENV_DIR = os.path.join(LM_EVAL_HARNESS_WORK_DIR, ".venv")
    LM_EVAL_HARNESS_REQUIREMENTS = os.path.join(ACCURACY_REQUIREMENTS_DIR, "requirements_lm_evaluation_harness.txt")
    LMMS_EVAL_WORK_DIR = os.path.join(config.tmp_dir, "lmms-eval")
    LMMS_EVAL_VENV_DIR = os.path.join(LMMS_EVAL_WORK_DIR, ".venv")
    LMMS_EVAL_REQUIREMENTS = os.path.join(ACCURACY_REQUIREMENTS_DIR, "requirements_lmms_eval.txt")
    BFCL_WORK_DIR = os.path.join(config.tmp_dir, "gorilla")
    BFCL_VENV_DIR = os.path.join(BFCL_WORK_DIR, ".venv")
    MTEB_WORK_DIR = os.path.join(config.tmp_dir, "mteb")
    MTEB_VENV_DIR = os.path.join(MTEB_WORK_DIR, ".venv")
    MTEB_REQUIREMENTS = os.path.join(ACCURACY_REQUIREMENTS_DIR, "requirements_mteb.txt")
    OPEN_ASR_LEADERBOARD_WORK_DIR = os.path.join(config.tmp_dir, "open_asr_leaderboard")
    OPEN_ASR_LEADERBOARD_VENV_DIR = os.path.join(OPEN_ASR_LEADERBOARD_WORK_DIR, ".venv")
    OPEN_ASR_LEADERBOARD_API_DIR = os.path.join(OPEN_ASR_LEADERBOARD_WORK_DIR, "api")
    OPEN_ASR_LEADERBOARD_RUN_EVAL = os.path.join(OPEN_ASR_LEADERBOARD_API_DIR, "run_eval.py")
    OPEN_ASR_LEADERBOARD_RUN_API_SCRIPT = os.path.join(OPEN_ASR_LEADERBOARD_API_DIR, "run_api.sh")
    OPEN_ASR_LEADERBOARD_REQUIREMENTS = os.path.join(
        OPEN_ASR_LEADERBOARD_WORK_DIR, "requirements", "requirements.txt"
    )
    OPEN_ASR_LEADERBOARD_REQUIREMENTS_API = os.path.join(
        OPEN_ASR_LEADERBOARD_WORK_DIR, "requirements", "requirements-api.txt"
    )

    CORRUPTED_DLL_PATH = os.path.join(config.ovms_test_repo_path, "data", "dll", "corrupted_library.dll")

    # AUDIO FILES
    REFERENCE_AUDIO_FILES_DIR = os.path.join(config.datasets_path, "audio", "reference_audio")
    REFERENCE_AUDIO_DATA_DIR = os.path.join(config.ovms_test_repo_path, "data", "llm", "audio", "reference_texts")

    OVMS_C_PREPARE_LLM_MODELS_FILE = os.path.join(config.ovms_c_repo_path, "prepare_llm_models.sh")

    @staticmethod
    def CAPI_WRAPPER_PACKAGE_CONTENT_PATH(base_os):
        return os.path.join(config.c_api_wrapper_dir, base_os, "ovms")

    @property
    def VR_NOT_EXECUTED_YET_UPLOAD_FINISHED_MUTEX(self):
        return os.path.join(config.tmp_repos_dir, "vr_not_executed_yet_upload_finished.mutex")

    @staticmethod
    def get_target_device_lock_file(target_device, i):
        if isinstance(target_device, str):
            assert not all(x in target_device for x in [TargetDevice.GPU, TargetDevice.NPU])

            # generalize HETERO/AUTO/MUTLI:X => `X`
            if TargetDevice.GPU in target_device:
                return os.path.join(config.ovms_file_locks_dir, f"target_device_{TargetDevice.GPU}_{i}.lock")
            if TargetDevice.NPU in target_device:
                return os.path.join(config.ovms_file_locks_dir, f"target_device_{TargetDevice.NPU}_{i}.lock")

        return os.path.join(config.ovms_file_locks_dir, f"target_device_{target_device}_{i}.lock")


def any_is_relative_to(paths, subpath):
    return any([_path in subpath for _path in paths])


performance_test_directories = ["tests/benchmark"]

uat_test_directories = ["tests/non_functional/documentation"]

stress_and_load_test_directories = ["tests/non_functional/load"]

operator_test_files = ["tests/functional_extended/test_ovms_operator.py"]

provisioning_tests_files = ["tests/non_functional/utils/test_provisioning.py"]

ov_tests_files = ["tests/non_functional/openvino"]
