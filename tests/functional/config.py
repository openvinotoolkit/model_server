#
# Copyright (c) 2020-2026 Intel Corporation
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

from tests.functional.constants.os_type import OsType
from tests.functional.constants.target_device import TargetDevice
from tests.functional.utils.core import TmpDir
from tests.functional.utils.helpers import (
    generate_test_object_name,
    get_bool,
    get_int,
    get_list,
    get_path,
    get_target_devices,
)


try:
    # In user_config.py, user might export custom environment variables
    import user_config
except ImportError:
    pass

"""TEST_DIR -  location where models and test data should be copied from TEST_DIR_CACHE and deleted after tests"""
test_dir = os.environ.get("TEST_DIR", "/tmp/{}".format(generate_test_object_name(prefix='ovms_models')))

"""TEST_DIR_CACHE -  location where models and test data should be downloaded to and serve as cache for TEST_DIR"""
test_dir_cache = os.environ.get("TEST_DIR_CACHE", "/tmp/ovms_models_cache")

"""TEST_DIR_CLEANUP - if set to True, TEST_DIR directory will be removed after tests execution"""
test_dir_cleanup = os.environ.get("TEST_DIR_CLEANUP", "True")
test_dir_cleanup = test_dir_cleanup.lower() == "true"

"""BUILD_LOGS -  path to dir where artifacts should be stored"""
artifacts_dir = get_path("BUILD_LOGS", os.path.join("~", "ovms_test_logs"))

""" TT_NGNIX_CERTS_DIR - Custom nodes directory path"""
nginx_certs_dir = get_path("TT_NGINX_CERTS_DIR", os.path.join("~", "ovms_nginx_certs"))

""" TT_DATASETS_PATH - Datasets local repo path"""
datasets_path = get_path("TT_DATASETS_PATH", os.path.join("~", "ovms_datasets"))

""" TT_CLEAN_ARTIFACTS_DIR """
clean_artifacts_dir = get_bool("TT_CLEAN_ARTIFACTS_DIR", False)

""" TT_LANGUAGE_MODELS_ENABLED - model UniversalSentenceEncoder added to various models """
language_models_enabled = get_bool("TT_LANGUAGE_MODELS_ENABLED", True)

""" MEDIAPIE_DISABLE - if OVMS image has mediapipe feature """
mediapipe_disable = bool(get_int("MEDIAPIPE_DISABLE", 0))

""" PYTHON_DISABLE - if OVMS image has python feature """
python_disable = bool(get_int("PYTHON_DISABLE", 0))

""" TT_WIN_PY_VERSION - Python version for virtualenv on Windows OS """
windows_python_version = os.environ.get("TT_WIN_PY_VERSION", "3.12")

""" TT_DOCKER_REGISTRY - Docker registry"""
docker_registry = os.environ.get("TT_DOCKER_REGISTRY", "registry.toolbox.iotg.sclab.intel.com")

""" OVMS_CPP_DOCKER_IMAGE """
ovms_cpp_docker_image = os.environ.get("OVMS_CPP_DOCKER_IMAGE", None)

""" TT_OVMS_IMAGE_NAME """
ovms_image = os.environ.get("TT_OVMS_IMAGE_NAME", None)

""" OVMS_CPP_IMAGE_TAG - tag of OVMS image to test (compatible with build parameter) """
ovms_image_tag = os.environ.get("OVMS_CPP_IMAGE_TAG", None)

""" TT_OVMS_TEST_IMAGE_NAME - image name for cpu extensions and custom nodes """
ovms_test_image_name = os.environ.get("TT_OVMS_TEST_IMAGE_NAME", None)

""" TT_FORCE_USE_OVMS_IMAGE - force to use TT_OVMS_IMAGE_NAME """
force_use_ovms_image = get_bool("TT_FORCE_USE_OVMS_IMAGE", False)

""" TT_OVMS_C_RELEASE_ARTIFACTS_PATH - path to current release artifacts """
# multiple local and remote location supported
# example: TT_OVMS_C_RELEASE_ARTIFACTS_PATH="../ubuntu24/ovms.tar.gz,../redat/ovms.tar.gz"
ovms_c_release_artifacts_path = get_list("TT_OVMS_C_RELEASE_ARTIFACTS_PATH")

"""START_CONTAINER_COMMAND - command to start ovms container"""
start_container_command = os.environ.get("START_CONTAINER_COMMAND", "")

"""CONTAINER_LOG_LINE - log line to check in container"""
# For multiple log lines, pass them separated with ':'
container_log_line = os.environ.get("CONTAINER_LOG_LINE", "Started model manager thread")
container_log_line = container_log_line.split(":")

"""OVMS_BINARY_PATH - path to ovms binary file; when specified, tests are executed against provided binary."""
ovms_binary_path = os.environ.get("OVMS_BINARY_PATH", None)

"""LOG_LEVEL - set log level """
log_level = os.environ.get("LOG_LEVEL", "INFO")

path_to_mount = os.path.join(test_dir, "saved_models")
os.makedirs(path_to_mount, exist_ok=True)

path_to_mount_cache = os.path.join(test_dir_cache, "saved_models")

models_path = path_to_mount if ovms_binary_path else "/opt/ml"

"""TT_MINIO_IMAGE_NAME - Docker image for Minio"""
minio_image = os.environ.get("TT_MINIO_IMAGE_NAME", "minio/minio:latest")

""" TT_TARGET_DEVICE - list of devices separated by a comma "CPU,GPU,NPU" """
target_devices = get_target_devices()
target_device = target_devices[0]

"""IMAGE - docker image name which should be used to run tests"""
if target_device == TargetDevice.GPU:
    _default_image = "openvino/model_server-gpu"
else:
    _default_image = "openvino/model_server"
image = os.environ.get("IMAGE", _default_image)

start_minio_container_command = 'server --address ":{}" /data'

container_minio_log_line = "Console endpoint is listening on a dynamic port"

# Reservation manager values, for details study common_libs.reservation_manager
""" TT_GRPC_OVMS_STARTING_PORT - Grpc port where ovms should be exposed"""
grpc_ovms_starting_port = get_int("TT_GRPC_OVMS_STARTING_PORT", None)

""" TT_REST_OVMS_STARTING_PORT - Rest port where ovms should be exposed"""
rest_ovms_starting_port = get_int("TT_REST_OVMS_STARTING_PORT", None)

""" TT_PORTS_POOL_SIZE- Ports pool size"""
ports_pool_size = get_int("TT_PORTS_POOL_SIZE", None)
# NOTE: Above values will be validated and could be changed if invalid

""" TT_CONVERTED_MODELS_EXPIRE_TIME - Time after converted models are not up-to-date and needs to be refreshed(s) """
converted_models_expire_time = get_int("TT_CONVERTED_MODELS_EXPIRE_TIME", 7*24*3600)  # Set default to one week

""" TT_DEFAULT_INFER_TIMEOUT - Timeout for CPU target device"""
default_infer_timeout = get_int("TT_DEFAULT_INFER_TIMEOUT", 10)

""" TT_DEFAULT_GPU_INFER_TIMEOUT - Timeout for GPU target device"""
default_gpu_infer_timeout = get_int("TT_DEFAULT_GPU_INFER_TIMEOUT", 10*default_infer_timeout)

""" TT_DEFAULT_NPU_INFER_TIMEOUT - Timeout for NPU target device"""
default_npu_infer_timeout = get_int("TT_DEFAULT_NPU_INFER_TIMEOUT", 10*default_infer_timeout)

""" INFER TIMEOUT """
infer_timeouts = {
    TargetDevice.CPU: default_infer_timeout,
    TargetDevice.GPU: default_gpu_infer_timeout,
    TargetDevice.NPU: default_npu_infer_timeout,
    TargetDevice.AUTO: default_gpu_infer_timeout,
    TargetDevice.HETERO: default_gpu_infer_timeout,
    TargetDevice.AUTO_CPU_GPU: default_gpu_infer_timeout,
}
infer_timeout = infer_timeouts[target_device]

""" TT_IS_NGINX_MTLS - Specify if given image is OVSA nginx mtls image. """
is_nginx_mtls = get_bool("TT_IS_NGINX_MTLS", False)

""" TT_SKIP_TEST_IF_IS_NGINX_MTLS """
skip_nginx_test = get_bool("TT_SKIP_TEST_IF_IS_NGINX_MTLS", "True")
skip_nginx_test = skip_nginx_test and is_nginx_mtls

""" TT_ENABLE_OVMS_C_PYTEST_PLUGINS - enable pytest plugins """
enable_pytest_plugins = get_bool("TT_ENABLE_OVMS_C_PYTEST_PLUGINS", "True")

""" TT_OVMS_C_REPO_PATH - path to ovms-c repository. Can be relative or absolute. """
ovms_c_repo_path = get_path("TT_OVMS_C_REPO_PATH", get_path("PWD", "./"))

""" TT_REPOSITORY_NAME - repository name provided by user """
repository_name = os.environ.get("TT_REPOSITORY_NAME", "ovms-c")

""" TT_ENVIRON_NAME - Environment name to be used while reporting test results
                      to be presented on test reports as a environment name."""
environment_name = os.environ.get("TT_ENVIRONMENT_NAME", "")

""" TT_PRODUCT_BUILD_NUMBER  - Test product build number provided by user (last number from version - 0.8.0.XXXX)"""
product_build_number = os.environ.get("TT_PRODUCT_BUILD_NUMBER", "1")

""" TT_PRODUCT_BUILD_FROM_ENV - If set to True, environment build number is taken from tested environment,
                                If set to False environment build number is taken from TT_PRODUCT_BUILD_NUMBER
                                If TT_PRODUCT_BUILD_NUMBER not set default environment build number is taken"""
product_build_number_from_env = get_bool("TT_PRODUCT_BUILD_FROM_ENV", True)

""" TT_PRODUCT_VERSION - Environment version provided by user"""
product_version = os.environ.get("TT_PRODUCT_VERSION", "1.0.0")

""" TT_PRODUCT_VERSION_FROM_ENV - If set to True, version is taken from tested environment,
                                  If set to False version is taken from TT_PRODUCT_VERSION
                                  If TT_PRODUCT_VERSION not set default version is taken"""
product_version_number_from_env = get_bool("TT_PRODUCT_VERSION_FROM_ENV", False)

""" TT_PRODUCT_VERSION_SUFFIX - Environment version suffix provided by user"""
product_version_suffix = os.environ.get("TT_PRODUCT_VERSION_SUFFIX", "ovms")

""" TT_DELAY_BETWEEN_TESTS - Time of pause between test case runs"""
delay_between_test = get_int("TT_DELAY_BETWEEN_TESTS", 0)

""" TEST_TIMEOUT - default timeout (number of hours) for whole test session inherited from CI """
pytest_global_session_timeout = get_int("TEST_TIMEOUT", 15)

""" TT_BUILD_TEST_IMAGE - build ovms test image (cpu extensions, custom nodes etc.) """
build_test_image = get_bool("TT_BUILD_TEST_IMAGE", False)

""" TT_RUN_OVMS_WITH_VALGRIND - run ovms using Valgrind """
run_ovms_with_valgrind = get_bool("TT_RUN_OVMS_WITH_VALGRIND", False)

""" TT_RUN_OVMS_WITH_OPENCL_TRACE - run OVMS with cliloader """
run_ovms_with_opencl_trace = get_bool("TT_RUN_OVMS_WITH_OPENCL_TRACE", False)

""" TT_SERVER_ADDRESS - OVMS server address"""
server_address = os.environ.get("TT_SERVER_ADDRESS", "localhost")

""" TT_RESOURCE_MONITOR_ENABLED - Dump ovms container resource statistics once per second """
resource_monitor_enabled = get_bool("TT_RESOURCE_MONITOR_ENABLED", False)

""" TT_TEST_TEMP_DIR - directory path where all temporary files are stored, default is not set """
test_temp_dir = os.environ.get("TT_TEST_TEMP_DIR", None)
tmp_dir = TmpDir(test_temp_dir)

"""TT_LOGGING_LEVEL_OVMS - ovms docker default log level, default: INFO"""
logging_level_ovms = os.environ.get("TT_LOGGING_LEVEL_OVMS", "INFO")

"""TT_CONTAINER_PROXY - Proxy settings to be used in container """
container_proxy = os.environ.get("TT_CONTAINER_PROXY", os.environ.get("http_proxy", ""))

"""TT_DISABLE_DMESG_LOG_MONITOR"""
disable_dmesg_log_monitor = get_bool("TT_DISABLE_DMESG_LOG_MONITOR", False)

""" TT_MACHINE_IS_RESERVED_FOR_TEST_SESSION """
machine_is_reserved_for_test_session = get_bool("TT_MACHINE_IS_RESERVED_FOR_TEST_SESSION", False)

""" TT_WAIT_FOR_MESSAGES_TIMEOUT - timeout for ovms.wait_for_messages(...) method """
wait_for_messages_timeout = get_int("TT_WAIT_FOR_MESSAGES_TIMEOUT", 180)

""" TT_AIRPLANE_MODE - disable connecting to remote resources, disable all downloads and docker pull/build commands and
                       expect that all required data is available locally. """
airplane_mode = get_bool("TT_AIRPLANE_MODE", False)

""" TT_OVMS_IMAGE_LOCAL - ovms image can only be found locally """
ovms_image_local = get_bool("TT_OVMS_IMAGE_LOCAL", False)

""" TT_BASE_OS - os type used for calculating ovms_image name (if not given explicitly). 
        Possible options (case insensitive): 
        ubuntu22 - use default Ubuntu 22.04 image
        ubuntu24 - use default Ubuntu 24.04 image
        redhat - use UBI 8.10 based
        ubuntu22,ubuntu24,redhat - iterate all tests both for ubuntu and redhat   
        windows - can't iterate (supports only BINARY ovms type)
"""
__base_os = os.environ.get("BASE_OS", OsType.Ubuntu24)
base_os = get_list("TT_BASE_OS", fallback=[__base_os])

""" TT_ENABLE_PLUGIN_CONFIG_TARGET_DEVICE - use plugin_config globally set for target devices """
enable_plugin_config_target_device = get_bool("TT_ENABLE_PLUGIN_CONFIG_TARGET_DEVICE", False)

"""TT_DISABLE_CUSTOM_LOADER"""
disable_custom_loader = get_bool("TT_DISABLE_CUSTOM_LOADER", True)

""" TT_CUSTOM_NODES - Custom nodes directory path"""
custom_nodes_path = get_path("TT_CUSTOM_NODES", os.path.join("~", "ovms_custom_nodes"))

""" TT_BINARY_IO_IMAGES_PATH - Datasets local repo path"""
binary_io_images_path = get_path("TT_BINARY_IO_IMAGES_PATH", os.path.join("~", "ovms_binary_io"))

""" TT_KV_CACHE_SIZE - memory size in GB for storing KV cache """
kv_cache_size_value = get_int("TT_KV_CACHE_SIZE", 0)

""" TT_KV_CACHE_PRECISION - Reduced kv cache precision to u8 lowers the cache size consumption. """
kv_cache_precision_value = os.environ.get("TT_KV_CACHE_PRECISION", None)

""" "MEDIAPIPE_REPO_BRANCH" - https://github.com/openvinotoolkit/mediapipe/ branch name """
mediapipe_repo_branch = os.environ.get("MEDIAPIPE_REPO_BRANCH", "main")

""" TT_MAX_NUM_BATCHED_TOKENS -  max number of tokens processed in a single iteration """
max_num_batched_tokens = get_int("TT_MAX_NUM_BATCHED_TOKENS", None)

""" TT_PIPELINE_TYPE - pipeline type in LLM graph node_options, e.g. VLM, LM """
pipeline_type = os.environ.get("TT_PIPELINE_TYPE", None)

""" TT_ENABLE_PREFIX_CACHING - enable prefix caching for model """
enable_prefix_caching_config = get_bool("TT_ENABLE_PREFIX_CACHING", False)
