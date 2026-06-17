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
import re
from pathlib import Path

from tests.functional.constants.os_type import OsType
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.target_device import TargetDevice
from tests.functional.utils.core import TmpDir
from tests.functional.utils.helpers import (
    generate_test_object_name,
    get_bool,
    get_int,
    get_list,
    get_path,
    get_target_devices,
    validate_supported_values,
)


try:
    # In user_config.py, user might export custom environment variables
    import user_config
except ImportError:
    pass


def get_uses_mapping():
    _uses_mapping = get_list("TT_USES_MAPPING", fallback=[None])
    _uses_mapping = list(set([str(x).upper() for x in _uses_mapping]))  # make upper & remove duplicates
    # Reduce to True/False/None
    _uses_mapping = [x == "TRUE" if x in ["TRUE", "FALSE"] else None for x in _uses_mapping]
    validate_supported_values(_uses_mapping, [True, False, None])
    return _uses_mapping


"""
    TT_USES_MAPPING - use mapping JSON for model inputs/output name aliasing
    Possible TT_USES_MAPPING values (case insensitive):
    - (empty)/""/NONE - Default leave mapping.json provided alongside model untouched (if exists).
    - FALSE - forcibly remove mapping.json if provided with model.
    - TRUE - remove any previous mapping and add generic mapping.json 
            (see: ovms/object_model/ovms_mapping_config.py for details)
    - TRUE,FALSE,NONE - Iterate each test case from listed values in single test session.
"""
uses_mapping = get_uses_mapping()

"""TEST_DIR -  location where models and test data should be copied from TEST_DIR_CACHE and deleted after tests"""
test_dir = os.environ.get("TEST_DIR", "/tmp/{}".format(generate_test_object_name(prefix='ovms_models')))

"""TEST_DIR_CACHE -  location where models and test data should be downloaded to and serve as cache for TEST_DIR"""
test_dir_cache = os.environ.get("TEST_DIR_CACHE", "/tmp/ovms_models_cache")

"""TEST_DIR_CLEANUP - if set to True, TEST_DIR directory will be removed after tests execution"""
test_dir_cleanup = os.environ.get("TEST_DIR_CLEANUP", "True")
test_dir_cleanup = test_dir_cleanup.lower() == "true"

""" TT_OVMS_C_REPO_PATH - path to ovms-c repository. Can be relative or absolute. """
ovms_c_repo_path = get_path("TT_OVMS_C_REPO_PATH", get_path("PWD", "./"))

""" TT_SETUPVARS_SCRIPT_PATH - path to setupvars.bat script """
setupvars_script_path = os.environ.get(
    "TT_SETUPVARS_SCRIPT_PATH", os.path.join(ovms_c_repo_path, "setupvars.bat")
)

"""BUILD_LOGS -  path to dir where artifacts should be stored"""
artifacts_dir = get_path("BUILD_LOGS", os.path.join(ovms_c_repo_path, "tests", "functional", "test_log_build"))

""" TT_NGNIX_CERTS_DIR - Custom nodes directory path"""
nginx_certs_dir = get_path("TT_NGINX_CERTS_DIR", os.path.join("~", "ovms_nginx_certs"))

""" TT_MODELS_PATH - Models local repo path"""
models_path = get_path("TT_MODELS_PATH", os.path.join("~", "ovms_models"))

""" TT_DATASETS_PATH - Datasets local repo path"""
datasets_path = get_path("TT_DATASETS_PATH", os.path.join("~", "ovms_datasets"))

""" TT_CLEAN_ARTIFACTS_DIR """
clean_artifacts_dir = get_bool("TT_CLEAN_ARTIFACTS_DIR", False)

""" TT_LANGUAGE_MODELS_ENABLED - model UniversalSentenceEncoder added to various models """
language_models_enabled = get_bool("TT_LANGUAGE_MODELS_ENABLED", True)

""" MEDIAPIE_DISABLE - if OVMS image has mediapipe feature """
mediapipe_disable = bool(get_int("MEDIAPIPE_DISABLE", 0))

""" PYTHON_DISABLE - if OVMS image has Python feature disabled"""
python_disable = bool(get_int("PYTHON_DISABLE", 0))

""" TT_WIN_PY_VERSION - Python version for virtualenv on Windows OS """
windows_python_version = os.environ.get("TT_WIN_PY_VERSION", "3.12")

""" TT_DOCKER_REGISTRY - Docker registry"""
docker_registry = os.environ.get("TT_DOCKER_REGISTRY", None)

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

""" TT_MINIO_IMAGE_NAME - Docker image for Minio"""
minio_image = os.environ.get(
    "TT_MINIO_IMAGE_NAME",
    f"{docker_registry}/minio/minio:latest" if docker_registry is not None else "minio/minio:latest",
)

""" TT_TARGET_DEVICE - list of devices separated by a comma "CPU,GPU,NPU" """
target_devices = get_target_devices()

# Reservation manager values, for details study tests.functional.utils.reservation_manager
""" TT_GRPC_OVMS_STARTING_PORT - Grpc port where ovms should be exposed"""
grpc_ovms_starting_port = get_int("TT_GRPC_OVMS_STARTING_PORT", None)

""" TT_REST_OVMS_STARTING_PORT - Rest port where ovms should be exposed"""
rest_ovms_starting_port = get_int("TT_REST_OVMS_STARTING_PORT", None)

""" TT_PORTS_POOL_SIZE- Ports pool size"""
ports_pool_size = get_int("TT_PORTS_POOL_SIZE", None)
# NOTE: Above values will be validated and could be changed if invalid

""" TT_IS_NGINX_MTLS - Specify if given image is OVSA nginx mtls image. """
is_nginx_mtls = get_bool("TT_IS_NGINX_MTLS", False)

""" TT_FORCE_GENERATE_NEW_SSL_CERTIFICATES """
force_generate_new_ssl_certs = get_bool("TT_FORCE_GENERATE_NEW_SSL_CERTIFICATES", True)

""" TT_SKIP_TEST_IF_IS_NGINX_MTLS """
skip_nginx_test = get_bool("TT_SKIP_TEST_IF_IS_NGINX_MTLS", True)
skip_nginx_test = skip_nginx_test and is_nginx_mtls

""" TT_ENABLE_OVMS_C_PYTEST_PLUGINS - enable pytest plugins """
enable_pytest_plugins = get_bool("TT_ENABLE_OVMS_C_PYTEST_PLUGINS", True)

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

""" TT_SAVE_IMAGE_TO_ARTIFACTS - save generated or edited image to artifacts """
save_image_to_artifacts = get_bool("TT_SAVE_IMAGE_TO_ARTIFACTS", False)

"""TT_SET_NO_PROXY"""
set_no_proxy = get_bool("TT_SET_NO_PROXY", True)
no_proxy = os.environ.get("no_proxy", "")
if set_no_proxy:
    os.environ["NO_PROXY"] = no_proxy
http_proxy = os.environ.get("http_proxy", "")
https_proxy = os.environ.get("https_proxy", "")

""" TT_RUN_OVMS_WITH_VALGRIND - run ovms using Valgrind """
run_ovms_with_valgrind = get_bool("TT_RUN_OVMS_WITH_VALGRIND", False)

""" TT_RUN_OVMS_WITH_OPENCL_TRACE - run OVMS with cliloader """
run_ovms_with_opencl_trace = get_bool("TT_RUN_OVMS_WITH_OPENCL_TRACE", False)

""" TT_XDIST_WORKERS - number of workers """
xdist_workers = get_int("TT_XDIST_WORKERS", 0)

""" TT_DOCKER_CLIENT_TIMEOUT - Docker client timeout"""
docker_client_timeout = get_int("TT_DOCKER_CLIENT_TIMEOUT", 120)

""" TT_SERVER_ADDRESS - OVMS server address"""
server_address = os.environ.get("TT_SERVER_ADDRESS", "localhost")

""" TT_RESOURCE_MONITOR_ENABLED - Dump ovms container resource statistics once per second """
resource_monitor_enabled = get_bool("TT_RESOURCE_MONITOR_ENABLED", True)

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

""" TT_CLEANUP_ENVIRONMENT_ON_STARTUP """
cleanup_env_on_startup = get_bool("TT_CLEANUP_ENVIRONMENT_ON_STARTUP", False)

""" TT_TEARDOWN_DOCKER_IMAGES - at teardown remove docker images build during test session """
teardown_docker_images = get_bool("TT_TEARDOWN_DOCKER_IMAGES", True)

""" TT_TEARDOWN_DOCKER_CONTAINERS - at teardown remove stopped docker containers """
teardown_docker_containers = get_bool("TT_TEARDOWN_DOCKER_CONTAINERS", False)

""" TT_TEARDOWN_OVMS_PROCESSES - at teardown remove all ovms.exe processes """
teardown_ovms_processes = get_bool("TT_TEARDOWN_OVMS_PROCESSES", False)

""" TT_WAIT_FOR_MESSAGES_TIMEOUT - timeout for ovms.wait_for_messages(...) method """
wait_for_messages_timeout = get_int("TT_WAIT_FOR_MESSAGES_TIMEOUT", 180)

""" TT_WAIT_FOR_MESSAGES_LOG_FLUSH - dump collected OVMS output to logger on interrupt/timeout (default: True) """
wait_for_messages_log_flush = get_bool("TT_WAIT_FOR_MESSAGES_LOG_FLUSH", True)

""" TT_WAIT_FOR_MESSAGES_LOG_FLUSH_MAX_LINES - max OVMS lines to dump on interrupt (0 = unlimited) """
wait_for_messages_log_flush_max_lines = get_int("TT_WAIT_FOR_MESSAGES_LOG_FLUSH_MAX_LINES", 500)

""" TT_WAIT_FOR_MESSAGES_EXCEPTION_MAX_LINES - max OVMS lines included in exception message (0 = unlimited) """
wait_for_messages_exception_max_lines = get_int("TT_WAIT_FOR_MESSAGES_EXCEPTION_MAX_LINES", 500)

""" TT_WAIT_FOR_MESSAGES_PROGRESS_INTERVAL - interval in seconds for periodic progress logging (default: 60) """
wait_for_messages_progress_interval = get_int("TT_WAIT_FOR_MESSAGES_PROGRESS_INTERVAL", 60)

""" TT_AIRPLANE_MODE - disable connecting to remote resources, disable all downloads and docker pull/build commands and
                       expect that all required data is available locally. """
airplane_mode = get_bool("TT_AIRPLANE_MODE", False)

""" TT_OVMS_IMAGE_LOCAL - ovms image can only be found locally """
ovms_image_local = get_bool("TT_OVMS_IMAGE_LOCAL", False)

""" TT_REQUIREMENTS - Requirements """
req_ids = get_list("TT_REQUIREMENTS")

""" TT_EXCLUDE_REQUIREMENTS - Requirements to exclude """
exclude_req_ids = get_list("TT_EXCLUDE_REQUIREMENTS")

""" TT_COMPONENTS - Components """
components_ids = get_list("TT_COMPONENTS")

""" TT_EXCLUDE_COMPONENTS - Components to exclude """
exclude_components_ids = get_list("TT_EXCLUDE_COMPONENTS")

""" TT_TESTS_PRIORITY_LIST - tests priority to run - high, medium or low """
tests_priority_list_raw = get_list("TT_TESTS_PRIORITY_LIST", fallback=["high", "medium", "low"])
tests_priority_list = [f"priority_{p}" for p in tests_priority_list_raw if "priority" not in p]

""" TT_PERFORMANCE_TEST_TIMEOUT_MINUTES - timeout (in minutes) for each performance test """
performance_test_timeout_minutes = get_int("TT_PERFORMANCE_TEST_TIMEOUT_MINUTES", 10)

""" TT_BASE_OS - os type used for calculating ovms_image name (if not given explicitly). 
        Possible options (case insensitive): 
        ubuntu22 - use default Ubuntu 22.04 image
        ubuntu24 - use default Ubuntu 24.04 image
        redhat - use UBI 9.7 based
        ubuntu22,ubuntu24,redhat - iterate all tests both for ubuntu and redhat   
        windows - can't iterate (supports only BINARY ovms type)
"""
__base_os = os.environ.get("BASE_OS", OsType.Ubuntu24)
base_os = get_list("TT_BASE_OS", fallback=[__base_os])

""" GLOBAL_TEMP_DIR - global temporary directory """
global_tmp_dir_default = os.path.join("~", "AppData", "Local", "Temp") if OsType.Windows in base_os else "/tmp"
global_tmp_dir = get_path("GLOBAL_TEMP_DIR", global_tmp_dir_default)

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

""" TT_SSL_VALIDATION - if set to True for https request ssl protocol validation is performed,
                           default False"""
ssl_validation = get_bool("TT_SSL_VALIDATION", False)

"""TT_LOGGED_RESPONSE_BODY_LENGTH - length of http response logged , default: 1024 """
logged_response_body_length = os.environ.get("TT_LOGGED_RESPONSE_BODY_LENGTH", 1024)

""" TT_C_API_WRAPPER_DIR - Cython wrapper files for C_API """
c_api_wrapper_dir = get_path("TT_C_API_WRAPPER_DIR", os.path.join("~", "ovms_c_api_wrapper_dir"))

""" TT_OVMS_FILE_LOCKS_DIR """
ovms_file_locks_dir = get_path("TT_OVMS_FILE_LOCKS_DIR", os.path.join("~", "ovms_locks"))

""" TT_USE_LEGACY_MODELS """
use_legacy_models = get_bool("TT_USE_LEGACY_MODELS", True)


class StrippingLists:
    DEFAULT_SENSITIVE_KEYS_TO_BE_MASKED = [
        r"(?!zabbix_operator_initial_).*pass(word)?",
        r".*client_id",
        r".*(access)?(_)?(?<!ssh_)key(?!s|_path)",
        r"id_token",
        r"Authorization",
        r"database_url",
        r"gmail_",
    ]


""" Fields for logger """
host_os_user = os.environ.get("TT_HOST_OS_USER", None)
log_username = os.environ.get("TT_LOG_USERNAME", False)
sensitive_keys = get_list("TT_SENSITIVE_KEYS", fallback=StrippingLists.DEFAULT_SENSITIVE_KEYS_TO_BE_MASKED)
sensitive_keys_to_be_masked = re.compile("|".join(sensitive_keys), re.IGNORECASE)
strip_sensitive_data = get_bool("TT_STRIP_SENSITIVE_DATA", False)
logging_level = os.environ.get("TT_LOGGING_LEVEL", "INFO")
logger_format = "{}%(asctime)s {}- %(name)s - %(levelname)s: %(message)s".format

"""TEST_LOG_DIR - directory path where all logs are stored, default test_log"""
test_log_directory = get_path("TEST_LOG_DIR", Path().absolute() / "test_log")
Path(test_log_directory).mkdir(exist_ok=True, parents=True)
test_log_directory = str(test_log_directory)


def get_ovms_types():
    ovms_types_list = get_list("TT_OVMS_TYPE", fallback=[OvmsType.DOCKER])
    ovmstype_values = [value for value in vars(OvmsType).values() if isinstance(value, str) and value.isupper()]
    ovms_types_list = validate_supported_values(detected_list=ovms_types_list, supported_list=ovmstype_values)
    return ovms_types_list


""" TT_OVMS_TYPE - ovms type runtime to be executed:
DOCKER, BINARY, BINARY_DOCKER, CAPI, CAPI_DOCKER, DOCKER_CMD_LINE """
ovms_types = get_ovms_types()

""" TT_DIVIDE_TARGET_DEVICE_PER_WORKER - spread tests across pytest workers based on target device """
divide_target_device_per_worker = get_bool("TT_DIVIDE_TARGET_DEVICE_PER_WORKER", False)

""" TT_PYTEST_KEYWORD_FILTER """
pytest_keyword_filter = os.environ.get("TT_PYTEST_KEYWORD_FILTER", None)
