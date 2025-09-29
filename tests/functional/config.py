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

import os

from tests.functional.constants.constants import TARGET_DEVICE_CPU, TARGET_DEVICE_GPU, TARGET_DEVICE_CUDA, TARGET_DEVICE_MYRIAD, TARGET_DEVICE_HDDL
from tests.functional.utils.helpers import get_int, get_bool
from tests.functional.utils.parametrization import generate_test_object_name

try:
    # In user_config.py, user might export custom environment variables
    import user_config
except ImportError:
    pass

"""TEST_DIR -  location where models and test data should be copied from TEST_DIR_CACHE and deleted after tests"""
test_dir = os.environ.get("TEST_DIR", "/tmp/{}".format(generate_test_object_name(prefix='ovms_models')))

"""TEST_DIR_CACHE -  location where models and test data should be downloaded to and serve as cache for TEST_DIR"""
test_dir_cache = os.environ.get("TEST_DIR_CACHE", "/tmp/ovms_models_cache")

"""TEST_DIR_CLEANUP - if set to True, TEST_DIR directory will be removed
                      after tests execution"""
test_dir_cleanup = os.environ.get("TEST_DIR_CLEANUP", "True")
test_dir_cleanup = test_dir_cleanup.lower() == "true"

"""BUILD_LOGS -  path to dir where artifacts should be stored"""
artifacts_dir = os.environ.get("BUILD_LOGS", "")

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

path_to_mount_cache = os.path.join(test_dir_cache, "saved_models")

models_path = path_to_mount if ovms_binary_path else "/opt/ml"

"""TT_MINIO_IMAGE_NAME - Docker image for Minio"""
minio_image = os.environ.get("TT_MINIO_IMAGE_NAME", "minio/minio:latest")

""" TT_TARGET_DEVICE - one of "CPU", "GPU" """
target_device = os.environ.get("TT_TARGET_DEVICE", TARGET_DEVICE_CPU)

"""IMAGE - docker image name which should be used to run tests"""
if target_device == TARGET_DEVICE_GPU:
    _default_image = "openvino/model_server-gpu"
else:
    _default_image = "openvino/model_server"
image = os.environ.get("IMAGE", _default_image)

start_minio_container_command = 'server --address ":{}" /data'

container_minio_log_line = "Console endpoint is listening on a dynamic port"

""" TT_GRPC_OVMS_STARTING_PORT - Grpc port where ovms should be exposed"""
grpc_ovms_starting_port = get_int("TT_GRPC_OVMS_STARTING_PORT", 9001)

""" TT_REST_OVMS_STARTING_PORT - Rest port where ovms should be exposed"""
rest_ovms_starting_port = get_int("TT_REST_OVMS_STARTING_PORT", 18001)

""" TT_PORTS_POOL_SIZE- Ports pool size"""
ports_pool_size = get_int("TT_PORTS_POOL_SIZE", 5000)

""" TT_CONVERTED_MODELS_EXPIRE_TIME - Time after converted models are not up-to-date and needs to be refreshed(s) """
converted_models_expire_time = get_int("TT_CONVERTED_MODELS_EXPIRE_TIME", 7*24*3600)  # Set default to one week

""" TT_DEFAULT_INFER_TIMEOUT - Timeout for CPU target device"""
default_infer_timeout = get_int("TT_DEFAULT_INFER_TIMEOUT", 10)

""" TT_DEFAULT_GPU_INFER_TIMEOUT - Timeout for CPU target device"""
default_gpu_infer_timeout = get_int("TT_DEFAULT_GPU_INFER_TIMEOUT", 10*default_infer_timeout)

""" TT_DEFAULT_GPU_INFER_TIMEOUT - Timeout for CPU target device"""
default_cuda_infer_timeout = get_int("TT_DEFAULT_CUDA_INFER_TIMEOUT", 10*default_infer_timeout)

""" TT_DEFAULT_HDDL_INFER_TIMEOUT - Timeout for CPU target device"""
default_hddl_infer_timeout = get_int("TT_DEFAULT_HDDL_INFER_TIMEOUT", 3*default_infer_timeout)

""" TT_DEFAULT_MYRIAD_INFER_TIMEOUT - Timeout for CPU target device"""
default_myriad_infer_timeout = get_int("TT_DEFAULT_MYRIAD_INFER_TIMEOUT", 5*default_infer_timeout)

""" INFER TIMEOUT """
infer_timeouts = {
    TARGET_DEVICE_CPU : default_infer_timeout,
    TARGET_DEVICE_GPU : default_gpu_infer_timeout,
    TARGET_DEVICE_CUDA : default_cuda_infer_timeout,
    TARGET_DEVICE_HDDL : default_hddl_infer_timeout,
    TARGET_DEVICE_MYRIAD : default_myriad_infer_timeout,
}
infer_timeout = infer_timeouts[target_device]

""" TT_IS_NGINX_MTLS - Specify if given image is OVSA nginx mtls image. If not specified, detect from image name"""
is_nginx_mtls = get_bool("TT_IS_NGINX_MTLS", "nginx-mtls" in image)

""" TT_SKIP_TEST_IF_IS_NGINX_MTLS """
skip_nginx_test = get_bool("TT_SKIP_TEST_IF_IS_NGINX_MTLS", "True")
skip_nginx_test = skip_nginx_test and is_nginx_mtls
