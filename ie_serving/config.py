#
# Copyright (c) 2018 Intel Corporation
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

DEVICE = os.environ.get('DEVICE', "CPU")
if os.path.isfile("/usr/local/lib/libcpu_extension.so"):
    default_cpu_extension = "/usr/local/lib/libcpu_extension.so"
else:
    default_cpu_extension = "/opt/intel/openvino/" \
                            "deployment_tools/inference_engine/lib/" \
                            "/intel64/libcpu_extension_avx2.so"
CPU_EXTENSION = os.environ.get('CPU_EXTENSION', default_cpu_extension)
PLUGIN_DIR = os.environ.get('PLUGIN_DIR', None)
LOGGING_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_PATH = os.getenv('LOG_PATH', None)
FILE_SYSTEM_POLL_WAIT_SECONDS = float(
    os.getenv('FILE_SYSTEM_POLL_WAIT_SECONDS', 1))

MAPPING_CONFIG_FILENAME = 'mapping_config.json'

REQUESTS_QUEUE_SIZE = os.getenv("REQUESTS_QUEUE_SIZE", 100)


class S3_Config:
    S3_ENDPOINT = os.environ.get('S3_ENDPOINT', None)
    S3_REGION = os.environ.get('AWS_REGION', None)
    S3_SIGNATURE = os.environ.get('S3_SIGNATURE', None)
    S3_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID', 'default')
    S3_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', 'default')
