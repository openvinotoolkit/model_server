#
# Copyright (c) 2018-2019 Intel Corporation
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

if os.path.isfile("/usr/local/lib/libcpu_extension.so"):
    default_cpu_extension = "/usr/local/lib/libcpu_extension.so"
else:
    default_cpu_extension = "/opt/intel/openvino/" \
                            "deployment_tools/inference_engine/lib/" \
                            "/intel64/libcpu_extension_avx2.so"

GLOBAL_CONFIG = {
    'cpu_extension': os.environ.get('CPU_EXTENSION', default_cpu_extension),
    'plugin_dir': os.environ.get('PLUGIN_DIR', None),
    'logging_level': os.getenv('LOG_LEVEL', 'INFO'),
    'log_path': os.getenv('LOG_PATH', None),
    'file_system_poll_wait_seconds': float(
        os.getenv('FILE_SYSTEM_POLL_WAIT_SECONDS', 1)),
    'mapping_config_filename': 'mapping_config.json',
    'rest_requests_queue_size': os.getenv('REST_REQUESTS_QUEUE_SIZE', 100),
    'engine_requests_queue_size': os.getenv('ENGINE_REQUESTS_QUEUE_SIZE',
                                            None),
    'engine_requests_queue_timeout': os.getenv(
        'ENGINE_REQUESTS_QUEUE_TIMEOUT', 5),
}

S3_CONFIG = {
    'endpoint': os.environ.get('S3_ENDPOINT', None),
    'region': os.environ.get('AWS_REGION', None),
    'signature': os.environ.get('S3_SIGNATURE', None),
    'access_key_id': os.environ.get('AWS_ACCESS_KEY_ID', 'default'),
    'secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY', 'default'),
}
