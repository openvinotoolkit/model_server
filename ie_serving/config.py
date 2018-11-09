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
    default_cpu_extension =  "/opt/intel/computer_vision_sdk/" \
                             "deployment_tools/inference_engine/lib/" \
                             "ubuntu_16.04/intel64/libcpu_extension_avx2.so"
CPU_EXTENSION = os.environ.get('CPU_EXTENSION', default_cpu_extension)
PLUGIN_DIR = os.environ.get('PLUGIN_DIR', None)
LOGGING_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_PATH = os.getenv('LOG_PATH', None)
