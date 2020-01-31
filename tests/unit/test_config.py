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
import importlib
from ie_serving import config


def test_config_default_values():
    assert config.GLOBAL_CONFIG['cpu_extension'] == \
           config.default_cpu_extension
    assert config.GLOBAL_CONFIG['plugin_dir'] is None


def test_setting_env_variables():
    cpu_extension_test_value = "cpu_extension_test"
    plugin_dir_test_value = "plugin_dir_test"
    os.environ['CPU_EXTENSION'] = cpu_extension_test_value
    os.environ['PLUGIN_DIR'] = plugin_dir_test_value
    importlib.reload(config)
    assert config.GLOBAL_CONFIG['plugin_dir'] == plugin_dir_test_value
    assert config.GLOBAL_CONFIG['cpu_extension'] == cpu_extension_test_value
