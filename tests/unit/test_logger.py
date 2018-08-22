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
import importlib
import pytest
from ie_serving import config
from ie_serving import logger


@pytest.mark.parametrize("set_env, value_env, expected_lvl", [
    (False, '', 'INFO'),
    (True, 'DEBUG', 'DEBUG'),
    (True, 'TEST', 'INFO'),
    (True, 'debug', 'DEBUG')])
def test_get_logger_lvl(set_env, value_env, expected_lvl):
    if set_env:
        os.environ['LOG_LEVEL'] = value_env
        importlib.reload(config)
        importlib.reload(logger)
    logger_lvl = logger.get_logger_lvl()
    assert expected_lvl == logger_lvl
