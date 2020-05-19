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

import pytest
import os
from src.logger import get_logger, get_logger_level


def test_get_logger():
    path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "test.log")
    os.environ['LOG_PATH'] = path
    log = get_logger(__name__)
    log.info("test")

    with open(path) as f:
        read_data = f.read()

    os.remove(path)
    assert read_data.find("test\n") > 0


@pytest.mark.parametrize("set_env, value_env, expected_lvl", [
    (False, '', 'INFO'),
    (True, 'DEBUG', 'DEBUG'),
    (True, 'TEST', 'INFO'),
    (True, 'debug', 'DEBUG')])
def test_get_logger_lvl(set_env, value_env, expected_lvl):

    logger = get_logger(__name__)
    if set_env:
        os.environ['LOG_LEVEL'] = value_env
    logger_lvl = get_logger_level()
    assert logger
    assert expected_lvl == logger_lvl
