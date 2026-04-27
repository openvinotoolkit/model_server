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

from tests.functional.utils.logger import get_logger

logger = get_logger(__name__)


def log_configuration_variables():
    logger.info("============== configuration variables ==============")
    pt_env_vars = list(filter(lambda x: x[0].startswith("TT_"), os.environ.items()))
    pt_env_vars.sort()
    for env_var in pt_env_vars:
        logger.info("{}={}".format(*env_var))
