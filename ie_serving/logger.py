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

import logging
from ie_serving.config import GLOBAL_CONFIG

available_lvl_of_logger = ['INFO', 'DEBUG', 'ERROR']


def get_logger_lvl():
    requested_lvl = GLOBAL_CONFIG['logging_level']
    requested_lvl = requested_lvl.upper()
    global LOGGER_LVL
    if requested_lvl in available_lvl_of_logger:
        return requested_lvl
    return 'INFO'


LOGGER_LVL = get_logger_lvl()


def get_logger(name):
    logger = logging.getLogger(name)
    log_formatter = logging.Formatter("%(asctime)s - %(name)s - "
                                      "%(levelname)s - %(message)s")
    logger.setLevel(LOGGER_LVL)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    if GLOBAL_CONFIG['log_path'] is not None:
        file_handler = logging.FileHandler(GLOBAL_CONFIG['log_path'])
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    return logger
