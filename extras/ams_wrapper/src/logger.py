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

import logging
import os

available_logger_levels = ['INFO', 'DEBUG', 'ERROR']


def get_logger_level():
    requested_level = os.getenv('LOG_LEVEL', 'INFO')
    requested_level = requested_level.upper()
    global LOGGER_LVL
    if requested_level in available_logger_levels:
        return requested_level
    return 'INFO'


LOGGER_LVL = get_logger_level()


def get_logger(name):
    logger = logging.getLogger(name)
    log_formatter = logging.Formatter("%(asctime)s - %(name)s - "
                                      "%(levelname)s - %(message)s")
    logger.setLevel(LOGGER_LVL)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    log_path = os.getenv('LOG_PATH', None)
    if log_path is not None:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    return logger
