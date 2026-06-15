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

from docker.errors import APIError as DockerAPIError
from docker.errors import NotFound as DockerObjectNotFound

import tests.functional.utils.assertions as assertions_module
from tests.functional.utils.assertions import DmesgError, UnexpectedResponseError
from tests.functional.utils.core import get_children_from_module
from tests.functional.utils.logger import get_logger


class Context(object):
    logger = get_logger("context")
    EXCEPTIONS_TO_CATCH = [
        UnexpectedResponseError,
        AssertionError,
        DockerObjectNotFound,
        DockerAPIError,
        PermissionError,
    ]

    def __init__(self, scope: str, name: str = None, function_item=None):
        self.name = name
        self.scope = scope
        self.cleaning = False
        self.transfers = []
        self.test_objects = []
        self.setup_process_pid = os.getpid()
        self.function_item = function_item

    def _cleanup_test_objects(self, object_list: list):
        self.logger.debug(f"Cleanup of {self.scope} context {' for ' + self.name if self.name else ''}.")
        while len(object_list) > 0:
            item = object_list.pop()
            if callable(item):
                try:
                    self.logger.info(f"calling {item!s} to /get object to/ clean.")
                    item = item()
                except BaseException as exc:
                    self.logger.exception(f"Cannot call on callable item {item!r}", exc_info=exc)
                    continue
            if item is None:
                self.logger.debug("After calling callable object result is None. "
                                  "Assuming object is already cleaned.")
            else:
                exceptions_to_catch = self.EXCEPTIONS_TO_CATCH.copy()
                dmesg_exceptions = get_children_from_module(DmesgError, assertions_module)
                dmesg_exceptions = [exc[1] for exc in dmesg_exceptions]
                exceptions_to_catch.extend(dmesg_exceptions)
                if hasattr(item, "cleanup_exceptions_to_catch"):
                    exceptions_to_catch.extend(item.cleanup_exceptions_to_catch)
                try:
                    self.logger.info(f"cleaning: {item!r}")
                    if hasattr(item, "cleanup"):
                        item.cleanup()
                    else:
                        self.logger.warning(f"Cannot call cleanup on item {item!r}")
                        continue
                except tuple(exceptions_to_catch) as e:
                    self.logger.exception(f"Error while deleting {item!r}.", exc_info=e)
                    if not hasattr(self.function_item, "runtime_bugmarks") or not self.function_item.runtime_bugmarks:
                        # raise exception only if test is not bugmarked since raising exception here will cause
                        # whole job to fail
                        raise e

    def cleanup(self):
        """Context cleanup only when cleanup process == setup process"""
        if self.setup_process_pid == os.getpid():
            self.cleaning = True
            self._cleanup_test_objects(self.test_objects)
            self.cleaning = False
