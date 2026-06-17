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

import filelock


class Locker(object):
    """Manage reservation lock across multiple reservation manager processes"""
    def __init__(
            self,
            lock_path="/tmp/res_mgr.lock",
            timeout=20,
    ):

        self.lock_path = lock_path
        self.timeout = timeout
        self.lock = filelock.FileLock(lock_path)

    def acquire(self):
        """
        Acquire lock or timeout for this reservation manager process
        Unsuccessful acquiring after timeout raises exception.

        Exceptions:
          - Timeout
        """
        self.lock.acquire(timeout=self.timeout)

    def release(self):
        self.lock.release()
