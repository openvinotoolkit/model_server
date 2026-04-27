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
from tests.functional.constants.ovms import Ovms


class Valgrind:
    name = "valgrind"
    basic_params_set = f"--quiet --max-threads={Ovms.MAX_THREADS_VALGRIND}"
    full_params_set = f"--leak-check=full --show-leak-kinds=all --track-origins=yes --verbose " \
                      f"--max-threads={Ovms.MAX_THREADS_VALGRIND}"

    @classmethod
    def get_valgrind_params(cls, valgrind_mode="basic"):
        params = cls.basic_params_set if valgrind_mode == "basic" else cls.full_params_set
        return params


class Cliloader:
    # https://wiki.ith.intel.com/display/OVMS/How+to+capture+openCL+traces
    name = "cliloader"
    path = os.path.join("/opencl-intercept-layer/install/bin", name)
    env = {"CLI_CallLogging": "1", "CLI_DevicePerformanceTiming": "1"}
