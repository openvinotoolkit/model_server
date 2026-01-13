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

from collections import defaultdict


class TargetDevice:
    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"
    AUTO = "AUTO:GPU,CPU"
    HETERO = "HETERO:GPU,CPU"
    AUTO_CPU_GPU = "AUTO:CPU,GPU"


MAX_WORKERS_PER_TARGET_DEVICE = defaultdict(
    lambda: 1,
    {  # Quite conservative for any non-listed device
        TargetDevice.CPU: 0,  # no limits !
        TargetDevice.GPU: 4,
        TargetDevice.NPU: 4,
        TargetDevice.AUTO: 4,   # keep in sync with `GPU`
        TargetDevice.HETERO: 4,  # keep in sync with `GPU`
        TargetDevice.AUTO_CPU_GPU: 4,   # keep in sync with `GPU`
    },
)
