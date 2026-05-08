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
from pathlib import Path

from tests.functional.utils.process import Process


def copy_custom_lib_to_host(ovms_test_image, custom_library_path, new_library_path):
    dirpath = Path(os.path.dirname(new_library_path))
    dirpath.mkdir(parents=True, exist_ok=True)
    cmd = f"docker cp $(docker create --rm {ovms_test_image}):{custom_library_path} {new_library_path}"
    proc = Process()
    proc.run_and_check(cmd)
