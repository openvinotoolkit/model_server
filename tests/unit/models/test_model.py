#
# Copyright (c) 2019 Intel Corporation
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
import json
from ie_serving.models.model import Model


@pytest.mark.parametrize("model_ver_policy, throw_error, expected_output", [
    (json.loads('{"specific": { "versions":[1,2] }}'), False, [1, 2]),
    (json.loads('{"specific": { "ver":[1,2] }}'), True, []),
    (json.loads('{"latest": { "num_versions":2 }}'), False, [3, 4]),
    (json.loads('{"latests": { "num_version":1 }}'), True, []),
    (json.loads('{"all": {}}'), False, [1, 2, 3, 4]),
    (json.loads('{"test": {}}'), True, []),
    (None, False, [4])
])
def test_get_model_policy(model_ver_policy, throw_error, expected_output):
    if throw_error:
        with pytest.raises(Exception):
            Model.get_model_version_policy(model_ver_policy)
    else:
        example_array = [1, 2, 3, 4]
        output_lambda = Model.get_model_version_policy(model_ver_policy)
        assert expected_output == output_lambda(example_array)
