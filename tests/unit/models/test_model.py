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
import threading
import time
from ie_serving.models.model import Model
from ie_serving.models.models_utils import ModelVersionState


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
            Model.get_model_version_policy_filter(model_ver_policy)
    else:
        example_array = [1, 2, 3, 4]
        output_lambda = Model.get_model_version_policy_filter(
            model_ver_policy)
        assert expected_output == output_lambda(example_array)


@pytest.mark.parametrize("new_versions, expected_to_delete, "
                         "expected_to_create",
                         [([1, 2, 3], [], []), ([1, 2], [3], []),
                          ([1, 3, 4], [2], [4])])
def test_mark_differences(get_fake_model, new_versions, expected_to_delete,
                          expected_to_create):
    model = get_fake_model
    to_create, to_delete = model._mark_differences(new_versions)

    for new_version in to_create:
        assert model.versions_statuses[new_version].state == \
               ModelVersionState.START
    for old_version in to_delete:
        assert model.versions_statuses[old_version].state == \
               ModelVersionState.UNLOADING

    assert expected_to_create == to_create
    assert expected_to_delete == to_delete


def test_delete_engine(get_fake_model):
    model = get_fake_model
    version = 2
    update_locks = {version: threading.Lock()}
    assert version in model.engines
    process_thread = threading.Thread(target=model._delete_engine,
                                      args=[version, update_locks])
    process_thread.start()
    time.sleep(7)
    assert version not in model.engines
    assert model.versions_statuses[version].state == ModelVersionState.END


@pytest.mark.parametrize("input, expected_output", [
    ('/test/test/2/', 2),
    ('/test/test/test/', -1)
])
def test_get_version_number(input, expected_output):
    output = Model.get_version_number(input)
    assert expected_output == output


def test_get_version_metadata(mocker):
    test_attributes = [{'xml_file': 'test', 'bin_file': 'test',
                        'mapping_config': 'test', 'version_number': 1,
                        'batch_size': 'test'}]
    attributes_mock = mocker.patch(
        "ie_serving.models.model.Model.get_versions_attributes")
    attributes_mock.return_value = test_attributes
    output_attributes, output_versions = Model.get_version_metadata(
        'test', None, None, lambda versions: versions[:], num_ireq=1,
        target_device='CPU', plugin_config=None)
    assert output_attributes == test_attributes
    assert output_versions == [1]
