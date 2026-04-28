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

import json
from pathlib import Path

from tests.functional.utils.logger import get_logger
from tests.functional.object_model.test_environment import TestEnvironment

logger = get_logger(__name__)


class OvmsMappingConfig(object):
    FILE_NAME = "mapping_config.json"

    @staticmethod
    def generate(model, context, mapped_input_names: list = None, mapped_output_names: list = None):

        container_folder = TestEnvironment.current.prepare_container_folders(context.test_object_name, [model])[0]

        config_dict = {"inputs": {}, "outputs": {}}
        inputs_pairs = {}
        outputs_pairs = {}

        model_inputs_keys = list(model.inputs.keys())
        model_outputs_keys = list(model.outputs.keys())

        if mapped_input_names is not None and mapped_output_names is not None:
            assert len(mapped_input_names) <= len(
                model_inputs_keys
            ), f"Incorrect format of mapped input names: {mapped_input_names} - too many elements in list"
            assert len(mapped_output_names) <= len(
                model_outputs_keys
            ), f"Incorrect format of mapped output names: {mapped_output_names} - too many elements in list"

            for model_input_key, mapped_input_name in zip(model_inputs_keys, mapped_input_names):
                inputs_pair = {model_input_key: mapped_input_name}
                inputs_pairs.update(inputs_pair)

            for model_output_key, mapped_output_name in zip(model_outputs_keys, mapped_output_names):
                outputs_pair = {model_output_key: mapped_output_name}
                outputs_pairs.update(outputs_pair)
        else:
            for i, model_input_key in enumerate(model_inputs_keys):
                inputs_pair = {model_input_key: f"{model.name}_input_{i}"}
                inputs_pairs.update(inputs_pair)

            outputs_pairs = {}
            for i, model_output_key in enumerate(model_outputs_keys):
                outputs_pair = {model_output_key: f"{model.name}_output_{i}"}
                outputs_pairs.update(outputs_pair)

        config_dict["inputs"].update(inputs_pairs)
        config_dict["outputs"].update(outputs_pairs)

        mapping_dict, model = OvmsMappingConfig.prepare_model_inputs_outputs(config_dict, model)
        mapping_config_path = OvmsMappingConfig.save(mapping_dict, container_folder, model)

        return mapping_dict, mapping_config_path, container_folder

    @staticmethod
    def mapping_config_purepath(ovms_container, model):
        return Path(ovms_container, f".{model.base_path}", str(model.version), OvmsMappingConfig.FILE_NAME)

    @staticmethod
    def mapping_config_path(ovms_container, model):
        return str(OvmsMappingConfig.mapping_config_purepath(ovms_container, model))

    @staticmethod
    def mapping_exists(ovms_container, model):
        return OvmsMappingConfig.mapping_config_purepath(ovms_container, model).exists()

    @staticmethod
    def delete_mapping(ovms_container, model):
        mapping_file = OvmsMappingConfig.mapping_config_purepath(ovms_container, model)
        assert mapping_file.exists(), f"Trying to delete unexisting mapping in {mapping_file}"

        # NOTE:
        # If OVMS container is running, please reload config.json. For details please take a peek:
        # OvmsInstance.update_model_list_and_config(...)
        mapping_file.unlink()

    @staticmethod
    def save(config_dict: dict, ovms_container, model):
        file_dst_path = OvmsMappingConfig.mapping_config_path(ovms_container, model)

        logger.info("Saving config file to {}, content:\n{}".format(file_dst_path, config_dict))

        with open(file_dst_path, "w") as fp:
            json.dump(config_dict, fp, indent=2)

        return file_dst_path

    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as f:
            config_json = f.read()
            try:
                config_dict = json.loads(config_json)
            except ValueError as e:
                logger.error("Error while loading json: {}".format(config_json))
                raise e
        return config_dict

    @staticmethod
    def prepare_model_inputs_outputs(mapping_dict: dict, model):
        for config_inputs_key, config_inputs_value in mapping_dict["inputs"].items():
            model.inputs[config_inputs_value] = model.inputs.pop(config_inputs_key)

        for config_outputs_key, config_outputs_value in mapping_dict["outputs"].items():
            model.outputs[config_outputs_value] = model.outputs.pop(config_outputs_key)

        return mapping_dict, model
