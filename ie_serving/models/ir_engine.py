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

from ie_serving.config import CPU_EXTENSION, DEVICE, PLUGIN_DIR
from openvino.inference_engine import IENetwork, IEPlugin
import json
from ie_serving.logger import get_logger

logger = get_logger(__name__)


class IrEngine():

    def __init__(self, model_xml, model_bin, mapping_config, exec_net,
                 inputs: dict,
                 outputs: list):
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.exec_net = exec_net
        self.input_tensor_names = list(inputs.keys())
        self.input_tensors = inputs
        self.output_tensor_names = outputs
        self.model_keys = self.set_keys(mapping_config)
        self.input_key_names = list(self.model_keys['inputs'].keys())
        logger.info("Matched keys for model: {}".format(self.model_keys))

    @classmethod
    def build(cls, model_xml, model_bin, mapping_config):
        plugin = IEPlugin(device=DEVICE, plugin_dirs=PLUGIN_DIR)
        if CPU_EXTENSION and 'CPU' in DEVICE:
            plugin.add_cpu_extension(CPU_EXTENSION)
        net = IENetwork.from_ir(model=model_xml, weights=model_bin)
        inputs = net.inputs
        batch_size = list(inputs.values())[0][0]
        outputs = net.outputs
        exec_net = plugin.load(network=net, num_requests=batch_size)
        ir_engine = cls(model_xml=model_xml, model_bin=model_bin,
                        mapping_config=mapping_config,
                        exec_net=exec_net, inputs=inputs, outputs=outputs)
        return ir_engine

    def _get_mapping_data_if_exists(self, mapping_config):
        if mapping_config is not None:
            try:
                with open(mapping_config, 'r') as f:
                    mapping_data = json.load(f)
                return mapping_data
            except Exception as e:
                logger.error("Error occurred while reading mapping_config "
                             "in path {}. Message error {}"
                             .format(mapping_config, e))
        return None

    def _return_proper_key_value(self, data: dict, which_way: str,
                                 tensors: list):
        temp_keys = {}
        for input_tensor in tensors:
            if which_way in data:
                if input_tensor in data[which_way]:
                    temp_keys.update({data[which_way][input_tensor]:
                                          input_tensor})
                else:
                    temp_keys.update({input_tensor: input_tensor})
            else:
                temp_keys.update({input_tensor: input_tensor})
        return temp_keys

    def _set_tensor_names_as_keys(self):
        keys_names = {'inputs': {}, 'outputs': {}}
        for input_tensor in self.input_tensor_names:
            keys_names['inputs'].update({input_tensor: input_tensor})
        for output_tensor in self.output_tensor_names:
            keys_names['outputs'].update({output_tensor: output_tensor})
        return keys_names

    def _set_names_in_config_as_keys(self, data: dict):
        keys_names = {'inputs': self.
            _return_proper_key_value(data=data, which_way='inputs',
                                     tensors=self.
                                     input_tensor_names),
                      'outputs': self.
                          _return_proper_key_value(data=data,
                                                   which_way='outputs',
                                                   tensors=self.
                                                   output_tensor_names)}
        return keys_names

    def set_keys(self, mapping_config):
        mapping_data = self._get_mapping_data_if_exists(mapping_config)
        if mapping_data is None:
            return self._set_tensor_names_as_keys()
        else:
            return self._set_names_in_config_as_keys(mapping_data)

    def infer(self, data: dict):
        results = self.exec_net.infer(inputs=data)
        return results
