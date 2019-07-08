#
# Copyright (c) 2018-2019 Intel Corporation
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

from threading import Lock
from ie_serving.config import CPU_EXTENSION, DEVICE, PLUGIN_DIR
from openvino.inference_engine import IENetwork, IEPlugin
import json
from ie_serving.logger import get_logger

logger = get_logger(__name__)


def _set_batch_size(config_batch_size, model_batch_size):
    if config_batch_size is not None:
        if config_batch_size.isdigit():
            config_batch_size = int(config_batch_size)
            if config_batch_size > 0:
                net_batch_size = config_batch_size
                engine_batch_size = config_batch_size
                effective_batch_size = str(config_batch_size)
            else:  # zero is ignored as invalid value
                effective_batch_size = str(model_batch_size)
                engine_batch_size = None
                net_batch_size = None
        elif config_batch_size == 'auto':
            engine_batch_size = 0
            net_batch_size = None
            effective_batch_size = "auto"
        else:  # invalid value in config_batch_size to be ignored
            effective_batch_size = str(model_batch_size)
            engine_batch_size = None
            net_batch_size = None
    else:  # empty config_batch_size - default
        effective_batch_size = str(model_batch_size)
        engine_batch_size = None
        net_batch_size = None
    return engine_batch_size, net_batch_size, effective_batch_size


class IrEngine():

    def __init__(self, model_xml, model_bin, net, plugin, mapping_config,
                 exec_net, inputs: dict, outputs: list, batch_size):
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.exec_net = exec_net
        self.net = net
        self.batch_size = batch_size
        self.plugin = plugin
        self.input_tensor_names = list(inputs.keys())
        self.input_tensors = inputs
        self.output_tensor_names = list(outputs.keys())
        self.output_tensors = outputs
        self.model_keys = self.set_keys(mapping_config)
        self.input_key_names = list(self.model_keys['inputs'].keys())
        self.in_use = Lock()
        logger.info("Matched keys for model: {}".format(self.model_keys))

    @classmethod
    def build(cls, model_xml, model_bin, mapping_config, batch_size):
        plugin = IEPlugin(device=DEVICE, plugin_dirs=PLUGIN_DIR)
        if CPU_EXTENSION and 'CPU' in DEVICE:
            plugin.add_cpu_extension(CPU_EXTENSION)
        net = IENetwork(model=model_xml, weights=model_bin)

        engine_batch_size, net_batch_size, effective_batch_size = \
            _set_batch_size(batch_size, net.batch_size)
        if net_batch_size is not None:
            net.batch_size = net_batch_size
        logger.debug("effective batch size - {}".format(effective_batch_size))
        inputs = net.inputs
        outputs = net.outputs
        exec_net = plugin.load(network=net, num_requests=1)
        ir_engine = cls(model_xml=model_xml, model_bin=model_bin,
                        mapping_config=mapping_config, net=net, plugin=plugin,
                        exec_net=exec_net, inputs=inputs, outputs=outputs,
                        batch_size=engine_batch_size)
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
                    temp_keys.update({
                        data[which_way][input_tensor]: input_tensor
                    })
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
        keys_names = {'inputs': self._return_proper_key_value(
            data=data, which_way='inputs', tensors=self.input_tensor_names),
            'outputs': self._return_proper_key_value(
                data=data, which_way='outputs',
                tensors=self.output_tensor_names)}
        return keys_names

    def set_keys(self, mapping_config):
        mapping_data = self._get_mapping_data_if_exists(mapping_config)
        if mapping_data is None:
            return self._set_tensor_names_as_keys()
        else:
            return self._set_names_in_config_as_keys(mapping_data)

    def infer(self, data: dict, batch_size=None):
        if batch_size is not self.net.batch_size and self.batch_size == 0:
            self.net.batch_size = batch_size
            self.exec_net = self.plugin.load(network=self.net)
        results = self.exec_net.infer(inputs=data)
        return results
