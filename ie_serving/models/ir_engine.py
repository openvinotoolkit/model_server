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
from ie_serving.models.shape_management.utils import BatchingMode, ShapeMode

logger = get_logger(__name__)


class IrEngine():

    def __init__(self, model_name, model_version, net, plugin,
                 mapping_config, exec_net, inputs: dict,
                 outputs: list, batching_info, shape_info):
        self.model_name = model_name
        self.model_version = model_version
        self.exec_net = exec_net
        self.net = net
        self.batching_info = batching_info
        self.shape_info = shape_info
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
    def build(cls, model_name, model_version, model_xml, model_bin,
              mapping_config, batching_info, shape_info):
        plugin = IEPlugin(device=DEVICE, plugin_dirs=PLUGIN_DIR)
        if CPU_EXTENSION and 'CPU' in DEVICE:
            plugin.add_cpu_extension(CPU_EXTENSION)
        net = IENetwork(model=model_xml, weights=model_bin)

        if batching_info.mode == BatchingMode.FIXED:
            net.batch_size = batching_info.batch_size
        else:
            batching_info.batch_size = net.batch_size

        effective_batch_size = batching_info.get_effective_batch_size()
        logger.debug("effective batch size - {}".format(effective_batch_size))
        inputs = net.inputs
        outputs = net.outputs
        # Temporary workaround to make sure network is healthy and ready for
        # further reshaping
        if shape_info.mode != ShapeMode.DISABLED:
            net.reshape({})

        exec_net = plugin.load(network=net, num_requests=1)
        ir_engine = cls(model_name=model_name, model_version=model_version,
                        mapping_config=mapping_config, net=net, plugin=plugin,
                        exec_net=exec_net, inputs=inputs, outputs=outputs,
                        batching_info=batching_info, shape_info=shape_info)
        return ir_engine

    def _get_mapping_data_if_exists(self, mapping_config):
        if mapping_config is not None:
            try:
                with open(mapping_config, 'r') as f:
                    mapping_data = json.load(f)
                return mapping_data
            except Exception as e:
                message = "Error occurred while reading mapping_config in " \
                          "path {}. Message error {}".format(mapping_config, e)
                logger.error("[Model: {}, version: {}] --- {}".format(
                    self.model_name, self.model_version, message))
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

    def infer(self, data: dict):
        try:
            results = self.exec_net.infer(inputs=data)
        except Exception as e:
            message = "Error occurred during inference execution: {}".format(
                str(e))
            logger.debug("[Model: {}, version: {}] --- {}".format(
                self.model_name, self.model_version, message))
            return None, message
        return results, None

    def scan_input_shapes(self, data: dict):
        #   Takes dictionary of input_name:numpy_array pairs.
        changed_input_shapes = {}
        for input_name, input_data in data.items():
            net_input_shape = tuple(self.net.inputs[input_name].shape)
            if net_input_shape != input_data.shape:
                changed_input_shapes[input_name] = input_data.shape
                logger.debug("[Model: {}, version: {}] --- Shape change "
                             "required for input: {}. Current "
                             "shape: {}. Required shape: {}"
                             .format(self.model_name, self.model_version,
                                     input_name, net_input_shape,
                                     input_data.shape))
        return changed_input_shapes

    def reshape(self, reshape_param):
        if type(reshape_param) is dict:
            return self._reshape(reshape_param)
        elif type(reshape_param) is int:
            return self._change_batch_size(reshape_param)
        else:
            message = "Unknown error occurred in input reshape preparation"
            logger.debug("[Model: {}, version: {}] --- {}".format(
                self.model_name, self.model_version, message))
            return message

    def _reshape(self, inputs_shapes: dict):
        #   Takes dictionary of input_name:shape pairs as parameter
        #   (obtained from scan_input_shapes method)
        #   Returns error message on error and None if operation succeeded
        logger.debug("[Model: {}, version: {}] --- Reshaping "
                     "network...".format(self.model_name, self.model_version))
        message = None
        try:
            self.net.reshape(inputs_shapes)
        except Exception as e:
            message = "Error occurred while reshaping: {}".format(str(e))
            logger.debug("[Model: {}, version: {}] --- {}".format(
                self.model_name, self.model_version, message))
            return message
        logger.debug("[Model: {}, version: {}] --- Reshaped successfully".
                     format(self.model_name, self.model_version))

        logger.debug("[Model: {}, version: {}] --- Loading network...".
                     format(self.model_name, self.model_version))
        try:
            self.exec_net = self.plugin.load(network=self.net)
        except Exception as e:
            message = "Error occurred while loading network: {}".format(
                str(e))
            logger.debug("[Model: {}, version: {}] --- {}".format(
                self.model_name, self.model_version, message))
            return message
        logger.debug("[Model: {}, version: {}] --- Network loaded "
                     "successfully".format(self.model_name,
                                           self.model_version))
        return message

    def _change_batch_size(self, batch_size: int):
        #   Takes load batch size as a parameter. Used to change input batch
        #   size in non-reshapable models
        logger.debug("[Model: {}, version: {}] --- Changing batch size. "
                     "Loading network...".format(self.model_name,
                                                 self.model_version))
        message = None
        old_batch_size = self.net.batch_size
        self.net.batch_size = batch_size

        try:
            self.exec_net = self.plugin.load(network=self.net)
        except Exception as e:
            message = "Error occurred while loading network: {}".format(
                str(e))
            logger.debug("[Model: {}, version: {}] --- {}".format(
                self.model_name, self.model_version, message))
            self.net.batch_size = old_batch_size
            return message

        logger.debug("[Model: {}, version: {}] --- Network loaded "
                     "successfully. Batch size changed.".
                     format(self.model_name, self.model_version))
        return message
