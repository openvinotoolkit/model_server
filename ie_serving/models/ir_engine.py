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
import datetime
import json
import queue
from threading import Thread

from openvino.inference_engine import IENetwork, IEPlugin

from ie_serving.config import GLOBAL_CONFIG
from ie_serving.logger import get_logger
from ie_serving.models import InferenceStatus
from ie_serving.models.shape_management.batching_info import BatchingInfo
from ie_serving.models.shape_management.shape_info import ShapeInfo
from ie_serving.models.shape_management.utils import BatchingMode, ShapeMode

logger = get_logger(__name__)


def inference_callback(status, py_data):
    ir_engine = py_data['ir_engine']
    request = py_data['request']
    ireq_index = py_data['ireq_index']
    start_time = py_data['start_time']
    duration = (datetime.datetime.now() - start_time).total_seconds() * 1000

    if status == InferenceStatus.OK:
        request.set_result(ireq_index=ireq_index,
                           result=ir_engine.exec_net.requests[ireq_index].
                           outputs)
    else:
        request.set_result(ireq_index=ireq_index,
                           result="Error occurred during inference execution")

    logger.debug("[Inference callback] --- Inference completed in {} ms".
                 format(duration))


class IrEngine():

    def __init__(self, model_name, model_version, net, plugin,
                 mapping_config, exec_net, batching_info, shape_info,
                 free_ireq_index_queue, num_ireq, requests_queue,
                 target_device, plugin_config):
        self.model_name = model_name
        self.model_version = model_version
        self.exec_net = exec_net
        self.net = net
        self.batching_info = batching_info
        self.shape_info = shape_info
        self.plugin = plugin
        self.input_tensor_names = list(net.inputs.keys())
        self.output_tensor_names = list(net.outputs.keys())
        self.model_keys = self.set_keys(mapping_config)
        self.input_key_names = list(self.model_keys['inputs'].keys())

        self.free_ireq_index_queue = free_ireq_index_queue
        self.num_ireq = num_ireq
        self.requests_queue = requests_queue

        self.target_device = target_device
        self.plugin_config = plugin_config

        logger.info("Matched keys for model: {}".format(self.model_keys))

        self.engine_active = True
        self.inference_thread = Thread(target=self.start_inference_thread)
        self.inference_thread.daemon = True
        self.inference_thread.start()

    @classmethod
    def build(cls, model_name, model_version, model_xml, model_bin,
              mapping_config, batch_size_param, shape_param, num_ireq,
              target_device, plugin_config):
        plugin = IEPlugin(device=target_device,
                          plugin_dirs=GLOBAL_CONFIG['plugin_dir'])
        if GLOBAL_CONFIG['cpu_extension'] is not None \
                and 'CPU' in target_device:
            plugin.add_cpu_extension(GLOBAL_CONFIG['cpu_extension'])
        net = IENetwork(model=model_xml, weights=model_bin)
        batching_info = BatchingInfo(batch_size_param)
        shape_info = ShapeInfo(shape_param, net.inputs)
        if batching_info.mode == BatchingMode.FIXED:
            net.batch_size = batching_info.batch_size
        else:
            batching_info.batch_size = net.batch_size

        effective_batch_size = batching_info.get_effective_batch_size()
        logger.debug("[Model: {}, version: {}] --- effective batch size - {}"
                     .format(model_name, model_version, effective_batch_size))
        ###############################
        # Initial shape setup
        if shape_info.mode == ShapeMode.FIXED:
            logger.debug("[Model: {}, version: {}] --- Setting shape to "
                         "fixed value: {}".format(model_name, model_version,
                                                  shape_info.shape))
            net.reshape(shape_info.shape)
        elif shape_info.mode == ShapeMode.AUTO:
            logger.debug("[Model: {}, version: {}] --- Setting shape to "
                         "automatic".format(model_name, model_version))
            net.reshape({})
        elif shape_info.mode == ShapeMode.DEFAULT:
            logger.debug("[Model: {}, version: {}] --- Setting shape to "
                         "default".format(model_name, model_version))
        ###############################
        # Creating free infer requests indexes queue
        free_ireq_index_queue = queue.Queue(maxsize=num_ireq)
        for ireq_index in range(num_ireq):
            free_ireq_index_queue.put(ireq_index)
        ###############################
        requests_queue = queue.Queue(maxsize=GLOBAL_CONFIG[
            'engine_requests_queue_size'])

        exec_net = plugin.load(network=net, num_requests=num_ireq,
                               config=plugin_config)
        ir_engine = cls(model_name=model_name, model_version=model_version,
                        mapping_config=mapping_config, net=net, plugin=plugin,
                        exec_net=exec_net, batching_info=batching_info,
                        shape_info=shape_info,
                        free_ireq_index_queue=free_ireq_index_queue,
                        num_ireq=num_ireq, requests_queue=requests_queue,
                        target_device=target_device,
                        plugin_config=plugin_config)
        return ir_engine

    def _get_mapping_data_if_exists(self, mapping_config):
        if mapping_config is not None:
            try:
                with open(mapping_config, 'r') as f:
                    mapping_data = json.load(f)
                return mapping_data
            except Exception as e:
                message = "Error occurred while reading mapping_config in " \
                          "path {}. Message error {}".format(mapping_config,
                                                             e)
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

    def start_inference_thread(self):
        logger.debug("Starting inference service for model {} version {}"
                     .format(self.model_name, self.model_version))
        while self.engine_active:
            try:
                request = self.requests_queue.get(timeout=GLOBAL_CONFIG[
                    'engine_requests_queue_timeout'])
            except queue.Empty:
                continue
            error_message = self.adjust_network_inputs_if_needed(
                request.inference_input)
            if error_message is not None:
                request.result = error_message
                continue
            ireq_index = self.free_ireq_index_queue.get()
            py_data = {
                'ir_engine': self,
                'ireq_index': ireq_index,
                'request': request,
                'start_time': datetime.datetime.now()
            }
            self.exec_net.requests[ireq_index].set_completion_callback(
                py_callback=inference_callback, py_data=py_data)
            self.exec_net.requests[ireq_index].async_infer(
                request.inference_input)
        logger.debug("Stopping inference service for model {} version {}"
                     .format(self.model_name, self.model_version))

    def stop_inference_service(self):
        self.engine_active = False
        self.inference_thread.join()

    def suppress_inference(self):
        # Wait for all inferences executed on deleted engines to end
        logger.debug("[Model: {} version: {}] --- Waiting for in progress "
                     "inferences to finish...".
                     format(self.model_name, self.model_version))
        engine_suppressed = False
        while not engine_suppressed:
            if self.free_ireq_index_queue.full():
                engine_suppressed = True
        logger.debug("[Model: {} version: {}] --- In progress inferences "
                     "has been finalized...".
                     format(self.model_name, self.model_version))

    def adjust_network_inputs_if_needed(self, inference_input):
        error_message = None
        reshape_param = self.detect_shapes_incompatibility(
            inference_input)
        if reshape_param is not None:
            self.suppress_inference()
            error_message = self.reshape(reshape_param)
        return error_message

    def detect_shapes_incompatibility(self, inference_input):
        # Compares workload shapes with engine inputs shapes. Returns
        # reshape_param
        # reshape_param is inputs shapes dictionary (input_name:shape pairs)
        # for reshapable models and batch size for non-reshapable. If no
        # changes needed - reshape_param is None

        reshape_param = None
        inputs_shapes = self.scan_input_shapes(
            inference_input)
        if inputs_shapes:
            reshape_param = inputs_shapes
            # For non-reshapable models, batch_size of first input is the
            # reshape parameter
            if self.shape_info.mode == ShapeMode.DISABLED:
                input_shape = inputs_shapes[list(inputs_shapes.keys())[0]]
                batch_size = list(input_shape)[0]
                reshape_param = batch_size
        return reshape_param

    def scan_input_shapes(self, data: dict):
        # Takes dictionary of input_name:numpy_array pairs.
        # returns dict of input_name:shape pairs with shapes different from
        # current inputs shapes in a network - empty dict if inference
        # workload and network inputs shapes are equal.
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
        reshape_start_time = datetime.datetime.now()
        if type(reshape_param) is dict:
            error_message = self._reshape(reshape_param)
        elif type(reshape_param) is int:
            error_message = self._change_batch_size(reshape_param)
        else:
            error_message = "Unknown error occurred in input " \
                            "reshape preparation"
        reshape_end_time = datetime.datetime.now()
        if error_message is not None:
            logger.debug("[Model: {}, version: {}] --- {}".format(
                self.model_name, self.model_version, error_message))
            return error_message
        duration = \
            (reshape_end_time - reshape_start_time).total_seconds() * 1000
        logger.debug(
            "IR_ENGINE; network reshape completed; {}; {}; {}ms".format(
                self.model_name, self.model_version, duration))
        return None

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
            self.exec_net = self.plugin.load(network=self.net,
                                             num_requests=self.num_ireq,
                                             config=self.plugin_config)
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
            self.exec_net = self.plugin.load(network=self.net,
                                             num_requests=self.num_ireq,
                                             config=self.plugin_config)
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
