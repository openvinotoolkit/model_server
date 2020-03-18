#
# Copyright (c) 2020 Intel Corporation
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
import zmq

from openvino.inference_engine import IENetwork, IECore

from ie_serving.config import GLOBAL_CONFIG
from ie_serving.engines.engine import Engine
from ie_serving.logger import get_logger
from ie_serving.models import InferenceStatus
from ie_serving.models.shape_management.batching_info import BatchingInfo
from ie_serving.models.shape_management.shape_info import ShapeInfo
from ie_serving.models.shape_management.utils import BatchingMode, ShapeMode
from ie_serving.messaging.apis.endpoint_responses_pb2 \
    import EndpointResponse, ErrorResponse
from ie_serving.server.constants import INVALID_INPUT_KEY

logger = get_logger(__name__)


class OpenvinoEngine(Engine):  # Engine class inheritance

    ###################################
    # ENGINE INTERFACE IMPLEMENTATION #
    ###################################

    def build_engine(self, engine_properties):
        self.num_ireq = engine_properties["num_ireq"]
        self.target_device = engine_properties["target_device"]
        self.plugin_config = engine_properties["plugin_config"]

        self.core = IECore()

        if GLOBAL_CONFIG['cpu_extension'] is not None \
                and 'CPU' in self.target_device:
            self.core.add_extension(extension_path=GLOBAL_CONFIG['cpu_extension'],
                               device_name='CPU')
        self.net = IENetwork(model=engine_properties["model_xml"],
                weights=engine_properties["model_bin"])
        self.input_tensor_names = list(self.net.inputs.keys())
        self.output_tensor_names = list(self.net.outputs.keys())
        self.model_keys = self._set_keys(engine_properties["mapping_config"])
        logger.info("Matched keys for model: {}".format(self.model_keys))
        self.input_key_names = list(self.model_keys["inputs"].keys())
        self.batching_info = BatchingInfo(engine_properties["batch_size_param"])
        self.shape_info = ShapeInfo(engine_properties["shape_param"], self.net.inputs)
        if self.batching_info.mode == BatchingMode.FIXED:
            self.net.batch_size = self.batching_info.batch_size
        else:
            self.batching_info.batch_size = self.net.batch_size

        self.effective_batch_size = self.batching_info.get_effective_batch_size()
        logger.debug("[Model: {}, version: {}] --- effective batch size - {}"
                     .format(self.model_name, self.model_version, self.effective_batch_size))
        ###############################
        # Initial shape setup
        if self.shape_info.mode == ShapeMode.FIXED:
            logger.debug("[Model: {}, version: {}] --- Setting shape to "
                         "fixed value: {}".format(self.model_name, self.model_version,
                                                  self.shape_info.shape))
            self.net.reshape(self.shape_info.shape)
        elif self.shape_info.mode == ShapeMode.AUTO:
            logger.debug("[Model: {}, version: {}] --- Setting shape to "
                         "automatic".format(self.model_name, self.model_version))
            self.net.reshape({})
        elif self.shape_info.mode == ShapeMode.DEFAULT:
            logger.debug("[Model: {}, version: {}] --- Setting shape to "
                         "default".format(self.model_name, self.model_version))
        ###############################
        # Creating free infer requests indexes queue
        self.free_ireq_index_queue = queue.Queue(maxsize=self.num_ireq)
        for ireq_index in range(self.num_ireq):
            self.free_ireq_index_queue.put(ireq_index)
        ###############################
        self.requests_queue = queue.Queue(maxsize=GLOBAL_CONFIG[
            'engine_requests_queue_size'])

        self.exec_net = self.core.load_network(network=self.net,
                                     device_name=self.target_device,
                                     config=self.plugin_config,
                                     num_requests=self.num_ireq)

    def predict(self, data, return_socket_name):
        inputs_in_input_request = list(dict(data).keys())

        for requested_input in inputs_in_input_request:
            if requested_input not in self.input_key_names:
                message = INVALID_INPUT_KEY % (inputs_in_input_request,
                                               self.input_key_names)
                logger.debug("PREDICT error: {}".format(message))
                zmq_return_context = zmq.Context()
                zmq_return_socket = zmq_return_context.socket(zmq.REQ)
                zmq_return_socket.connect(
                    "ipc://{}".format(return_socket_name))
                ipc_endpoint_response = EndpointResponse()
                error_resp = ErrorResponse()
                msg = "predict error - invalid key"
                error_resp.error_code = 5
                error_resp.error_message = msg
                ipc_endpoint_response.error_response.CopyFrom(error_resp)
                mesg = ipc_endpoint_response.SerializeToString()
                zmq_return_socket.send(mesg)
                zmq_return_socket.recv()
                return
        # TODO: Error handling
        self._adjust_network_inputs_if_needed(data)
        ireq_index = self.free_ireq_index_queue.get()
        py_data = {
            'ireq_index': ireq_index,
            'return_socket_name': return_socket_name,
        }
        # due to OV asynchronous mode usage, _inference_callback
        # MUST return results to the server by calling Engine method
        # return_results(...)
        self.exec_net.requests[ireq_index].set_completion_callback(
            py_callback=self._inference_callback, py_data=py_data)
        self.exec_net.requests[ireq_index].async_infer(data)

    #########################################
    # OV ENGINE BUILDING UNDERLYING METHODS #
    #########################################

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

    def _set_keys(self, mapping_config):
        mapping_data = self._get_mapping_data_if_exists(mapping_config)
        if mapping_data is None:
            return self._set_tensor_names_as_keys()
        else:
            return self._set_names_in_config_as_keys(mapping_data)

    ##############################
    # PREDICT UNDERLYING METHODS #
    ##############################

    def _inference_callback(self, status, py_data):
        ireq_index = py_data['ireq_index']
        return_socket_name = py_data['return_socket_name']

        if status == InferenceStatus.OK:
            inference_output = self.exec_net.requests[ireq_index].outputs
            # Parent class method call
            self.return_results(inference_output, return_socket_name)
            self.free_ireq_index_queue.put(ireq_index)
        else:
            # TODO: Include error handling
            pass

    def _adjust_network_inputs_if_needed(self, inference_input):
        error_message = None
        reshape_param = self._detect_shapes_incompatibility(
            inference_input)
        if reshape_param is not None:
            self._suppress_inference()
            error_message = self._reshape(reshape_param)
        return error_message

    def _detect_shapes_incompatibility(self, inference_input):
        # Compares workload shapes with engine inputs shapes. Returns
        # reshape_param
        # reshape_param is inputs shapes dictionary (input_name:shape pairs)
        # for reshapable models and batch size for non-reshapable. If no
        # changes needed - reshape_param is None

        reshape_param = None
        inputs_shapes = self._scan_input_shapes(
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

    def _scan_input_shapes(self, data: dict):
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

    def _suppress_inference(self):
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

    def _reshape(self, reshape_param):
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

    def _change_shape(self, inputs_shapes: dict):
        #   Takes dictionary of input_name:shape pairs as parameter
        #   (obtained from scan_input_shapes method)
        #   Returns error message on error and None if operation succeeded
        logger.debug("[Model: {}, version: {}] --- Changing shape. "
                     "Loading network...".format(self.model_name,
                                                 self.model_version))
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
