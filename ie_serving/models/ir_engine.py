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
import os
from urllib.parse import urlparse

from google.cloud import storage

from ie_serving.config import CPU_EXTENSION, DEVICE, PLUGIN_DIR
from openvino.inference_engine import IENetwork, IEPlugin
import glob
import json
from os.path import dirname
from ie_serving.logger import get_logger

logger = get_logger(__name__)


class IrEngine():

    def __init__(self, model_xml, model_bin, exec_net, inputs: dict,
                 outputs: list):
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.exec_net = exec_net
        self.input_tensor_names = list(inputs.keys())
        self.input_tensors = inputs
        self.output_tensor_names = outputs
        self.model_keys = self.set_keys()
        self.input_key_names = list(self.model_keys['inputs'].keys())
        logger.info("Matched keys for model: {}".format(self.model_keys))

    @classmethod
    def build(cls, model_xml, model_bin):
        plugin = IEPlugin(device=DEVICE, plugin_dirs=PLUGIN_DIR)
        if CPU_EXTENSION and 'CPU' in DEVICE:
            plugin.add_cpu_extension(CPU_EXTENSION)
        local_model_xml, local_model_bin = cls.get_local_paths(
            model_xml, model_bin)
        net = IENetwork.from_ir(model=local_model_xml, weights=local_model_bin)
        if local_model_xml != model_xml and local_model_bin != model_bin:
            cls.delete_tmp_files([local_model_xml, local_model_bin])
        inputs = net.inputs
        batch_size = list(inputs.values())[0][0]
        outputs = net.outputs
        exec_net = plugin.load(network=net, num_requests=batch_size)
        ir_engine = cls(model_xml=model_xml, model_bin=model_bin,
                        exec_net=exec_net, inputs=inputs, outputs=outputs)
        return ir_engine

    @classmethod
    def get_local_paths(cls, model_xml, model_bin):
        parsed_model_xml = urlparse(model_xml)
        parsed_model_bin = urlparse(model_bin)
        if parsed_model_xml.scheme == '' and parsed_model_bin.scheme == '':
            return model_xml, model_bin
        elif parsed_model_xml.scheme == 'gs' and \
                parsed_model_bin.scheme == 'gs':
            local_model_xml = cls.gs_download_file(model_xml)
            local_model_bin = cls.gs_download_file(model_bin)
            return local_model_xml, local_model_bin
        elif parsed_model_xml.scheme == 's3' and \
                parsed_model_bin.scheme == 's3':
            pass
        return None, None

    @classmethod
    def gs_download_file(cls, path):
        parsed_path = urlparse(path)
        bucket_name = parsed_path.netloc
        file_path = parsed_path.path[1:]
        gs_client = storage.Client()
        bucket = gs_client.get_bucket(bucket_name)
        blob = bucket.blob(file_path)
        tmp_path = os.path.join('/tmp', file_path.split(os.sep)[-1])
        blob.download_to_filename(tmp_path)
        return tmp_path

    @classmethod
    def delete_tmp_files(cls, files_paths):
        for file_path in files_paths:
            os.remove(file_path)

    def _get_mapping_config_file_if_exists(self):
        parent_dir = dirname(self.model_bin)
        config_path = glob.glob("{}/mapping_config.json".format(parent_dir))
        if len(config_path) == 1:
            try:
                with open(config_path[0], 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.error("Error occurred while reading mapping_config in "
                             "path {}. Message error {}"
                             .format(config_path, e))
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
                      _return_proper_key_value(data=data, which_way='outputs',
                                               tensors=self.
                                               output_tensor_names)}
        return keys_names

    def set_keys(self):
        config_file = self._get_mapping_config_file_if_exists()
        if config_file is None:
            return self._set_tensor_names_as_keys()
        else:
            return self._set_names_in_config_as_keys(config_file)

    def infer(self, data: dict):
        results = self.exec_net.infer(inputs=data)
        return results
