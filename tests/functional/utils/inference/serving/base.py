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

import abc

import numpy as np


class AbstractServingWrapper(metaclass=abc.ABCMeta):
    def __init__(self, model_meta_from_serving, **kwargs):
        # During inference client init fetch model description via get_model_meta call.
        self.model_meta_from_serving = model_meta_from_serving

    @abc.abstractmethod
    def set_grpc_stubs(self):
        """
            Assigns objects for inference purposes.
        """
        pass

    @abc.abstractmethod
    def create_inference(self):
        """
            Assigns objects for inference purposes.
        """
        pass

    @abc.abstractmethod
    def predict(self, request):
        pass

    @abc.abstractmethod
    def get_rest_path(self, operation, model_version=None, model_name=None):
        """
            REST path construction is dependent from serving used: (Tensorflow / KServe)
        """
        pass

    @abc.abstractmethod
    def get_inputs_outputs_from_response(self, response):
        pass

    @abc.abstractmethod
    def get_model_meta_grpc_request(self, model_name=None):
        pass

    @abc.abstractmethod
    def get_predict_grpc_request(self):
        pass

    def get_and_validate_meta(self, model, model_name=None):
        """
            Validates and returns model metadata.
            Parameters:
                model (ModelInfo): model class object
            Returns:
                meta (ModelMetadataResponse): model metadata
        """
        meta = self.get_model_meta(version=model.version, model_name=model_name)
        self.validate_meta(model, meta)
        return meta

    def get_and_validate_metadata(self, model, model_name=None):
        return self.get_and_validate_meta(model, model_name=model_name)

    @property
    def model_name(self):
        return self.model.name

    @property
    def model_version(self):
        return None if not self.model.version else str(self.model.version)

    def set_inputs(self, inputs):
        self.model.inputs = inputs

    def set_outputs(self, outputs):
        self.model.outputs = outputs

    @property
    def inputs(self):
        return self.model.inputs

    @property
    def outputs(self):
        return self.model.outputs

    @property
    def input_names(self):
        return list(self.model.input_names or self.model.inputs.keys())

    @property
    def output_names(self):
        return list(self.model.output_names or self.model.outputs.keys())

    @property
    def input_dims(self):
        return {k: v['shape'] for k, v in self.model.inputs.items()}

    @property
    def output_dims(self):
        return {k: v['shape'] for k, v in self.model.outputs.items()}

    @property
    def input_data_types(self):
        return {k: v['dtype'] for k, v in self.model.inputs.items()}

    @property
    def output_data_types(self):
        return {k: v['dtype'] for k, v in self.model.outputs.items()}

    def create_client_data(self, inference_request):
        if inference_request is not None and inference_request.dataset:
            input_data = inference_request.load_data()
        else:
            input_data = self.model.prepare_input_data(inference_request.batch_size)
        return input_data

    def prepare_and_predict_stateful_request(self, input_objects: dict, sequence_ctrl=None,
                                             sequence_id=None, timeout=900):
        """
            Prepares and predicts stateful request.
            Parameters:
                input_objects (dict): model inputs
                sequence_ctrl (int): inference sequence control
                sequence_id (int): inference sequence id
                timeout (int): timeout
            Returns:
                result (Predict): prediction result
        """
        request = self.prepare_stateful_request(input_objects, sequence_ctrl, sequence_id)
        result = self.predict_stateful_request(request, timeout)
        return result

    def get_and_validate_status(self, model):
        """
            Validates and returns model status.
            Parameters:
                model (ModelInfo): model class object
            Returns:
                status (GetModelStatusResponse): model status

        """
        status = self.get_model_status()
        self.validate_status(model, status)
        return status

    @staticmethod
    def validate_v2_model_name_version(name, version, model):
        assert name == model.name, f"Expected {model.name} model_name; Actual: {name}"
        if version is not None:
            assert version == str(model.version), f"Expected {model.version} model_version; Actual: {version}"

    def cast_type_to_string(self, data_type):
        # https://github.com/openvinotoolkit/model_server/blob/main/src/kfs_frontend/kfs_utils.cpp
        if data_type == np.float32:
            result = 'FP32'
        elif data_type == np.int32:
            result = 'INT32'
        elif data_type == np.int64:
            result = 'INT64'
        elif data_type == str:
            result = 'BYTES'
        elif data_type == np.uint8:
            result = 'UINT8'
        elif data_type == np.object_:
            result = 'BYTES'
        else:
            raise NotImplementedError()
        return result
