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

import base64
import json
import os
import struct
import time
import cohere
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import reduce
from inspect import isclass
from pathlib import Path
from threading import Event
from typing import List, Union

import grpc
import numpy as np
import requests
import tritonclient
from google.protobuf.json_format import MessageToJson
from grpc import RpcError
from grpc._channel import _InactiveRpcError
from openai import OpenAI
from pydantic import BaseModel
from retry.api import retry_call
from tensorflow import make_tensor_proto
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.grpc.service_pb2 import ModelInferRequest
from tritonclient.utils import InferenceServerException, deserialize_bytes_tensor, serialize_byte_tensor

from tests.functional.utils.assertions import ModelNotReadyException, StreamingApiException, UnexpectedResponseError
from tests.functional.utils.inference.communication.grpc import GRPC, GrpcCommunicationInterface, channel_options
from tests.functional.utils.inference.communication.rest import REST, RestCommunicationInterface
from tests.functional.utils.inference.serving.cohere import RerankApi, CohereRequestParams, CohereWrapper
from tests.functional.utils.inference.serving.kf import KFS, KserveWrapper
from tests.functional.utils.inference.serving.openai import (
    ChatCompletionsApi,
    CompletionsApi,
    EmbeddingsApi,
    OpenAIRequestParams,
    OpenAIWrapper,
    ImagesApi,
    AudioApi,
    ResponsesApi,
)
from tests.functional.utils.inference.serving.tf import TensorFlowServingWrapper
from tests.functional.utils.logger import get_logger
from tests.functional.utils.test_framework import FrameworkMessages, skip_if_runtime
from llm.validation_utils import LLMValidationUtils
from ovms import config
from ovms.config import binary_io_images_path
from ovms.constants.model_dataset import (
    BinaryDummyModelDataset,
    DefaultBinaryDataset,
    ExactShapeBinaryDataset,
    LanguageModelDataset,
    ModelDataset,
)
from tests.functional.constants.models import ModelInfo
from tests.functional.constants.ovms import CurrentTarget as ct
from tests.functional.constants.ovms import MediaPipeConstants, Ovms
from tests.functional.constants.pipelines import SimpleMediaPipe
from tests.functional.object_model.ovms_instance import OvmsInstance
from ovms.object_model.ovsa import OvsaCerts
from ovms.object_model.python_custom_nodes.common import STREAMING_CHANNEL_ARGS
from ovms.object_model.python_custom_nodes.python_custom_nodes import SimplePythonCustomNodeMediaPipe
from tests.functional.object_model.test_environment import TestEnvironment
from tests.functional.object_model.test_helpers import run_all_actions

logger = get_logger(__name__)


class InferenceBuilder(object):

    def __init__(self, model):
        self.model = model

    def create_client(
        self, api_type, port, batch_size=Ovms.BATCHSIZE, ovsa_certs=None, model_version=None, client_type=None
    ):
        ovsa_certs = ovsa_certs if ovsa_certs is not None else OvsaCerts.default_certs
        if client_type == KFS:
            assert 0, "Please check flow for KFS client"
            kfs_api_type = InferenceClientKFS if api_type == InferenceClientTFS else InferenceRestClientKFS
            inference_client = self.create_kfs_client(kfs_api_type, port)
        else:
            inference_client = api_type(
                port=port,
                model_name=self.model.name,
                batch_size=batch_size,
                input_names=list(self.model.inputs.keys()),
                output_names=list(self.model.outputs.keys()),
                model_meta_from_serving=False,
                ssl_certificates=ovsa_certs,
                model_version=model_version,
            )
        inference_client._model = self.model
        return inference_client

    def create_client_and_data(self, inference_request, random_data=False):
        if inference_request.client_type == KFS:
            # This should be included into KserveWrapper, please correct calling test not to use `create_client_and_data`
            assert False, "Please correct it"
            kfs_api_type = (
                InferenceClientKFS if inference_request.api_type == InferenceClientTFS else InferenceRestClientKFS
            )
            port = inference_request.get_port()
            inference_client = self.create_kfs_client(kfs_api_type, port)
        else:
            inference_client = self.create_client(
                inference_request.api_type,
                inference_request.get_port(),
                inference_request.batch_size,
                client_type=inference_request.client_type,
                model_version=inference_request.model_version,
            )
        if inference_request is not None and inference_request.dataset:
            input_data = inference_request.load_data()
        else:
            input_data = self.model.prepare_input_data(inference_request.batch_size, random_data=random_data)
        return inference_client, input_data

    def create_kfs_client(self, api_type, port):
        kfs_api_client = api_type(port, model_name=self.model.name, batch_size=self.model.batch_size)
        kfs_api_client.model = self.model
        kfs_api_client.port = port
        kfs_api_client.model_name = self.model.name
        return kfs_api_client


@dataclass(frozen=False)
class InferenceRequest(object):
    ovms: OvmsInstance = None
    model: ModelInfo = None
    api_type: object = None
    batch_size: int = None
    dataset: ModelDataset = None
    client_type: str = None
    model_version: str = None

    def get_port(self):
        return self.ovms.ovms_ports[self.api_type.type]

    def prepare_request_to_send(self, client, input_data, mediapipe_name=None):
        request = client.prepare_request(input_objects=input_data, mediapipe_name=mediapipe_name)
        logger.debug(f"Request: {request}")
        return request

    def validate(self, response):
        pass

    def get_expected_output_shape(self):
        expected_shape = {}
        for output_name, output_data in self.model.outputs.items():
            expected_shape[output_name] = deepcopy(output_data["shape"])
            if self.batch_size:
                expected_shape[output_name][0] = self.batch_size
            if self.model.get_demultiply_count() is not None:
                expected_shape[output_name].insert(0, self.model.default_demultiply_count_value)

        return expected_shape


@dataclass(frozen=False)
class BinaryInferenceRequest(InferenceRequest):
    layout: str = Ovms.BINARY_IO_LAYOUT_ROW_NAME
    dataset: ModelDataset = field(default_factory=lambda: DefaultBinaryDataset())
    format: str = None
    validate_match: bool = True
    batch_size: int = 1

    def _create_post_request(self, input_names, image_data, request_format=Ovms.BINARY_IO_LAYOUT_ROW_NAME):
        signature = "serving_default"
        instances = []
        for input_name in input_names:
            data = image_data[input_name]
            for single_data in data:
                image_bytes_encoded = base64.b64encode(single_data).decode("utf-8")
                if request_format == Ovms.BINARY_IO_LAYOUT_ROW_NAME:
                    instances.append({input_name: {"b64": image_bytes_encoded}})
                else:
                    instances.append({"b64": image_bytes_encoded})

        if request_format == Ovms.BINARY_IO_LAYOUT_ROW_NAME:
            data_obj = {"signature_name": signature, "instances": instances}
        elif request_format == Ovms.BINARY_IO_LAYOUT_ROW_NONAME:
            data_obj = {"signature_name": signature, "instances": instances}
        elif request_format == Ovms.BINARY_IO_LAYOUT_COLUMN_NAME:
            data_obj = {"signature_name": signature, "inputs": {input_name: instances}}
        elif request_format == Ovms.BINARY_IO_LAYOUT_COLUMN_NONAME:
            data_obj = {"signature_name": signature, "inputs": instances}
        else:
            print("invalid request format defined")
            exit(1)
        data_json = json.dumps(data_obj)
        return {"request": data_json}

    def prepare_request_to_send(self, client, input_data):
        if isinstance(client, KserveWrapper) and isinstance(client, GrpcCommunicationInterface):
            request = ModelInferRequest()
            request.model_name = self.model.name
            inputs, outputs = self.prepare_binary_inputs_outputs(input_data)
            request.inputs.extend(inputs)
            request.outputs.extend(outputs)
            request = {"request": request}
        elif isinstance(client, KserveWrapper) and isinstance(client, RestCommunicationInterface):
            request = self._create_kfs_post_request(input_data)
        elif isinstance(client, TensorFlowServingWrapper) and isinstance(client, RestCommunicationInterface):
            request = self._create_post_request(self.model.input_names, input_data, request_format=self.layout)
        elif isinstance(client, TensorFlowServingWrapper) and isinstance(client, GrpcCommunicationInterface):
            request = PredictRequest()
            request.model_spec.name = self.model.name
            for input_name, input_object in input_data.items():
                request.inputs[input_name].CopyFrom(make_tensor_proto(input_object, shape=[len(input_object)]))
            request = {"request": request}
        else:
            raise NotImplementedError
        return request

    def _create_kfs_post_request(self, input_data):
        batch_i = 0
        image_binary_size = []
        shape = []
        inputs = []
        input_object = None

        for input_name, input_object in input_data.items():
            for obj in input_object:
                if batch_i < len(input_object):
                    image_binary_size.append(len(obj))
                    batch_i += 1

            shape.extend([len(input_object)])
            _summarized_size = 4 * len(image_binary_size) + reduce(lambda x, y: x + y, image_binary_size)

            _input = {
                "name": input_name,
                "shape": shape,
                "datatype": "BYTES",
                "parameters": {"binary_data_size": _summarized_size},
            }
            inputs.append(_input)

        request_header = json.dumps({"inputs": inputs}, separators=(",", ":"))

        # https://wiki.ith.intel.com/display/OVMS/Changes+in+KFS+BYTES+format
        # https://github.com/openvinotoolkit/model_server/blob/main/docs/binary_input_kfs.md
        if input_object is not None:
            _ovms_formatted_binary_objects = b"".join([len(x).to_bytes(4, "little") + x for x in input_object])
            binary_data = _ovms_formatted_binary_objects
        else:
            binary_data = b""

        request_body = struct.pack(
            "{}s{}s".format(len(request_header), len(binary_data)), request_header.encode(), binary_data
        )
        return {
            "request": request_body,
            "inference_header": {"Inference-Header-Content-Length": str(len(request_header))},
        }

    def load_data(self):
        result = dict()
        for param_name, param_data in self.model.inputs.items():
            result[param_name] = self.dataset.get_data(
                param_data["shape"], self.batch_size, self.model.transpose_axes, None
            )
        return result

    def validate(self, response):
        if self.validate_match:
            assert self.dataset.verify_match(response)

    def prepare_binary_inputs_outputs(self, input_data):
        dtype = "BYTES"
        inputs = []
        outputs = []

        for input_name, input_object in input_data.items():
            input = service_pb2.ModelInferRequest().InferInputTensor()
            input.name = input_name
            input.datatype = dtype
            input.shape.extend([len(input_object)])
            batch_i = 0
            while batch_i < len(input_object):
                input.contents.bytes_contents.append(input_object[batch_i])
                batch_i += 1
            inputs.append(input)

        for output_name in self.model.outputs:
            output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            output.name = output_name
            outputs.append(output)

        return inputs, outputs


@dataclass(frozen=False)
class BinaryInferenceStaticShapeRequest(BinaryInferenceRequest):
    layout: str = Ovms.BINARY_IO_LAYOUT_ROW_NAME
    dataset: ModelDataset = None
    format: str = None
    validate_match: bool = True
    batch_size: int = 1

    def __post_init__(self):
        self.dataset = ExactShapeBinaryDataset(shape=[224, 224, 3])

    def validate(self, response):
        return True


@dataclass(frozen=False)
class BinaryInferenceHeteroShapeRequest(BinaryInferenceRequest):
    batch_config: list = field(default_factory=lambda: [])
    tmp_data_file_location: str = None
    format: str = Ovms.PNG_IMAGE_FORMAT

    def __post_init__(self):
        self.batch_size = len(self.batch_config)
        for _, in_data in self.model.inputs.items():
            in_data["dataset"] = BinaryDummyModelDataset()

    def load_data(self):
        result = defaultdict(lambda: [])
        tmp_binary_dir = Path(os.path.join(self.tmp_data_file_location, Path(binary_io_images_path).name))
        for param_name, param_data in self.model.inputs.items():
            for shape in self.batch_config:
                result[param_name].append(param_data["dataset"].create_data(tmp_binary_dir, shape, self.format))
        return result

    def validate(self, response):
        return True


@dataclass(frozen=False)
class LLMInferenceRequest(InferenceRequest):
    request_parameters: OpenAIRequestParams = None
    set_null_values: bool = False
    use_extra_body: bool = True

    def __post_init__(self):
        self.api_type.create_base_url()
        self.openai_client = OpenAI(base_url=self.api_type.base_url, api_key=self.api_type.API_KEY_UNUSED)
        self.request_parameters_dict = self.prepare_request_parameters_dict(
            set_null_values=self.set_null_values,
            use_extra_body=self.use_extra_body,
        )
        if hasattr(self.request_parameters, "stream"):
            self.stream = self.request_parameters.stream if self.request_parameters.stream is not None else False

    def prepare_request_parameters_dict(self, set_null_values=False, use_extra_body=True):
        if self.request_parameters:
            assert isinstance(self.request_parameters, (OpenAIRequestParams, CohereRequestParams)), (
                f"Wrong type of request_parameters expected: {OpenAIRequestParams} or {CohereRequestParams}"
                f"actual: {type(self.request_parameters)}"
            )
            return self.request_parameters.prepare_dict(
                set_null_values=set_null_values,
                use_extra_body=use_extra_body,
            )
        else:
            return {}

    def create_chat_completions(self, messages, model_name=None, timeout=None):
        model = model_name if model_name is not None else self.api_type.model.name
        response_format = self.request_parameters_dict.get("response_format", None)
        if response_format is not None and isclass(response_format) and issubclass(response_format, BaseModel):
            chat_completions = self.openai_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                timeout=timeout,
                **self.request_parameters_dict,
            )
        else:
            chat_completions = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=timeout,
                **self.request_parameters_dict,
            )
        outputs = [chat_completions] if not self.stream else chat_completions
        return outputs

    def create_completions(self, prompt, model_name=None, timeout=None):
        model = model_name if model_name is not None else self.api_type.model.name
        completions = self.openai_client.completions.create(
            model=model,
            prompt=prompt,
            **self.request_parameters_dict,
            timeout=timeout,
        )
        outputs = [completions] if not self.stream else completions
        return outputs

    def create_responses(self, input_content, model_name=None, timeout=None):
        model = model_name if model_name is not None else self.api_type.model.name
        response = self.openai_client.responses.create(
            model=model,
            input=input_content,
            **self.request_parameters_dict,
            timeout=timeout,
        )
        outputs = [response] if not self.stream else response
        return outputs

    def create_embeddings(self, embeddings_input, model_name=None, timeout=None):
        model = model_name if model_name is not None else self.api_type.model.name
        embeddings = self.openai_client.embeddings.create(
            model=model,
            input=embeddings_input,
            **self.request_parameters_dict,
            timeout=timeout,
        )
        return embeddings

    def create_image_generation(self, prompt, model_name=None, timeout=None):
        model = model_name if model_name is not None else self.api_type.model.name
        image = self.openai_client.images.generate(
            model=model,
            prompt=prompt,
            **self.request_parameters_dict,
            timeout=timeout,
        )
        return image

    def create_image_edit(self, prompt, image_path=None, mask_path=None, model_name=None, timeout=None):
        model = model_name if model_name is not None else self.api_type.model.name
        if mask_path is not None:
            self.request_parameters_dict["mask"] = open(mask_path, "rb")
        image = self.openai_client.images.edit(
            model=model,
            prompt=prompt,
            image=open(image_path, "rb"),
            **self.request_parameters_dict,
            timeout=timeout,
        )
        return image

    def create_models_list(self):
        models_list = self.openai_client.models.list()
        return models_list

    def create_models_retrieve(self, model_name=None):
        model = model_name if model_name is not None else self.api_type.model.name
        retrieve = self.openai_client.models.retrieve(model=model)
        return retrieve

    def create_audio_speech(self, input_text, speech_file_path, model_name=None, timeout=None):
        model = model_name if model_name is not None else self.api_type.model.name
        voice = self.request_parameters_dict.pop("voice", None)
        with self.openai_client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,  # voice is a required parameter in OpenAI API; OVMS accepts None for default
            input=input_text,
            **self.request_parameters_dict,
            timeout=timeout,
        ) as response:
            response.stream_to_file(speech_file_path)
        return response

    def create_audio_transcription(self, audio_file_path, model_name=None, timeout=None):
        model = model_name if model_name is not None else self.api_type.model.name
        with open(audio_file_path, "rb") as audio_file:
            transcript = self.openai_client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                **self.request_parameters_dict,
                timeout=timeout,
            )
        return transcript.text

    def create_audio_translation(self, audio_file_path, model_name=None, timeout=None):
        model = model_name if model_name is not None else self.api_type.model.name
        with open(audio_file_path, "rb") as audio_file:
            translation = self.openai_client.audio.translations.create(
                model=model,
                file=audio_file,
                **self.request_parameters_dict,
                timeout=timeout,
            )
        return translation.text


@dataclass(frozen=False)
class RerankLLMInferenceRequest(LLMInferenceRequest):
    request_parameters: OpenAIRequestParams = None

    def __post_init__(self):
        self.api_type.create_base_url()
        self.cohere_client = cohere.Client(base_url=self.api_type.base_url, api_key=self.api_type.API_KEY_NOT_USED)
        self.request_parameters_dict = self.prepare_request_parameters_dict()

    def create_rerank(self, rerank_input, model_name=None):
        model = model_name if model_name is not None else self.api_type.model.name
        rerank = self.cohere_client.rerank(
            model=model,
            query=rerank_input["query"],
            documents=rerank_input["documents"],
            **self.request_parameters_dict,
        )
        return rerank


class InferenceResponse(object):

    def __init__(self, inference_info, response):
        self.inference_info = inference_info
        self.response = response

    @classmethod
    def create(cls, inference_info, response):
        return InferenceResponse(inference_info, response)

    def ensure_outputs_exist(self):
        for output_name in self.inference_info.model.outputs:
            assert output_name in self.response, "Incorrect output name, expected: {}, found: {}.".format(
                output_name, ", ".join(self.response.keys())
            )

    def validate(self, input_data):
        self.ensure_outputs_exist()
        expected_output = self.inference_info.model.get_expected_output(input_data)
        if expected_output is not None:
            self.validate_expected_output(expected_output, self.response)

        self.validate_expected_shape(self.response)
        self.inference_info.inference_request.validate(self.response)

    def validate_expected_output(self, expected_output, response, equal=False):
        for key in expected_output.keys():
            if equal:
                result = np.array_equal(expected_output[key], self.response[key])
            else:
                # Note: rtol=1.e-3 was changed from default value (rtol=1.e-5) after a bug was found: CVS-107839
                rtol = (
                    1.0e-3
                    if any([ct.is_gpu_target(), ct.is_auto_target(), ct.is_hetero_target()])
                    else 1.0e-5
                )
                result = np.allclose(expected_output[key], response[key], rtol=rtol)
            if not result:
                logger.info(f"Received output for key '{key}': {response[key]}")
                logger.info(f"Expected output for key '{key}': {expected_output[key]}")

            assert result, "Received output is different than expected"

    def validate_expected_shape(self, response):
        output_shape = {name: list(data.shape) for name, data in response.items()}
        expected_shape = self.inference_info.inference_request.get_expected_output_shape()

        validation_pass = True
        for name, expected_data in expected_shape.items():
            for dim, expected_dim_value in enumerate(expected_data):
                if expected_dim_value > 0:
                    validation_pass = expected_dim_value == output_shape[name][dim]

        assert validation_pass, "Incorrect output shape, expected: {}, found: {}.".format(expected_shape, output_shape)
        logger.debug(f"Output shape: {output_shape} (expected: {expected_shape})")


class MediaPipeInferenceResponse(InferenceResponse):

    def __init__(self, inference_info, response):
        super().__init__(inference_info, response)

    @classmethod
    def create(cls, inference_info, response):
        return MediaPipeInferenceResponse(inference_info, response)

    def ensure_outputs_exist(self, output_key=None):
        for elem in self.response:
            if output_key is not None:
                elem == output_key
            else:
                assert MediaPipeConstants.DEFAULT_OUTPUT_STREAM in elem

    def validate(self, input_data, output_key=None):
        logger.debug(f"MediaPipeInferenceResponse: {self.response}")
        self.ensure_outputs_exist(output_key)
        expected_output = self.inference_info.model.get_expected_output(input_data, self.inference_info.client.type)
        if expected_output is not None:
            if isinstance(self.inference_info.model, SimplePythonCustomNodeMediaPipe):
                self.validate_expected_output(expected_output, self.response, equal=True)
            else:
                self.validate_expected_output(expected_output, self.response)
                self.validate_expected_shape(self.response, output_key=output_key)
                self.inference_info.inference_request.validate(self.response)

    def validate_expected_shape(self, response, output_key=None):
        output_shape = {name: list(data.shape) for name, data in response.items()}
        expected_shape = self.inference_info.inference_request.get_expected_output_shape()

        validation_pass = True
        for name, expected_data in expected_shape.items():
            for dim, expected_dim_value in enumerate(expected_data):
                if expected_dim_value > 0:
                    if output_key is not None:
                        output_shape_key = output_key
                    else:
                        expected_shape_idx = list(expected_shape.keys()).index(name) + 1
                        output_shape_key = f"out_{expected_shape_idx}"
                    validation_pass = expected_dim_value == output_shape[output_shape_key][dim]

        assert validation_pass, "Incorrect output shape, expected: {}, found: {}.".format(expected_shape, output_shape)
        logger.debug(f"Output shape: {output_shape} (expected: {expected_shape})")


class LLMInferenceResponse(InferenceResponse):

    def __init__(self, inference_info, response):
        super().__init__(inference_info, response)

    @classmethod
    def create(cls, inference_info, response):
        return LLMInferenceResponse(inference_info, response)

    def validate(self):
        logger.info(self.response)
        for choice in self.response["choices"]:
            assert len(choice["message"]["content"]) > 1, f"Empty message: {choice}"
        assert self.response["model"] == self.inference_info.model.name, f"Invalid model name: {self.response['model']}"


class InferenceInfo(object):

    @classmethod
    def create(cls, client, model, timeout=config.wait_for_messages_timeout, input_data=None, inference_request=None):
        if model.is_stateful:
            return StatefulInferenceInfo(client, model, timeout, input_data, inference_request)
        else:
            return InferenceInfo(client, model, timeout, input_data, inference_request)

    def __init__(
        self, client, model, timeout=config.wait_for_messages_timeout, input_data=None, inference_request=None
    ):
        self.client = client
        self.input_data = input_data
        self.model = model
        self.timeout = timeout
        self.inference_request = inference_request

    def clear_input_data(self):
        for name in self.model.input_names:
            data = self.input_data[name]
            self.input_data[name] = np.zeros(data.shape, dtype=data.dtype)

    def predict(self):
        request = self.inference_request.prepare_request_to_send(self.client, self.input_data)
        result = self.client.predict(request, self.timeout)
        return result

    def predict_sequence_step(self, tensor_data, sequence_ctrl, sequence_id):
        request = self.client.prepare_stateful_request(tensor_data, sequence_ctrl, sequence_id)
        return self.client.predict_stateful_request(request, self.timeout)

    def get_metadata(self):
        meta = self.client.get_model_meta()
        json_data = json.loads(MessageToJson(meta))
        return json_data

    def get_and_validate_metadata(self, expected_shape_dict):
        metadata = self.get_metadata()
        assert self.model.name == metadata["modelSpec"]["name"]
        assert self.model.version == int(metadata["modelSpec"]["version"])

        metadata_inputs = metadata["metadata"]["signature_def"]["signatureDef"]["serving_default"]["inputs"]
        for in_name in self.model.inputs:
            assert in_name in metadata_inputs

            for idx, expected_dim_value in enumerate(expected_shape_dict[in_name]):
                dim_value = metadata_inputs[in_name]["tensorShape"]["dim"][idx]["size"]
                if isinstance(expected_dim_value, str):
                    assert expected_dim_value == dim_value
                else:
                    assert expected_dim_value == int(dim_value)


class StatefulInferenceInfo(InferenceInfo):
    SEQUENCE_START = 1
    SEQUENCE_END = 2

    def __init__(self, client, model, timeout=30, input_data=None, inference_request=None):
        super().__init__(client, model, timeout, input_data, inference_request)

    def _get_sequence_control_data(self, data_length, iteration_index):
        result = None
        if iteration_index == 0:
            result = sequence_ctrl = StatefulInferenceInfo.SEQUENCE_START
        elif iteration_index == data_length + self.model.context_window_left + self.model.context_window_right - 1:
            result = sequence_ctrl = StatefulInferenceInfo.SEQUENCE_END
        return result

    def get_utterance_name_list(self):
        input_param_name = self.model.input_names[0]
        return list(self.input_data[input_param_name].keys())

    def predict(self, sequence_id=None):
        result = {}
        for utterance in self.get_utterance_name_list():
            result[utterance] = self.predict_utterance(utterance, sequence_id)
        return result

    def predict_utterance(self, utterance_name, sequence_id=None):
        logger.info(f"Model ({self.model.name}) predict [{utterance_name}]")
        result = []
        utterance_length = self.get_utterance_length(utterance_name)
        offset = self.model.context_window_left + self.model.context_window_right

        for idx in range(utterance_length):
            sequence_ctrl = self._get_sequence_control_data(utterance_length, idx)
            tensor_data = self.get_utterance_data(utterance_name, idx)
            sequence_id, output = self.predict_sequence_step(tensor_data, sequence_ctrl, sequence_id)
            if idx >= offset:
                result.append(output)  # collect data for idx: <offset; utterance_length)
        return result

    def get_utterance_length(self, utterance_name):
        input_param_name = self.model.input_names[0]
        offset = self.model.context_window_left + self.model.context_window_right
        return offset + len(self.input_data[input_param_name][utterance_name])

    def get_utterance_data(self, utterance_name, idx):
        input_param_name = self.model.input_names[0]
        data_length = len(self.input_data[input_param_name][utterance_name])
        data_idx = idx - self.model.context_window_left
        if data_idx < 0:
            data_idx = 0  # fill first data with tensor[0]
        elif data_idx >= data_length:
            data_idx = data_length - 1  # fill last data with tensor[-1]
        result = {}
        for name in self.model.input_names:
            result[name] = self.input_data[name][utterance_name][data_idx]
        return result

    def clear_input_data(self):
        step = 10
        for name in self.model.input_names:
            for utterance in self.input_data[name]:
                for idx, data in enumerate(self.input_data[name][utterance]):
                    if idx % step == 0:
                        data = self.input_data[name][utterance][idx]
                        self.input_data[name][utterance][idx] = np.zeros(data.shape, dtype=data.dtype)


def prepare_requests(
    inference_requests: List[InferenceRequest], timeout=config.wait_for_messages_timeout, random_data=False
):
    inference_infos = []
    for inference_request in inference_requests:
        port = inference_request.ovms.get_port(inference_request.api_type)
        inference_client = inference_request.api_type(port=port, model=inference_request.model)
        input_data = inference_client.create_client_data(inference_request)
        inference_info = InferenceInfo.create(
            inference_client,
            inference_request.model,
            input_data=input_data,
            inference_request=inference_request,
            timeout=timeout,
        )
        inference_infos.append(inference_info)
    return inference_infos


def predict_and_assert(inference_infos: List[InferenceInfo], validate_results=True, output_key=None):
    for i, inference_info in enumerate(inference_infos):
        logger.debug(f"Running predict request with {i} index for model '{inference_info.client.model_name}'")
        outputs = inference_info.predict()
        assert outputs, "Prediction returned no output"
        if validate_results:
            if inference_info.model.is_mediapipe:
                if isinstance(inference_info.model, SimpleMediaPipe) and output_key is None:
                    output_key = "output"
                else:
                    output_key = output_key
                MediaPipeInferenceResponse.create(inference_info, outputs).validate(
                    inference_info.input_data, output_key=output_key
                )
            elif inference_info.model.is_llm:
                LLMInferenceResponse.create(inference_info, outputs).validate()
            else:
                InferenceResponse.create(inference_info, outputs).validate(inference_info.input_data)
    logger.info("Predict finished.")


def predict_request(inference_request_list, parallel=False):
    result = {}
    if parallel:

        def execute_predict(infer_request):
            infer_result = infer_request.predict()
            return (infer_request.model.name, infer_result)

        arguments_list = [[x] for x in inference_request_list]
        result_list = run_all_actions(execute_predict, arguments_list)

        for model_name, infer_result in result_list:
            if model_name not in result:
                result[model_name] = []
            result[model_name].append(infer_result)
    else:
        for i, inference_info in enumerate(inference_request_list):
            logger.info(f"Running {i} predict request for model '{inference_info.client.model_name}'")
            outputs = inference_info.predict()
            assert outputs, "Prediction returned no output"
            if inference_info.model.name not in result:
                result[inference_info.model.name] = []
            result[inference_info.model.name].append(outputs)
    return result


def ensure_predict(inference_info):

    return retry_call(
        predict_and_assert,
        fargs=[[inference_info]],
        exceptions=(AssertionError, UnexpectedResponseError, RpcError),
        tries=60,
        delay=0.1,
    )


def healthy_check(models, port, api_type):
    for model in models:
        client = InferenceBuilder(model).create_client(api_type, port)
        if not issubclass(api_type, KserveWrapper):
            get_model_status(client, [Ovms.ModelStatus.AVAILABLE])

        data = model.prepare_input_data()
        request = client.prepare_request(input_objects=data)
        outputs = client.predict(request)
        model.validate_outputs(outputs)


def prepare_requests_and_run_predict(
    inference_requests: List[InferenceRequest], repeat: int = 10, validate_results=True, timeout=30, random_data=False
):
    inference_requests_total = inference_requests * repeat
    inference_infos = prepare_requests(inference_requests_total, timeout=timeout, random_data=random_data)
    predict_and_assert(inference_infos, validate_results=validate_results)


def prepare_and_run_set_of_predict_requests(ovms: OvmsInstance, models, api_type, number_of_infer_request_dict=None):
    inference_request_list = []
    for model in models:
        infer_requests_multiplier = 1
        if number_of_infer_request_dict is not None:
            infer_requests_multiplier = number_of_infer_request_dict[model.name]
        inference_request_list.extend(
            prepare_requests([InferenceRequest(model=model, api_type=api_type, ovms=ovms)] * infer_requests_multiplier)
        )

    return predict_request(inference_request_list)


def validate_accuracy_for_stateful_models(models, results, accuracy_level=0.1):
    for model in models:
        error_report_dict_list = model.calculate_error(results[model.name])
        for error_report_dict in error_report_dict_list:
            for utterance_name, error_result in error_report_dict.items():
                for output_name in model.output_names:
                    error_msg = (
                        f"Detect unexpected error level for model: {model.name} (utternace: "
                        f"{utterance_name} output: {output_name})! Expected error level < "
                        f"{accuracy_level} (detected: {error_result[output_name]})"
                    )
                    assert error_result[output_name] < accuracy_level, error_msg
    logger.info(f"Validate accuracy for stateful models - PASSED")


def get_model_status(client, accepted_model_states=None, model_version=None, port=None):
    model_state = None
    port = port if port is not None else client.port
    if client.serving == KFS:
        if accepted_model_states is not None:
            for elem in accepted_model_states:
                is_ready = True if elem == Ovms.ModelStatus.AVAILABLE else False
                try:
                    model_state = check_model_readiness(client.model, port, type(client), timeout=30, is_ready=is_ready)
                except ModelNotReadyException:
                    logger.info(f"Model state not in accepted state: {elem}")
                finally:
                    break
            else:
                raise ModelNotReadyException(f"Failed to check model: {client.model}")
        else:
            model_state = check_model_readiness(client.model, port, type(client))
    else:
        status = client.get_model_status()
        logger.debug(f"status: {status}")
        if model_version is None:
            model_state = Ovms.ModelStatus(status.model_version_status[0].state)
        else:
            for model_version_status in status.model_version_status:
                if model_version_status.version == model_version:
                    model_state = Ovms.ModelStatus(model_version_status.state)
                    break
        if accepted_model_states:
            if model_state not in accepted_model_states:
                model_str_name = client.model_name
                raise ValueError(f"Incorrect state of {model_str_name}: {model_state}")
    return model_state


def get_and_validate_model_status(inference, expected_models_status):
    status = inference.get_model_status()
    if expected_models_status is not None:
        assert len(expected_models_status) == len(status.model_version_status)

    for i, model_version_status in enumerate(status.model_version_status):
        model_state = model_version_status.state
        error_message = model_version_status.status.error_message
        version = model_version_status.version

        if expected_models_status is None or expected_models_status[i].get("accepted_states", None) is None:
            model_accepted_states = [
                get_model_status_pb2.ModelVersionStatus.START,
                get_model_status_pb2.ModelVersionStatus.AVAILABLE,
                get_model_status_pb2.ModelVersionStatus.UNLOADING,
                get_model_status_pb2.ModelVersionStatus.LOADING,
                get_model_status_pb2.ModelVersionStatus.END,
            ]
        else:
            model_accepted_states = expected_models_status[i]["accepted_states"]

        if model_state not in model_accepted_states:
            raise ValueError(f"Incorrect model state: {model_state}")

        if expected_models_status is None or expected_models_status[i].get("accepted_error_messages", None) is None:
            if model_state == get_model_status_pb2.ModelVersionStatus.LOADING:
                model_accepted_error_messages = ["OK", "UNKNOWN"]
            else:
                model_accepted_error_messages = ["OK"]
        else:
            model_accepted_error_messages = expected_models_status[i]["accepted_error_messages"]

        if error_message not in model_accepted_error_messages:
            raise ValueError(f"Incorrect error message: {model_state}")

        if expected_models_status is not None:
            assert version == expected_models_status[i]["version"]

    return status


def get_multiple_model_status(models_and_expected_state):
    for client, state in models_and_expected_state:
        try:
            model_state = get_model_status(client, accepted_model_states=[state])
        except (_InactiveRpcError, UnexpectedResponseError) as e:
            if state in [Ovms.ModelStatus.UNKNOWN, Ovms.ModelStatus.UNDEFINED]:
                pass  # It is expected exceptions for given ModelStatus so proceed.
            else:
                assert False, f"Unexpected exception {e} for fetching given model state: {client.model_name}"


def wait_for_model_status(client, accepted_model_states, model_version=None, timeout=None):
    skip_if_runtime(client.serving == KFS, FrameworkMessages.KFS_GET_MODEL_STATUS_NOT_SUPPORTED)
    if not timeout:
        timeout = client._model.get_ovms_loading_time()

    expected_exceptions = (
        UnexpectedResponseError,
        ConnectionError,
        _InactiveRpcError,
        IndexError,
        requests.exceptions.ConnectionError,
    )
    last_exception = ""
    received_status = None
    end_timeout = time.time() + timeout
    while time.time() <= end_timeout:
        try:
            model_status_response = client.get_model_status(version=model_version)
            received_status = model_status_response.model_version_status[0].state
            if Ovms.ModelStatus(received_status) in accepted_model_states:
                break
        except expected_exceptions as exception:
            last_exception = str(exception)
    else:
        raise TimeoutError()

        time.sleep(1)

    assert received_status, f"Failed to obtain model version status: {last_exception}"

    model_status_str = ", ".join(map(str, accepted_model_states))
    msg = f"Unexpected model status, current: {received_status} expected: {model_status_str}"
    assert Ovms.ModelStatus(received_status) in accepted_model_states, msg


def wait_for_model_meta(client, model, wait_time=1):
    end_time = datetime.now() + timedelta(seconds=60)
    validation_passed = False
    received_meta = ""
    logger.info(f"Waiting for receiving proper metadata for model {model.name}")
    while datetime.now() < end_time:
        try:
            received_meta = client.get_model_meta()
            client.validate_meta(model, received_meta)
            validation_passed = True
            if client.communication == REST:
                response = received_meta.text
            else:
                response = MessageToJson(received_meta)
            received_meta_str = json.loads(response)
            logger.info(f"Expected metadata received for model {model.name}:\r\n{received_meta_str}")
            break
        except (RpcError, AssertionError) as ex:
            time.sleep(wait_time)

    assert validation_passed, f"Unexpected model metadata, current: {received_meta} for model: {model}"


def prepare_v2_grpc_stub(port):
    final_server_address = TestEnvironment.get_server_address()
    url = f"{final_server_address}:{port}"
    channel = grpc.insecure_channel(url, options=channel_options)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    return grpc_stub


def prepare_v2_model_infer_request(port, api_type, input_data=None):
    if api_type.communication == GRPC:
        grpc_stub = prepare_v2_grpc_stub(port)
        request = api_type.get_predict_grpc_request(input_data)
        return request, grpc_stub
    else:
        raise NotImplementedError()


def check_model_readiness(model, port, kfs_api_type, is_ready=True, timeout=None):
    if timeout is None:
        timeout = model.get_ovms_loading_time()
    end_time = datetime.now() + timedelta(seconds=timeout)
    last_exception = ""
    success = False
    kfs_api = kfs_api_type(model=model, port=port)
    while datetime.now() < end_time:
        try:
            response = kfs_api.is_model_ready(model.name, model.version)
            if response and is_ready:
                logger.info(f"Model {model.name} Ready:\n{response}")
                success = True
                break
            elif not response and not is_ready:
                logger.info(f"Model {model.name} is not Ready: {response}")
                success = True
                break
        except (UnexpectedResponseError, _InactiveRpcError, InferenceServerException) as e:
            last_exception = str(e)
            if not is_ready and kfs_api.type == REST and (e.status == 404 or e.status == 503):
                success = True
                response = None
                break

        time.sleep(0.5)

    assert success, ModelNotReadyException(f"Failed to check model: {last_exception}")
    return response


def execute_kfs_live_ready(model, api_type, port, model_name=None, model_version=None):
    model_name = model_name if model_name is not None else model.name
    model_version = model_version if model_version is not None else model.version
    kfs_client = api_type(model=model, port=port)
    assert kfs_client.is_server_ready(), "Server is not ready"
    assert kfs_client.is_model_ready(model_name, model_version), f"Model {model_name} is not ready"


def prepare_mediapipe_requests(ovms, model, api_type, port, input_key):
    request = InferenceRequest(ovms=ovms, model=model, api_type=api_type)
    inference_client = api_type(port=port, model=model)
    input_data = model.prepare_input_data(input_key=input_key)
    inference_info = InferenceInfo.create(
        inference_client, request.model, input_data=input_data, inference_request=request
    )
    return [inference_info]


def run_mediapipe_inference(
    ovms,
    model,
    api_type,
    port,
    input_key=MediaPipeConstants.DEFAULT_INPUT_STREAM,
    output_key=MediaPipeConstants.DEFAULT_OUTPUT_STREAM,
):
    logger.info(f"Run inference for {model.name}")
    inference_infos = prepare_mediapipe_requests(ovms, model, api_type, port, input_key)
    predict_and_assert(inference_infos, output_key=output_key)


def prepare_mediapipe_binary_requests(ovms, model, api_type, input_key, dataset=None):
    dataset = DefaultBinaryDataset(image_format=Ovms.JPG_IMAGE_FORMAT, offset=0) if dataset is None else dataset
    request = BinaryInferenceRequest(ovms=ovms, model=model, api_type=api_type, dataset=dataset)

    inference_infos = prepare_requests([request])
    for inference_info in inference_infos:
        for key in inference_info.input_data.keys():
            inference_info.input_data[input_key] = inference_info.input_data[key]
            del inference_info.input_data[key]
            break
    return inference_infos


def run_streaming_inference(model, api_type, port):
    if api_type.type == GRPC:
        client = api_type(model=model, port=port)
        prompts = []
        for i in range(len(model.inputs)):
            prompts.extend(LanguageModelDataset(i).get_str_input_data())
        batch_size = model.batch_size
        if batch_size is not None and batch_size > 1:
            # Generate various random text
            prompts_data = [
                prompts if i == 0 else LanguageModelDataset.generate_random_text_list(model.inputs_number)
                for i in range(batch_size)
            ]
            provided_input = prompts_data
        else:
            prompts_data = [prompts]
            provided_input = prompts

        logger.debug(f"Provided input: {provided_input}")
        results = streaming_api_inference_language_models(model, client, prompts=prompts_data)
        model.validate_outputs(outputs=results, provided_input=provided_input)
    else:
        raise NotImplementedError


def log_request_info(request_params, model_name, prompt):
    logger.info(f"Run request with parameters: '{request_params}' for model: '{model_name}' with prompt: '{prompt}'.")


def validate_ttr(data, reference=0.5):
    # Calculate Type-Token Ratio (TTR)
    assert data != "", f"Empty data: {data}. Please verify model's response."
    words = data.split()
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    assert ttr > reference, f"Output TTR is too low: {ttr}"


def run_llm_inference(
        model: Union[ModelInfo, List[ModelInfo]],
        api_type,
        port,
        endpoint,
        dataset=None,
        input_data_type=None,
        validate_outputs=True,
        validate_outputs_ttr=True,
        allow_empty_response=False,
        timeout=None,
        log_request=True,
        ttr_reference=0.5,
        **kwargs,
):
    if api_type.type == REST:
        api_client = api_type(model=model, port=port)
        infer_request = kwargs.get("infer_request", None)
        request_parameters = kwargs.get("request_parameters", {})
        if isinstance(model, list):
            assert endpoint == OpenAIWrapper.MODELS_LIST, f"model parameter cannot be list when used with {endpoint}"
            input_objects = None
            model_name = None
        else:
            assert endpoint != OpenAIWrapper.MODELS_LIST, f"model parameter has to be list when used with {endpoint}"
            input_objects = model.prepare_input_data(dataset=dataset, input_data_type=input_data_type)
            model_base_path = kwargs.get("model_base_path", None)
            model_name = model_base_path if model_base_path is not None else model.name
        if endpoint != CohereWrapper.RERANK:
            infer_request = infer_request if infer_request is not None else \
                LLMInferenceRequest(api_type=api_client, request_parameters=request_parameters)
        else:
            infer_request = infer_request if infer_request is not None else \
                RerankLLMInferenceRequest(api_type=api_client, request_parameters=request_parameters)
        outputs = None
        if endpoint == OpenAIWrapper.CHAT_COMPLETIONS:
            messages = ChatCompletionsApi.prepare_chat_completions_input_content(input_objects)
            if log_request:
                log_request_info(request_parameters, model_name, messages)
            raw_outputs = infer_request.create_chat_completions(messages, model_name=model_name, timeout=timeout)
            raw_outputs = list(raw_outputs) if infer_request.stream and raw_outputs else raw_outputs
            if validate_outputs:
                outputs = LLMValidationUtils.validate_chat_completions_outputs(
                    model_name=model_name,
                    outputs=raw_outputs,
                    stream=infer_request.stream,
                    allow_empty_response=allow_empty_response,
                )
                if validate_outputs_ttr:
                    validate_ttr(outputs[0], reference=ttr_reference)
        elif endpoint == OpenAIWrapper.COMPLETIONS:
            prompt = CompletionsApi.prepare_completions_input_content(input_objects)
            if log_request:
                log_request_info(request_parameters, model_name, prompt)
            raw_outputs = infer_request.create_completions(prompt, model_name=model_name, timeout=timeout)
            raw_outputs = list(raw_outputs) if infer_request.stream and raw_outputs else raw_outputs
            if validate_outputs:
                outputs = LLMValidationUtils.validate_completions_outputs(
                    model_name=model_name,
                    outputs=raw_outputs,
                    stream=infer_request.stream,
                    allow_empty_response=allow_empty_response,
                    model_instance=model,
                )
                if validate_outputs_ttr:
                    validate_ttr(outputs[0], reference=ttr_reference)
        elif endpoint == OpenAIWrapper.RESPONSES:
            input_content = ResponsesApi.prepare_responses_input_content(input_objects)
            if log_request:
                log_request_info(request_parameters, model_name, input_content)
            raw_outputs = infer_request.create_responses(input_content, model_name=model_name, timeout=timeout)
            raw_outputs = list(raw_outputs) if infer_request.stream and raw_outputs else raw_outputs
            if validate_outputs:
                outputs = LLMValidationUtils.validate_responses_outputs(
                    model_name=model_name,
                    outputs=raw_outputs,
                    stream=infer_request.stream,
                    allow_empty_response=allow_empty_response,
                )
                if validate_outputs_ttr:
                    validate_ttr(outputs[0], reference=ttr_reference)
        elif endpoint == OpenAIWrapper.EMBEDDINGS:
            embeddings_input = EmbeddingsApi.prepare_embeddings_input_content(input_objects)
            if log_request:
                log_request_info(request_parameters, model_name, embeddings_input)
            raw_outputs = infer_request.create_embeddings(embeddings_input, model_name=model_name, timeout=timeout)
            if validate_outputs:
                outputs = LLMValidationUtils.validate_embeddings_outputs(
                    model_name=model_name,
                    outputs=raw_outputs,
                    allow_empty_response=allow_empty_response,
                )
        elif endpoint == CohereWrapper.RERANK:
            rerank_input = RerankApi.prepare_rerank_input_content(input_objects)
            if log_request:
                log_request_info(request_parameters, model_name, rerank_input)
            raw_outputs = infer_request.create_rerank(rerank_input, model_name=model_name)
            if validate_outputs:
                outputs = LLMValidationUtils.validate_rerank_outputs(
                    model_name=model_name,
                    outputs=raw_outputs,
                    allow_empty_response=allow_empty_response,
                )
        elif endpoint == OpenAIWrapper.IMAGES_GENERATIONS:
            prompt = ImagesApi.prepare_image_generation_input_content(input_objects)
            if log_request:
                log_request_info(request_parameters, model_name, prompt)
            raw_outputs = infer_request.create_image_generation(prompt, model_name=model_name, timeout=timeout)
            if validate_outputs:
                outputs = LLMValidationUtils.validate_image_outputs(
                    model_name=model_name,
                    outputs=raw_outputs,
                    image_path=kwargs.get("image_path", None),
                    request_parameters=request_parameters,
                )
        elif endpoint == OpenAIWrapper.IMAGES_EDITS:
            prompt, image_path = ImagesApi.prepare_image_edit_input_content(input_objects)
            mask_path = dataset.mask_path if hasattr(dataset, "mask_path") else None
            if log_request:
                message = f"Run request with parameters: '{request_parameters}' for model: '{model_name}' " \
                          f"with prompt: '{prompt}', image_path: '{image_path}'"
                if mask_path is not None:
                    message += f", mask_path: '{mask_path}'"
                logger.info(message)
            raw_outputs = infer_request.create_image_edit(
                prompt, image_path, mask_path=mask_path, model_name=model_name, timeout=timeout)
            if validate_outputs:
                outputs = LLMValidationUtils.validate_image_outputs(
                    model_name=model_name,
                    outputs=raw_outputs,
                    image_path=kwargs.get("image_path", None),
                    request_parameters=request_parameters,
                )
        elif endpoint == OpenAIWrapper.MODELS_LIST:
            raw_outputs = infer_request.create_models_list()
            if validate_outputs:
                outputs = LLMValidationUtils.validate_models_list_outputs(models=model, outputs=raw_outputs)
        elif endpoint == OpenAIWrapper.MODELS_RETRIEVE:
            raw_outputs = infer_request.create_models_retrieve()
            if validate_outputs:
                outputs = LLMValidationUtils.validate_models_retrieve_outputs(model_name=model_name, outputs=raw_outputs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return outputs, raw_outputs


def run_audio_inference(
        model: ModelInfo,
        api_type,
        port,
        endpoint,
        dataset=None,
        validate_outputs=True,
        validate_output_wer=False,
        allow_empty_response=False,
        timeout=None,
        log_request=True,
        wer_threshold=0.4,
        **kwargs,
):
    if api_type.type == REST:
        api_client = api_type(model=model, port=port)
        request_parameters = kwargs.get("request_parameters", {})
        infer_request = kwargs.get("infer_request") or LLMInferenceRequest(
            api_type=api_client, request_parameters=request_parameters
        )

        model_name = kwargs.get("model_base_path") or model.name
        reference_text, reference_audio_file = AudioApi.prepare_audio_input_content(
            model.prepare_input_data(dataset=dataset)
        )
        outputs = None
        raw_outputs = None

        if endpoint == OpenAIWrapper.AUDIO_SPEECH:
            speech_file_path = kwargs.get("speech_file_path")
            assert speech_file_path is not None, "speech_file_path is required for audio speech endpoint"
            if log_request:
                log_request_info(request_parameters, model_name, reference_text)
            raw_outputs = infer_request.create_audio_speech(
                reference_text, speech_file_path, model_name=model_name, timeout=timeout
            )
            if validate_outputs:
                outputs = LLMValidationUtils.validate_audio_speech_outputs(
                    speech_file_path=speech_file_path,
                    allow_empty_response=allow_empty_response,
                )
        elif endpoint in OpenAIWrapper.AVAILABLE_AUDIO_ASR_ENDPOINTS:
            if log_request:
                log_request_info(request_parameters, model_name, reference_audio_file)
            create_fn = (
                infer_request.create_audio_transcription
                if endpoint == OpenAIWrapper.AUDIO_TRANSCRIPTIONS
                else infer_request.create_audio_translation
            )
            raw_outputs = create_fn(reference_audio_file, model_name=model_name, timeout=timeout)
            if validate_outputs:
                outputs = LLMValidationUtils.validate_audio_asr_outputs(
                    outputs=raw_outputs,
                    allow_empty_response=allow_empty_response,
                )
                if validate_output_wer and reference_text:
                    LLMValidationUtils.validate_wer(reference_text, raw_outputs, threshold=wer_threshold)
        else:
            raise NotImplementedError

        return outputs, raw_outputs

    raise NotImplementedError

def run_llm_inference_and_validate_against_reference(
    model,
    api_type,
    port,
    endpoint,
    request_parameters,
    reference_outputs,
):
    outputs = run_llm_inference(
        model,
        api_type,
        port,
        endpoint,
        request_parameters=request_parameters,
    )
    assert (
        outputs == reference_outputs
    ), f"Output messages:\n'{outputs}'\ndo not match reference:\n'{reference_outputs}'"


def streaming_api_inference_language_models(mediapipe_model, kfs_client, prompts):
    # Fetched from: https://github.com/openvinotoolkit/model_server/blob/main/demos/python_demos/llm_text_generation/client_stream.py
    results_decoded = []
    event, client, results = prepare_streaming_api_inference(mediapipe_model, kfs_client, prompts, results_decoded)
    infer_inputs = streaming_api_inference_prepare_infer_inputs(mediapipe_model, prompts)
    client.async_stream_infer(model_name=mediapipe_model.name, inputs=infer_inputs)
    event.wait(timeout=10)
    client.stop_stream()
    logger.info("Stream stopped")
    return results


def prepare_streaming_api_inference(mediapipe_model, kfs_client, prompts, results_decoded):

    def callback(result, error):

        def decode_result(result, output_name, results_decoded):
            logger.debug(f"Result as numpy for output_name {output_name}: {result.as_numpy(output_name)}")
            if len(prompts) == 1:
                results_decoded.append(result.as_numpy(output_name).tobytes().decode())
            else:
                deserialized_results = deserialize_bytes_tensor(result._result.raw_output_contents[0])
                decoded_list = [content.decode() for content in deserialized_results]
                results_decoded.append(decoded_list)
            logger.debug(f"Results decoded for output_name {output_name}: {results_decoded}")
            return results_decoded

        if error:
            raise error
        elif any(result.as_numpy(output_name) is not None for output_name in mediapipe_model.output_names):
            for output_name in mediapipe_model.output_names:
                if result.as_numpy(output_name) is not None:
                    decode_result(result, output_name, results_decoded)
                    break
        else:
            raise StreamingApiException(f"Unexpected output: {result}")

    client = tritonclient.grpc.InferenceServerClient(kfs_client.url, channel_args=STREAMING_CHANNEL_ARGS, verbose=True)

    event = Event()
    client.start_stream(callback=callback)
    return event, client, results_decoded


def streaming_api_inference_prepare_infer_inputs(mediapipe_model, prompts):
    _infer_inputs = []
    for i, _input in enumerate(mediapipe_model.inputs.keys()):
        infer_input = tritonclient.grpc.InferInput(mediapipe_model.input_names[i], [len(prompts)], "BYTES")
        if len(prompts) == 1:
            if isinstance(prompts[0], list):
                infer_input._raw_content = prompts[0][i].encode()
            else:
                infer_input._raw_content = prompts[0].encode()
        else:
            if mediapipe_model.inputs_number == 1:
                infer_input._raw_content = serialize_byte_tensor(np.array(prompts, dtype=np.object_)).item()
            else:
                infer_input._raw_content = serialize_byte_tensor(np.array(prompts[i], dtype=np.object_)).item()
        _infer_inputs.append(infer_input)

    logger.debug(f"Infer inputs raw content: {[inp._raw_content for inp in _infer_inputs]}")
    return _infer_inputs
