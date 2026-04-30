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

import numpy as np

from tests.functional.utils.inference.communication import GRPC
from tests.functional.utils.logger import get_logger
from tests.functional.constants.generative_ai import GenerativeAIPluginConfig
from tests.functional.models.models_datasets import LanguageModelDataset
from tests.functional.constants.ovms import Ovms
from tests.functional.constants.pipelines import MediaPipe, NodesConnection, NodeType, PythonGraphNode
from tests.functional.object_model.mediapipe_calculators import HttpLLMCalculator, PythonCalculator, \
    ImageGenCalculator, EmbeddingsCalculatorOV, RerankCalculatorOV, S2tCalculator, T2sCalculator
from tests.functional import config

logger = get_logger(__name__)


class SimplePythonCustomNodeMediaPipe(MediaPipe):
    inputs_number = 1
    input_name: str = "text_input"
    outputs_number = 1
    output_name: str = "text_output"
    input_names: list = None
    output_names: list = None
    child_nodes: list = None
    inputs: dict = None
    outputs: list = None
    base_path: str = ""
    name: str = "python_model"
    is_python_custom_node: bool = True
    batch_size: int = 1

    def __init__(self, handler_path, node_name="upper_text", loopback=False, initialize_graphs=True, **kwargs):
        self.node_name = node_name
        self.config = {}
        self.prepare_llm_model_inputs_outputs(model=self, dataset=LanguageModelDataset, kwargs=kwargs)

        super().__init__(**kwargs)
        self.calculators = [
            PythonCalculator(handler_path=handler_path, model=self, node_name=node_name, loopback=loopback)
        ]
        self.graphs = [calc.create_proto_content(model=self) for calc in self.calculators] if initialize_graphs else []

        if self.child_nodes is None:
            self.child_nodes = []
        if self.inputs is None:
            self.inputs = {}
        if self.outputs is None:
            self.outputs = []

        self.handler_path = handler_path

    @staticmethod
    def prepare_llm_model_inputs_outputs(model, dataset, **kwargs):
        inputs_number = kwargs.get("inputs_number", None)
        model.inputs_number = inputs_number if inputs_number is not None else model.inputs_number
        model.inputs = {
            f"{model.input_name}{i}": {'shape': [-1, -1], 'dtype': str, 'dataset': dataset(data_sample=i)}
            for i in range(model.inputs_number)
        }
        outputs_number = kwargs.get("outputs_number", None)
        model.outputs_number = outputs_number if outputs_number is not None else model.outputs_number
        model.outputs = {
            f"{model.output_name}{i}": {'shape': [-1, 512], 'dtype': str} for i in range(model.outputs_number)
        }
        return model

    @property
    def input_names(self):
        return list(self.inputs.keys())

    @property
    def output_names(self):
        return list(self.outputs.keys())

    def get_config(self):
        return None

    def prepare_resources(self, base_location):
        resource_locations = []
        for calc in self.calculators:
            resource_locations.append(calc.prepare_resources(base_location))
        return resource_locations

    def prepare_input_data(self, batch_size=None, input_key=None, dataset=None, input_data_type=None):
        if dataset is not None:
            if not isinstance(dataset, type):
                dataset_obj = dataset
            else:
                dataset_obj = dataset()
            input_data = {input_name: dataset_obj.get_data(None, None, None) for input_name in self.input_names}
        elif input_data_type == "string":
            input_data = {
                input_name: self.inputs[input_name]["dataset"].get_string_data()
                for input_name in self.input_names
            }
        else:
            input_data = {
                input_name: self.inputs[input_name]["dataset"].get_data(None, None, None)
                for input_name in self.input_names
            }
        return input_data

    def get_expected_output(self, input_data: dict, client_type: str = None):
        output_data = {}

        for i, input_name in enumerate(self.inputs.keys()):
            if self.node_name == "incrementer":
                multiply_value = 8
                decoded_text = "".join(item.decode() for i in range(multiply_value) for item in input_data[input_name])
            else:
                multiply_value = 1
                decoded_text = "".join(item.decode() for item in input_data[input_name]).upper()

            if client_type == GRPC:
                filler = Ovms.OUTPUT_FILLER
                input_text = filler.join(decoded_text) + filler
                char_array = [ord(char) for char in input_text]
                output_data[self.output_names[i]] = np.array(char_array, dtype=np.object_)
            else:
                input_text = decoded_text
                char_array = [ord(char) for char in input_text]
                # For REST API and BYTES type, every batch is always preceding by the 4 bytes, that contains its size
                # [42, 0, 0, 0] is constant value for "Lorem ipsum dolor sit amet"
                # https://github.com/openvinotoolkit/model_server/blob/main/docs/model_server_rest_api_kfs.md
                splitted = np.array_split(np.array(char_array, dtype=np.object_), multiply_value)
                extended_with_length = [np.insert(elem, 0, [42, 0, 0, 0]) for elem in splitted]
                output_data[self.output_names[i]] = np.concatenate(extended_with_length, dtype=np.object_)

        return output_data

    def validate_outputs(self, outputs, expected_output_shapes=None, provided_input=None):
        assert outputs, f"No output collected for node {self.node_name}"
        output_mismatch_error = f"Output mismatch for node {self.node_name} outputs: {outputs}"
        if self.node_name == "upper_text":
            assert len(outputs) == self.outputs_number, (
                f"Invalid outputs length: {len(outputs)}; " f"Expected: {self.outputs_number}"
            )
            if self.batch_size is not None and self.batch_size > 1:
                for _input_elem, _output_elem in zip(provided_input, outputs):
                    for i, j in zip(_input_elem, _output_elem):
                        assert i.upper() == j, output_mismatch_error
            else:
                for i, _input in enumerate(provided_input):
                    assert _input.upper() == outputs[i], output_mismatch_error
        elif self.node_name == "loopback_upper_text":
            # Expected output: [
            # "Lorem ipsum dolor sit amet", "LOrem ipsum dolor sit amet", "LoRem ipsum dolor sit amet", ...
            # ]
            _input = provided_input[0]
            expected_outputs = [_input[:j] + _input[j].upper() + _input[j + 1:] for j in range(len(_input))]
            assert expected_outputs == outputs, output_mismatch_error
        elif self.node_name == "incrementer":
            for i, _input in enumerate(provided_input):
                # Incrementer value (2) defined in: incrementer.py
                assert _input * 2 ** self.chain_length == outputs[i], output_mismatch_error
        else:
            raise NotImplementedError


class PythonCustomNodeChainMediaPipe(SimplePythonCustomNodeMediaPipe):
    def __init__(self, models=None, handler_path=None, **kwargs):
        self.node_name = "incrementer"
        self.input_name = "first_number"
        self.output_name = "last_number"
        self.inputs = {self.input_name: {"shape": [-1, -1], "dtype": str, "dataset": LanguageModelDataset()}}
        self.outputs = {self.output_name: {"shape": [-1, 512], "dtype": str}}
        self.chain_length = len(models)
        self.calculators = []
        self.models = models
        self.handler_path = handler_path
        self.config = {}
        self.regular_models = []
        self.is_mediapipe = True
        self.is_python_custom_node = True

        for model in self.models:
            calc = PythonCalculator(handler_path=self.handler_path, model=model, node_name=model.node_name)
            self.calculators.append(calc)

        if self.child_nodes is None:
            self.child_nodes = []

        self.create_header = False
        self._initialize(models)

    def _create_nodes(self, models=None):
        final_input_name = "first_number"
        final_output_name = "last_number"

        request = PythonGraphNode("request", node_type=NodeType.Input, output_names=[final_input_name])
        output = PythonGraphNode("output", node_type=NodeType.Output, input_names=[final_output_name])

        model_nodes = []
        for i, model in enumerate(models):
            output_stream = f"{model.node_name}_{i}_output"
            calculator = self.calculators[i]
            if i == 0:
                python_node = PythonGraphNode(
                    name=model.node_name,
                    model=model,
                    calculator=calculator,
                    input_stream=final_input_name,
                    output_stream=output_stream,
                )
                model_nodes.append(python_node)
            elif i == (len(models) - 1):
                python_node = PythonGraphNode(
                    name=model.node_name,
                    model=model,
                    calculator=calculator,
                    input_stream=model_nodes[i - 1].output_stream,
                    output_stream=final_output_name,
                )
                model_nodes.append(python_node)
            elif 0 < i < len(models):
                python_node = PythonGraphNode(
                    name=model.node_name,
                    model=model,
                    calculator=calculator,
                    input_stream=model_nodes[i - 1].output_stream,
                    output_stream=output_stream,
                )
                model_nodes.append(python_node)

        for i, node in enumerate(model_nodes):
            if i == 0:
                NodesConnection.connect(node, 0, request, 0)
            elif i == (len(model_nodes) - 1):
                NodesConnection.connect(output, 0, model_nodes[i - 1], 0)
            elif 0 < i < len(model_nodes):
                NodesConnection.connect(node, 0, model_nodes[i - 1], 0)

        nodes = model_nodes + [request, output]
        return nodes

    @staticmethod
    def is_pipeline():
        return True


class SimpleLLM(SimplePythonCustomNodeMediaPipe):
    inputs_number = 1
    input_name: str = "input"
    outputs_number = 1
    output_name: str = "output"
    child_nodes: list = None
    inputs: dict = None
    outputs: list = None
    base_path: str = ""
    name: str = ""
    pbtxt_name: str = "simple_llm"
    is_python_custom_node: bool = True
    is_llm: bool = True
    calculator_class = HttpLLMCalculator
    precision: str = None
    allows_reasoning: bool = False

    def __init__(self, models_path, node_name="LLMExecutor", loopback=True, initialize_graphs=True, **kwargs):
        model = kwargs["model"]
        self.node_name = node_name
        self.config = {}
        dataset = model.get_default_dataset()
        self.prepare_llm_model_inputs_outputs(model=self, dataset=dataset, kwargs=kwargs)

        self.regular_models = []
        self.is_mediapipe = True
        self.is_python_custom_node = True
        best_of_limit = kwargs.get("best_of_limit", None)
        max_tokens_limit = kwargs.get("max_tokens_limit", None)
        kv_cache_size = kwargs.get("kv_cache_size", config.kv_cache_size_value)
        plugin_config = kwargs.get(
            "plugin_config", {GenerativeAIPluginConfig.KV_CACHE_PRECISION: config.kv_cache_precision_value}
        )
        enable_tool_guided_generation = kwargs.get("enable_tool_guided_generation", False)
        self.calculators = [
            self.calculator_class(
                models_path=models_path,
                model=model,
                node_name=node_name,
                loopback=loopback,
                best_of_limit=best_of_limit,
                max_tokens_limit=max_tokens_limit,
                kv_cache_size=kv_cache_size,
                plugin_config=plugin_config,
                device=self.target_device,
                enable_tool_guided_generation=enable_tool_guided_generation,
            )
        ]
        self.name = self.calculators[0].model.name
        self.graphs = [calc.create_proto_content(model=self) for calc in self.calculators] if initialize_graphs else []

        if self.child_nodes is None:
            self.child_nodes = []
        if self.inputs is None:
            self.inputs = {}
        if self.outputs is None:
            self.outputs = []

        self.models_path = models_path
        self.model_timeout = getattr(model, "model_timeout", None)


class SimpleImageGenerationLLM(SimpleLLM):
    inputs_number = 1
    input_name: str = "input"
    outputs_number = 1
    output_name: str = "output"
    child_nodes: list = None
    inputs: dict = None
    outputs: list = None
    base_path: str = ""
    name: str = ""
    pbtxt_name: str = "simple_llm"
    is_python_custom_node: bool = True
    is_llm: bool = True
    calculator_class = ImageGenCalculator
    precision: str = None

    def __init__(self, models_path, node_name="ImageGenExecutor", initialize_graphs=True, **kwargs):
        model = kwargs["model"]
        self.node_name = node_name
        self.config = {}
        dataset = model.get_default_dataset()
        self.prepare_llm_model_inputs_outputs(model=self, dataset=dataset, kwargs=kwargs)

        self.regular_models = []
        self.is_mediapipe = True
        self.is_python_custom_node = True
        kv_cache_size = kwargs.get("kv_cache_size", config.kv_cache_size_value)
        target_device = kwargs.get("target_device", self.target_device)
        resolution = kwargs.get("resolution", None)
        self.calculators = [
            self.calculator_class(
                models_path=models_path,
                model=model,
                node_name=node_name,
                kv_cache_size=kv_cache_size,
                device=target_device,
                resolution=resolution,
            )
        ]
        self.name = self.calculators[0].model.name
        self.graphs = [calc.create_proto_content(model=self) for calc in self.calculators] if initialize_graphs else []

        if self.child_nodes is None:
            self.child_nodes = []
        if self.inputs is None:
            self.inputs = {}
        if self.outputs is None:
            self.outputs = []

        self.models_path = models_path
        self.model_timeout = getattr(model, "model_timeout", None)


class SimpleFeatureExtractionLLM(SimpleLLM):
    inputs_number: int = 1
    input_name: str = "input"
    outputs_number: int = 1
    output_name: str = "output"
    child_nodes: list = None
    inputs: dict = None
    outputs: list = None
    base_path: str = ""
    name: str = ""
    pbtxt_name: str = "simple_llm"
    is_python_custom_node: bool = True
    is_llm: bool = True
    use_subconfig: bool = True
    is_feature_extraction: bool = True
    calculator_class = EmbeddingsCalculatorOV

    def __init__(self, models_path, node_name="LLMExecutor", loopback=False, initialize_graphs=True, **kwargs):
        super().__init__(models_path, node_name, loopback, initialize_graphs, **kwargs)


class SimpleRerankLLM(SimpleLLM):
    inputs_number: int = 1
    input_name: str = "input"
    outputs_number: int = 1
    output_name: str = "output"
    child_nodes: list = None
    inputs: dict = None
    outputs: list = None
    base_path: str = ""
    name: str = ""
    pbtxt_name: str = "simple_llm"
    is_python_custom_node: bool = True
    is_llm: bool = True
    use_subconfig: bool = True
    is_rerank: bool = True
    calculator_class = RerankCalculatorOV

    def __init__(self, models_path, node_name="LLMExecutor", loopback=False, initialize_graphs=True, **kwargs):
        super().__init__(models_path, node_name, loopback, initialize_graphs, **kwargs)


class SimpleAsrModel(SimpleLLM):
    inputs_number: int = 1
    input_name: str = "input"
    outputs_number: int = 1
    output_name: str = "output"
    child_nodes: list = None
    inputs: dict = None
    outputs: list = None
    base_path: str = ""
    name: str = ""
    pbtxt_name: str = "simple_llm"
    is_python_custom_node: bool = True
    is_llm: bool = False
    is_asr_model: bool = True
    calculator_class = S2tCalculator

    def __init__(self, models_path, node_name="S2tExecutor", loopback=False, initialize_graphs=True, **kwargs):
        super().__init__(models_path, node_name, loopback, initialize_graphs, **kwargs)


class SimpleTtsModel(SimpleLLM):
    inputs_number: int = 1
    input_name: str = "input"
    outputs_number: int = 1
    output_name: str = "output"
    child_nodes: list = None
    inputs: dict = None
    outputs: list = None
    base_path: str = ""
    name: str = ""
    pbtxt_name: str = "simple_llm"
    is_python_custom_node: bool = True
    is_llm: bool = False
    is_tts_model: bool = True
    calculator_class = T2sCalculator

    def __init__(self, models_path, node_name="T2sExecutor", loopback=False, initialize_graphs=True, **kwargs):
        super().__init__(models_path, node_name, loopback, initialize_graphs, **kwargs)
