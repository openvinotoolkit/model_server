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
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from tests.functional.utils.assertions import ConvertModelException, InvalidReturnCodeException
from tests.functional.utils.logger import get_logger
from tests.functional.utils.process import Process

from llm.constants import LLMPluginConfig
from llm.remote_downloader import link_llm_models

from ovms.config import mediapipe_repo_branch, ovms_c_repo_path, kv_cache_size_value, kv_cache_precision_value, \
    pipeline_type as config_pipeline_type, max_num_batched_tokens, enable_prefix_caching_config
from tests.functional.constants.models import ModelInfo
from tests.functional.constants.target_device import TargetDevice
from tests.functional.constants.ovms import Config, MediaPipeConstants
from tests.functional.constants.paths import Paths
from tests.functional.object_model.test_environment import TestEnvironment

logger = get_logger(__name__)

dummy_mediapipe_calculators = [os.path.join(ovms_c_repo_path, "src/test/mediapipe/graphdummyadapterfull.pbtxt")]


@dataclass
class MediaPipeCalculator:
    src_type: str = "cc"
    src_dir: str = None
    src_file_path: str = None
    src_filename: str = None
    name: str = None

    @classmethod
    def prepare_proto_calculator(cls, parameters, config_path_on_host, config_file=None):
        # Create new graph files content from MediaPipeCalculator.

        calculators = []
        mediapipe_models = [model for model in parameters.models if model.is_mediapipe]
        config_data = (
            parameters.custom_config
            if parameters.custom_config is not None
            else json.loads(Path(config_file).read_text()) if config_file is not None else {}
        )
        for mediapipe_model in mediapipe_models:
            dst_path = os.path.join(config_path_on_host, mediapipe_model.name) if config_path_on_host is not None \
                else os.path.join(TestEnvironment.current.base_dir, parameters.name, Paths.MODELS_PATH_NAME,
                                  mediapipe_model.name)
            # Scenario 1. With any graph path
            if parameters.use_custom_graphs and parameters.custom_graph_paths:
                # Copy existing graphs from custom_graph_paths.
                for calc in parameters.custom_graph_paths:
                    calculators.append(calc)
                    real_path = os.path.expanduser(calc)
                    real_path = os.path.realpath(real_path)
                    logger.info(
                        "Copy custom calculator file to {}, content:\n{}".format(dst_path, Path(real_path).read_text())
                    )
                    Path(dst_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(real_path, dst_path)
            # Scenario 2. With mediapipe pipelines and python custom nodes
            elif mediapipe_model.graphs:
                calculator_class = PythonCalculator if mediapipe_model.is_python_custom_node else cls
                model = mediapipe_model if mediapipe_model.is_python_custom_node else None
                for graph in mediapipe_model.graphs:
                    calculators.append(graph)
                    mediapipe_model_graph_paths = [
                        elem.get("graph_path", "")
                        for elem in config_data[Config.MEDIAPIPE_CONFIG_LIST]
                        if elem["name"] == mediapipe_model.name
                    ]
                    for graph_path in mediapipe_model_graph_paths:
                        calculator_class.save(
                            model=model,
                            content=graph,
                            dst_path=dst_path,
                            filename=os.path.basename(graph_path),
                            save_only=True,
                        )
            # Scenario 3. With SimpleModelMediapipe
            else:
                # Get graph paths from mediapipe_model
                if ((not config_data and mediapipe_model.single_mediapipe_model_mode)
                        or mediapipe_model.pbtxt_name is not None):
                    mediapipe_model_graph_paths = [f"{mediapipe_model.pbtxt_name}.pbtxt"]
                elif not Config.MEDIAPIPE_CONFIG_LIST in config_data:
                    mediapipe_model_graph_paths = [
                        elem["config"].get("graph_path", "")
                        for elem in config_data[Config.MODEL_CONFIG_LIST]
                        if elem["config"]["name"] == mediapipe_model.name
                    ]
                else:
                    mediapipe_model_graph_paths = [
                        elem.get("graph_path", "")
                        for elem in config_data[Config.MEDIAPIPE_CONFIG_LIST]
                        if elem["name"] == mediapipe_model.name
                    ]

                contents = {}
                for regular_model in mediapipe_model.regular_models:
                    for calc in mediapipe_model.calculators:
                        calculators.append(calc)
                        model = calc.model if calc.model is not None else regular_model
                        contents.update({calc.name: calc.create_proto_content(model=model)})

                content_to_save = " ".join(value for key, value in contents.items())
                if all(["pbtxt" in elem for elem in mediapipe_model_graph_paths]):
                    for path in mediapipe_model_graph_paths:
                        filename = os.path.basename(path)
                        cls.save(mediapipe_model, content_to_save, dst_path=dst_path, filename=filename)
                else:
                    filename = Paths.GRAPH_NAME
                    cls.save(mediapipe_model, content_to_save, dst_path=dst_path, filename=filename)

                mediapipe_model.graphs = [content_to_save]

        return calculators

    @classmethod
    def create_proto_header(
        cls,
        model,
        input_stream=MediaPipeConstants.DEFAULT_INPUT_STREAM,
        output_stream=MediaPipeConstants.DEFAULT_OUTPUT_STREAM,
    ):
        input_streams = ""
        output_streams = ""
        if model is not None:
            for i, model_input in enumerate(model.inputs, start=0):
                input_streams += f'input_stream: "{input_stream}_{i}" \n'

            for i, model_output in enumerate(model.outputs, start=0):
                output_streams += f'output_stream: "{output_stream}_{i}" \n'
        else:
            if isinstance(input_stream, List):
                for inp in input_stream:
                    input_streams += f'input_stream: "{inp}" \n'
            if isinstance(output_stream, List):
                for out in output_stream:
                    output_streams += f'output_stream: "{out}" \n'
        return f"{input_streams}\n{output_streams}\n"

    def create_input_output_streams(self, model, input_stream, output_stream):
        # Note: Key name must be capitalized:
        # https://github.com/google/mediapipe/blob/master/mediapipe/framework/tool/validate_name.cc#L35

        input_streams = []
        output_streams = []
        if isinstance(input_stream, List):
            for elem in input_stream:
                input_streams.append(elem.split(":")[-1])
        else:
            input_streams = [input_stream.split(":")[-1] for i in range(len(model.inputs))]

        if isinstance(output_stream, List):
            for elem in output_stream:
                output_streams.append(elem.split(":")[-1])
        else:
            output_streams = [output_stream.split(":")[-1] for i in range(len(model.outputs))]

        inputs = ""
        model_name = self.get_upper_model_name(model)
        for (i, model_input), inp_stream in zip(enumerate(model.inputs, start=0), input_streams):
            inp = f"{model_name}_INPUT_{i}"
            inputs += (
                f'input_stream: "{inp}:{inp_stream}_{i}" \n'
                if inp_stream == MediaPipeConstants.DEFAULT_INPUT_STREAM
                else f'input_stream: "{inp}:{inp_stream}" \n'
            )

        outputs = ""
        for (i, model_output), out_stream in zip(enumerate(model.outputs, start=0), output_streams):
            out = f"{model_name}_OUTPUT_{i}"
            outputs += (
                f'output_stream: "{out}:{out_stream}_{i}" \n'
                if out_stream == MediaPipeConstants.DEFAULT_OUTPUT_STREAM
                else f'output_stream: "{out}:{out_stream}" \n'
            )

        return inputs, outputs

    @classmethod
    def get_full_content(cls, content, model, input_stream, output_stream):
        header = cls.create_proto_header(model, input_stream, output_stream)
        content = header + content
        return content

    @classmethod
    def save(
        cls,
        model,
        content,
        dst_path,
        filename=Paths.GRAPH_NAME,
        input_stream=MediaPipeConstants.DEFAULT_INPUT_STREAM,
        output_stream=MediaPipeConstants.DEFAULT_OUTPUT_STREAM,
        save_only=False,
    ):
        """Save .pbtxt file to config_path_on_host location with given filename"""
        if not save_only:
            content = cls.get_full_content(content, model, input_stream, output_stream)
        Path(dst_path).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(dst_path, filename)
        with open(file_path, "w+") as f:
            f.write(content)
        logger.info(f"Saving calculator file to {file_path}, content:\n{content}")
        return file_path

    @staticmethod
    def load(filepath):
        with open(filepath, "r") as f:
            data = f.read()
        return data

    @classmethod
    def get_upper_model_name(cls, model):
        return cls.get_valid_model_name(model).upper()

    @staticmethod
    def get_valid_model_name(model):
        model_name = ("model_" + model.name) if model.name[0].isdigit() else model.name
        return model_name.replace("-", "_").lower()


@dataclass
class OvmsCMediaPipeCalculator(MediaPipeCalculator):

    def __post_init__(self):
        self.src_dir = (
            f"https://github.com/openvinotoolkit/mediapipe/blob/{mediapipe_repo_branch}/mediapipe/calculators/ovms"
        )
        if self.src_filename is not None:
            self.src_file_path = os.path.join(self.src_dir, f"{self.src_filename}.{self.src_type}")

    def create_input_output_tensor_names(self, model):
        # Note: Key name must be capitalized:
        # https://github.com/google/mediapipe/blob/master/mediapipe/framework/tool/validate_name.cc#L35

        model_inputs_keys = list(model.inputs.keys())
        model_outputs_keys = list(model.outputs.keys())

        input_tensors = ""
        model_name = self.get_upper_model_name(model)
        for i, model_input in enumerate(model_inputs_keys, start=0):
            inp = f"{model_name}_INPUT_{i}"
            input_tensor_names = f'tag_to_input_tensor_names {{key: "{inp}" value: "{model_input}"}}'
            input_tensors += f"{input_tensor_names} \n"

        output_tensors = ""
        for i, model_output in enumerate(model_outputs_keys, start=0):
            out = f"{model_name}_OUTPUT_{i}"
            output_tensor_names = f'tag_to_output_tensor_names {{key: "{out}" value: "{model_output}"}}'
            output_tensors += f"{output_tensor_names} \n"

        return input_tensors, output_tensors


@dataclass
class OvmsCUnitTestMediaPipeCalculator(MediaPipeCalculator):

    def __post_init__(self):
        self.src_dir = os.path.join(ovms_c_repo_path, "src", "test", "mediapipe")
        self.src_file_path = os.path.join(self.src_dir, f"{self.src_filename}.{self.src_type}")


@dataclass
class OVMSOVCalculator(OvmsCMediaPipeCalculator):
    name: str = "OVMSOVCalculator"
    src_filename: str = "ovms_calculator"
    model: ModelInfo = None
    src_dir: str = os.path.join(ovms_c_repo_path, "src", "mediapipe_calculators")

    def create_proto_content(
        self,
        model,
        input_stream=MediaPipeConstants.DEFAULT_INPUT_STREAM,
        output_stream=MediaPipeConstants.DEFAULT_OUTPUT_STREAM,
        create_header=True,
    ):
        model = self.model if self.model is not None else model
        input_streams, output_streams = self.create_input_output_streams(model, input_stream, output_stream)
        input_tensor_names, output_tensor_names = self.create_input_output_tensor_names(model)

        content = (
            "node {\n"
            f'calculator: "{self.name}"\n'
            f"{input_streams}\n"
            f"{output_streams}\n"
            "node_options: {\n"
            "[type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {\n"
            f'servable_name: "{model.name}"\n'
            f'servable_version: "{model.version}"\n'
            f"{input_tensor_names}\n"
            f"{output_tensor_names}"
            "}}}"
        )

        return content


@dataclass
class OpenVINOInferenceCalculator(OvmsCMediaPipeCalculator):
    name: str = "OpenVINOInferenceCalculator"
    src_filename: str = "openvinoinferencecalculator"
    model: ModelInfo = None
    session: str = None
    input_stream: str = None
    output_stream: str = None

    def create_proto_content(
        self,
        model,
        input_stream=MediaPipeConstants.DEFAULT_INPUT_STREAM,
        output_stream=MediaPipeConstants.DEFAULT_OUTPUT_STREAM,
        create_header=True,
    ):
        model = self.model if self.model is not None else model
        input_stream = self.input_stream if self.input_stream is not None else input_stream
        output_stream = self.output_stream if self.output_stream is not None else output_stream
        input_streams, output_streams = self.create_input_output_streams(model, input_stream, output_stream)
        input_tensor_names, output_tensor_names = self.create_input_output_tensor_names(model)
        session = MediaPipeCalculator.get_valid_model_name(model) if self.session is None else self.session

        content = (
            "node: {\n"
            f'calculator: "{self.name}"\n'
            f'input_side_packet: "SESSION:{session}"\n'
            f"{input_streams}\n"
            f"{output_streams}\n"
            "node_options: {\n"
            "[type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {\n"
            f"{input_tensor_names}\n"
            f"{output_tensor_names}"
            "}}}"
        )

        return content


@dataclass
class OpenVINOModelServerSessionCalculator(OvmsCMediaPipeCalculator):
    name: str = "OpenVINOModelServerSessionCalculator"
    src_filename: str = "openvinomodelserversessioncalculator"
    model: ModelInfo = None
    session: str = None
    model_name: str = None

    def create_proto_content(self, model, input_stream=None, output_stream=None, create_header=True):
        model = self.model if self.model is not None else model
        model_name = self.model_name if self.model_name is not None else model.name
        model.name = model_name
        session = MediaPipeCalculator.get_valid_model_name(model) if self.session is None else self.session
        content = (
            "node: {\n"
            f'calculator: "{self.name}"\n'
            f'output_side_packet: "SESSION:{session}" \n'
            "node_options: {\n"
            "[type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {\n"
            f'servable_name: "{model_name}"\n'
            f'servable_version: "{model.version}"\n'
            "}}}"
        )

        return content


@dataclass
class InputSidePacketUserTestCalc(OvmsCUnitTestMediaPipeCalculator):
    name: str = "InputSidePacketUserTestCalc"
    src_filename: str = "inputsidepacketusertestcalc"


@dataclass
class OVMSTestKFSPassCalculator(OvmsCUnitTestMediaPipeCalculator):
    name: str = "OVMSTestKFSPassCalculator"
    src_filename: str = "ovms_kfs_calculator"


@dataclass
class CorruptedFileCalculator(OVMSOVCalculator):
    name: str = "CorruptedFileCalculator"
    src_filename: str = None
    model: ModelInfo = None

    def create_proto_content(
        self,
        model,
        input_stream=MediaPipeConstants.DEFAULT_INPUT_STREAM,
        output_stream=MediaPipeConstants.DEFAULT_OUTPUT_STREAM,
        create_header=True,
    ):
        model = self.model if self.model is not None else model
        ovms_ov_content = super(CorruptedFileCalculator, self).create_proto_content(model)
        content = ovms_ov_content.replace("input_stream", self.name)
        return content


@dataclass
class PythonCalculator(MediaPipeCalculator):
    name: str = "PythonExecutorCalculator"
    src_filename: str = None
    input_side_packet: str = "PYTHON_NODE_RESOURCES:py"
    model: ModelInfo = None
    handler_path: str = None
    handler_path_internal: str = "/models/python_model/model.py"
    input_streams: str = None
    output_streams: str = None
    node_name: str = None
    loopback: bool = False

    def __post_init__(self):
        if self.model is not None:
            self.input_streams = "\n".join(
                f'input_stream: "PYTHON_MODEL_INPUT_{i}:{model_input}"'
                for i, model_input in enumerate(self.model.input_names)
            )
            self.output_streams = "\n".join(
                f'output_stream: "PYTHON_MODEL_OUTPUT_{i}:{model_output}"'
                for i, model_output in enumerate(self.model.output_names)
            )

    def create_proto_content(self, model, input_stream=None, output_stream=None, create_header=True):
        model = self.model if self.model is not None else model
        if input_stream is not None and output_stream is not None:
            # Scenario, when input_stream and output_stream have unique values, e.g. PythonCustomNodeChainMediaPipe
            input_streams, output_streams = self.create_input_output_streams(model, input_stream, output_stream)
        else:
            # Default scenario, when self.input_streams are defined in __post_init__
            input_streams = self.input_streams
            output_streams = self.output_streams

        header = (
            self.create_proto_header(model) if create_header else ""
        )  # If create_header=False, header will be create in graph_refresh()
        if not self.loopback:
            content = f"""{header}
node: {{
    name: "{self.node_name}"
    calculator: "{self.name}"
    input_side_packet: "{self.input_side_packet}"
    {input_streams}
    {output_streams}
    node_options: {{
        [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {{
            handler_path: "{self.handler_path_internal}"
        }}
    }}
}}"""
        else:
            content = f"""{header}
node: {{
    name: "{self.node_name}"
    calculator: "{self.name}"
    input_side_packet: "{self.input_side_packet}"
    {input_streams}
    input_stream: "LOOPBACK:loopback"
    input_stream_info: {{
        tag_index: "LOOPBACK:0",
        back_edge: true
    }}
    input_stream_handler {{
        input_stream_handler: "SyncSetInputStreamHandler",
        options {{
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {{
                sync_set {{
                    tag_index: "LOOPBACK:0"
                }}
            }}
        }}
    }}
    {output_streams}
    output_stream: "LOOPBACK:loopback"
    node_options: {{
        [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {{
            handler_path: "{self.handler_path_internal}"
        }}
    }}
}}"""
        return content

    def prepare_resources(self, base_location):
        dst = Path(base_location, f"./{self.handler_path_internal}")
        dst.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(self.handler_path, dst)
        return str(dst)

    @classmethod
    def create_proto_header(cls, model, input_stream="text_input", output_stream="text_output"):
        input_streams = ""
        output_streams = ""
        if model is not None:
            input_streams = "\n".join(
                f'input_stream: "OVMS_PY_TENSOR{i}:{model_input}"' for i, model_input in enumerate(model.input_names)
            )
            output_streams = "\n".join(
                f'output_stream: "OVMS_PY_TENSOR{i}:{model_output}"'
                for i, model_output in enumerate(model.output_names)
            )
        else:
            if isinstance(input_stream, List):
                for i, inp in enumerate(input_stream):
                    input_streams += f'input_stream: "OVMS_PY_TENSOR{i}:{inp}" \n'
            if isinstance(output_stream, List):
                for i, out in enumerate(output_stream):
                    output_streams += f'output_stream: "OVMS_PY_TENSOR{i}:{out}" \n'
        return f"{input_streams}\n{output_streams}\n"


@dataclass
class LLMCalculator(PythonCalculator):
    name: str = "LLMExecutor"
    src_filename: str = "llm_calculator"
    input_side_packet: str = "LLM_NODE_RESOURCES:llm"
    model: ModelInfo = None
    models_path: str = None
    models_path_internal: str = "./"
    input_streams: str = None
    output_streams: str = None
    node_name: str = None
    loopback: bool = False
    kv_cache_size: int = kv_cache_size_value
    plugin_config: dict = field(default_factory={LLMPluginConfig.KV_CACHE_PRECISION: kv_cache_precision_value})
    best_of_limit: int = None
    max_tokens_limit: int = None
    device: str = None
    enable_tool_guided_generation: bool = False

    def __post_init__(self):
        self.input_streams = 'input_stream: "REQUEST:in"'
        self.output_streams = 'output_stream: "RESPONSE:out"'
        self.src_dir = os.path.join(ovms_c_repo_path, "src", "llm")
        self.src_file_path = os.path.join(self.src_dir, f"{self.src_filename}.{self.src_type}")

    def create_node_content(self, header, input_streams, output_streams):
        best_of_limit_str = f"\nbest_of_limit: {self.best_of_limit}," if self.best_of_limit is not None else ""
        max_tokens_limit_str = (
            f"\nmax_tokens_limit: {self.max_tokens_limit}," if self.max_tokens_limit is not None else ""
        )
        device = f'device: "{self.device}",' if self.device is not None else ""
        content = f"""{header}
node: {{
    name: "{self.node_name}"
    calculator: "{self.name}"
    input_side_packet: "{self.input_side_packet}"
    {input_streams}
    {output_streams}
    node_options: {{
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {{
            models_path: "{self.models_path_internal}",{best_of_limit_str}{max_tokens_limit_str}
            cache_size: {self.kv_cache_size},
            {device}
        }}
    }}
}}"""
        return content

    def create_proto_content(self, model, input_stream=None, output_stream=None, create_header=True):
        input_streams = self.input_streams
        output_streams = self.output_streams
        header = self.create_proto_header(self.input_streams, self.output_streams)
        content = self.create_node_content(header, input_streams, output_streams)
        return content

    @staticmethod
    def _copy_model_tree(proc, src, dst):
        if "C:\\" in src:
            proc.run_and_check(
                f"robocopy /J /E /NP /NFL /NJH \"{src}\" \"{dst}\"",
                env=os.environ.copy(),
                exit_code_check=1,
                exception_type=InvalidReturnCodeException,
                timeout=1800,
            )
        else:
            shutil.copytree(src, dst)

    def prepare_resources(self, base_location):
        dst_base = Path(base_location, "models")
        dst = Path(dst_base, f"./{self.model.name}")
        dst.parent.mkdir(exist_ok=True, parents=True)
        if not Path.exists(dst):
            proc = Process()
            proc.disable_check_stderr()
            try:
                self._copy_model_tree(proc, self.models_path, dst)
            except Exception as e:  # pylint: disable=broad-exception-caught
                if dst.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                if self.model.is_llm and any([
                    isinstance(e, FileNotFoundError),   # unix
                    isinstance(e, InvalidReturnCodeException) and "Code: 16" in e.args[0],  # windows
                ]):
                    # try reloading llm model and retry copy
                    failed_models = link_llm_models([type(self.model)], proc, skip_existing_models=True)
                    if failed_models:
                        raise ConvertModelException(f"Couldn't link LLM models: {failed_models}.")
                    try:
                        self._copy_model_tree(proc, self.models_path, dst)
                    except Exception as e:
                        if dst.exists():
                            shutil.rmtree(dst, ignore_errors=True)
                        raise e
                else:
                    raise e
        return str(dst_base)

    @classmethod
    def create_proto_header(cls, input_stream="text_input", output_stream="text_output"):
        return "\n".join([input_stream, output_stream])


@dataclass
class HttpLLMCalculator(LLMCalculator):
    name: str = "HttpLLMCalculator"
    src_filename: str = "http_llm_calculator"
    input_side_packet: str = "LLM_NODE_RESOURCES:llm"
    model: ModelInfo = None
    models_path: str = None
    models_path_internal: str = "."
    input_streams: str = None
    output_streams: str = None
    node_name: str = None
    loopback: bool = True
    kv_cache_size: int = kv_cache_size_value
    plugin_config: dict = field(default_factory={LLMPluginConfig.KV_CACHE_PRECISION: kv_cache_precision_value})
    best_of_limit: int = None
    max_tokens_limit: int = None
    device: str = None

    def __post_init__(self):
        super().__post_init__()
        self.input_streams = 'input_stream: "HTTP_REQUEST_PAYLOAD:input"'
        self.output_streams = 'output_stream: "HTTP_RESPONSE_PAYLOAD:output"'

    def create_node_content(self, header, input_streams, output_streams):
        best_of_limit_str = f"\nbest_of_limit: {self.best_of_limit}," if self.best_of_limit is not None else ""
        max_tokens_limit_str = (
            f"\nmax_tokens_limit: {self.max_tokens_limit}," if self.max_tokens_limit is not None else ""
        )

        def get_plugin_config_params_list(plugin_config_dict):
            plugin_config_params = []
            for plugin_config_key, plugin_config_value in plugin_config_dict.items():
                if plugin_config_value is not None:
                    if type(plugin_config_value) == int:
                        plugin_config_params.append(f'"{plugin_config_key}": {plugin_config_value}')
                    elif type(plugin_config_value) == bool:
                        plugin_config_value = "true" if plugin_config_value else "false"
                        plugin_config_params.append(f'"{plugin_config_key}": {plugin_config_value}')
                    elif type(plugin_config_value) == str:
                        plugin_config_params.append(f'"{plugin_config_key}": "{plugin_config_value}"')
                    elif type(plugin_config_value) == dict:
                        plugin_config_params_dict = get_plugin_config_params_list(plugin_config_value)
                        plugin_config_params_dict_str = ', '.join(plugin_config_params_dict)
                        plugin_config_params.append(f"\"{plugin_config_key}\": {{ {plugin_config_params_dict_str} }}")
                    else:
                        raise NotImplementedError()
            return plugin_config_params

        plugin_config_str = ""
        plugin_config_params = get_plugin_config_params_list(self.plugin_config)
        if plugin_config_params:
            plugin_config_str = f"plugin_config: '{{ {', '.join(plugin_config_params)} }}',"

        tool_parser_str = ""
        if any([
            self.model.tool_parser is not None and self.model.tools_enabled,
            self.model.tool_parser is not None and self.model.is_agentic,
        ]):
            tool_parser_str = f'tool_parser: "{self.model.tool_parser}",'
            if self.model.reasoning_parser is not None:
                tool_parser_str += f' reasoning_parser: "{self.model.reasoning_parser}",'

        pipeline_type_str = ""
        model_pipeline_type = getattr(self.model, "pipeline_type", None)

        if model_pipeline_type is not None:
            pipeline_type_str = f'pipeline_type: {model_pipeline_type},'
        elif config_pipeline_type is not None:
            pipeline_type_str = f'pipeline_type: {config_pipeline_type},'

        max_num_batched_tokens_str = ""
        model_max_num_batched_tokens = self.model.max_num_batched_tokens
        max_num_batched_tokens_value = model_max_num_batched_tokens if model_max_num_batched_tokens is not None \
            else max_num_batched_tokens
        if max_num_batched_tokens_value is not None:
            max_num_batched_tokens_str = f"max_num_batched_tokens: {max_num_batched_tokens_value}"

        enable_prefix_caching_str = ""
        if enable_prefix_caching_config:
            enable_prefix_caching_str = f"enable_prefix_caching: true"

        tool_guided_str = ""
        if self.model.enable_tool_guided_generation and self.enable_tool_guided_generation:
            tool_guided_str = "enable_tool_guided_generation: true"

        device = f'device: "{self.device}",' if self.device is not None else ""

        content = f"""{header}
node: {{
    name: "{self.node_name}"
    calculator: "{self.name}"
    input_side_packet: "{self.input_side_packet}"
    {input_streams}
    input_stream: "LOOPBACK:loopback"
    input_stream_info: {{
        tag_index: "LOOPBACK:0",
        back_edge: true
    }}
    node_options: {{
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {{
            models_path: "{self.models_path_internal}",{best_of_limit_str}{max_tokens_limit_str}
            cache_size: {self.kv_cache_size},
            {device}
            {tool_parser_str}
            {plugin_config_str}
            {pipeline_type_str}
            {tool_guided_str}
            {max_num_batched_tokens_str}
            {enable_prefix_caching_str}
        }}
    }}
    input_stream_handler {{
        input_stream_handler: "SyncSetInputStreamHandler",
        options {{
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {{
                sync_set {{
                    tag_index: "LOOPBACK:0"
                }}
            }}
        }}
    }}
    {output_streams}
    output_stream: "LOOPBACK:loopback"
}}"""
        return content


@dataclass
class ImageGenCalculator(LLMCalculator):
    name: str = "ImageGenCalculator"
    src_filename: str = "image_gen_calculator"
    input_side_packet: str = "IMAGE_GEN_NODE_RESOURCES:pipes"
    model: ModelInfo = None
    models_path: str = None
    models_path_internal: str = "."
    input_streams: str = None
    output_streams: str = None
    node_name: str = None
    loopback: bool = True
    kv_cache_size: int = kv_cache_size_value
    plugin_config: dict = None
    best_of_limit: int = None
    max_tokens_limit: int = None
    device: str = None
    resolution: str = None

    def __post_init__(self):
        self.input_streams = 'input_stream: "HTTP_REQUEST_PAYLOAD:input"'
        self.output_streams = 'output_stream: "HTTP_RESPONSE_PAYLOAD:output"'

    def create_node_content(self, header, input_streams, output_streams):
        resolution = f'resolution: "{self.resolution}"' if self.resolution is not None else ""
        device = f'device: "{self.device}",' if self.device is not None else ""

        content = f"""{header}
node: {{
    name: "{self.node_name}"
    calculator: "{self.name}"
    input_side_packet: "{self.input_side_packet}"
    {input_streams}
    node_options: {{
        [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {{
            models_path: "{self.models_path_internal}",
            {device}
            {resolution}
        }}
    }}
    {output_streams}
}}"""
        return content


@dataclass
class EmbeddingsCalculatorOV(LLMCalculator):
    name: str = "EmbeddingsCalculatorOV"
    src_filename: str = "embeddings_calculator_ov"
    model: ModelInfo = None
    models_path: str = None
    models_path_internal: str = "."
    input_streams: str = None
    output_streams: str = None
    node_name: str = None
    loopback: bool = False
    kv_cache_size: int = kv_cache_size_value
    plugin_config: dict = None
    best_of_limit: int = None
    max_tokens_limit: int = None
    device: str = None
    normalize_embeddings: str = "true"
    truncate: str = "false"

    def __post_init__(self):
        super().__post_init__()
        self.input_streams = 'input_stream: "REQUEST_PAYLOAD:input"'
        self.output_streams = 'output_stream: "RESPONSE_PAYLOAD:output"'

    def create_node_content(self, header, input_streams, output_streams):
        limits = []
        if self.best_of_limit is not None: limits.append(f"best_of_limit: {self.best_of_limit}")
        if self.max_tokens_limit is not None: limits.append(f"max_tokens_limit: {self.max_tokens_limit}")

        models_path_line = f'models_path: "{self.models_path_internal}"'
        if limits: models_path_line += f", {', '.join(limits)}"

        pooling_str = ""
        model_obj = getattr(self, "model", None)
        if model_obj is not None and getattr(model_obj, "pooling", None) is not None:
            pooling_str = f'pooling: {model_obj.pooling},'

        target_device_str = f'target_device: "{self.device}",' if self.device is not None else ""
        num_streams = 2 if self.device == TargetDevice.GPU else 1
        plugin_config_str = f'plugin_config: \'{{ "NUM_STREAMS": "{num_streams}" }}\','

        content =f"""{header}
node {{
    name: "{self.node_name}",
    calculator: "{self.name}"
    input_side_packet: "EMBEDDINGS_NODE_RESOURCES:embeddings_servable"
    {input_streams}
    {output_streams}
    node_options: {{
        [type.googleapis.com / mediapipe.EmbeddingsCalculatorOVOptions]: {{
            {models_path_line}
            normalize_embeddings: {self.normalize_embeddings},
            truncate: {self.truncate},
            {pooling_str}
            {target_device_str}
            {plugin_config_str}
        }}
    }}
}}
"""
        return content


@dataclass
class RerankCalculatorOV(LLMCalculator):
    name: str = "RerankCalculatorOV"
    src_filename: str = "rerank_calculator_ov"
    model: ModelInfo = None
    models_path: str = None
    models_path_internal: str = "."
    input_streams: str = None
    output_streams: str = None
    node_name: str = None
    loopback: bool = False
    kv_cache_size: int = kv_cache_size_value
    plugin_config: dict = None
    best_of_limit: int = None
    max_tokens_limit: int = None
    device: str = None
    max_allowed_chunks: int = 10000

    def __post_init__(self):
        super().__post_init__()
        self.input_streams = 'input_stream: "REQUEST_PAYLOAD:input"'
        self.output_streams = 'output_stream: "RESPONSE_PAYLOAD:output"'

    def create_node_content(self, header, input_streams, output_streams):
        target_device_str = f'target_device: "{self.device}",' if self.device is not None else ""
        num_streams = 2 if self.device == TargetDevice.GPU else 1
        plugin_config_str = f'plugin_config: \'{{ "NUM_STREAMS": "{num_streams}" }}\','
        content = f"""{header}
node {{
    name: "{self.node_name}",
    calculator: "{self.name}"
    input_side_packet: "RERANK_NODE_RESOURCES:rerank_servable"
    {input_streams}
    {output_streams}
    node_options: {{
        [type.googleapis.com / mediapipe.RerankCalculatorOVOptions]: {{
            models_path: "{self.models_path_internal}",
            max_allowed_chunks: {self.max_allowed_chunks},
            {target_device_str}
            {plugin_config_str}
        }}
    }}
}}"""
        return content


@dataclass
class S2tCalculator(LLMCalculator):
    name: str = "S2tCalculator"
    src_filename: str = "s2t_calculator"
    model: ModelInfo = None
    models_path: str = None
    models_path_internal: str = "."
    input_streams: str = None
    output_streams: str = None
    node_name: str = None
    loopback: bool = False
    kv_cache_size: int = kv_cache_size_value
    plugin_config: dict = None
    best_of_limit: int = None
    max_tokens_limit: int = None
    device: str = None

    def __post_init__(self):
        super().__post_init__()
        self.input_streams = 'input_stream: "HTTP_REQUEST_PAYLOAD:input"'
        self.output_streams = 'output_stream: "HTTP_RESPONSE_PAYLOAD:output"'

    def create_node_content(self, header, input_streams, output_streams):
        target_device_str = f'target_device: "{self.device}",' if self.device is not None else ""
        num_streams = 2 if self.device == TargetDevice.GPU else 1
        plugin_config_str = f'plugin_config: \'{{ "NUM_STREAMS": "{num_streams}" }}\','
        content = f"""{header}
node {{
    name: "{self.node_name}",
    calculator: "{self.name}"
    input_side_packet: "STT_NODE_RESOURCES:s2t_servable"
    {input_streams}
    {output_streams}
    node_options: {{
        [type.googleapis.com / mediapipe.S2tCalculatorOptions]: {{
            models_path: "{self.models_path_internal}",
            {target_device_str}
            {plugin_config_str}
        }}
    }}
}}"""
        return content


@dataclass
class T2sCalculator(LLMCalculator):
    name: str = "T2sCalculator"
    src_filename: str = "t2s_calculator"
    model: ModelInfo = None
    models_path: str = None
    models_path_internal: str = "."
    input_streams: str = None
    output_streams: str = None
    node_name: str = None
    loopback: bool = False
    kv_cache_size: int = kv_cache_size_value
    plugin_config: dict = None
    best_of_limit: int = None
    max_tokens_limit: int = None
    device: str = None

    def __post_init__(self):
        super().__post_init__()
        self.input_streams = 'input_stream: "HTTP_REQUEST_PAYLOAD:input"'
        self.output_streams = 'output_stream: "HTTP_RESPONSE_PAYLOAD:output"'

    def create_node_content(self, header, input_streams, output_streams):
        target_device_str = f'target_device: "{self.device}",' if self.device is not None else ""
        num_streams = 2 if self.device == TargetDevice.GPU else 1
        plugin_config_str = f'plugin_config: \'{{ "NUM_STREAMS": "{num_streams}" }}\','
        content = f"""{header}
node {{
    name: "{self.node_name}",
    calculator: "{self.name}"
    input_side_packet: "TTS_NODE_RESOURCES:t2s_servable"
    {input_streams}
    {output_streams}
    node_options: {{
        [type.googleapis.com / mediapipe.T2sCalculatorOptions]: {{
            models_path: "{self.models_path_internal}",
            {target_device_str}
            {plugin_config_str}
        }}
    }}
}}"""
        return content
