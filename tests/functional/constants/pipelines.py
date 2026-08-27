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

import os
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from pathlib import Path

import numpy as np

from tests.functional.models.models import ModelInfo
from tests.functional.constants.ovms import Config
from tests.functional.constants.paths import Paths
from tests.functional.object_model.custom_node import CustomNode
from tests.functional.object_model.mediapipe_calculators import MediaPipeCalculator, PythonCalculator


class NodesConnection:
    def __init__(self, target_node, target_node_input_id, source_node, source_output_id):
        self.target_node = target_node
        self.target_node_input_id = target_node_input_id
        self.source_node = source_node
        self.source_node_output_id = source_output_id

    def __str__(self):
        return f"{self.target_node.name}[{self.target_node_input_id}]<-{self.source_node}[{self.source_node_output_id}]"

    def get_source_data_item_name(self):
        return self.source_node.get_output_name(self.source_node_output_id)

    def get_target_data_item_name(self):
        return self.target_node.get_input_name(self.target_node_input_id)

    def get_target_input(self):
        model_input_name = self.target_node.model.input_names[self.target_node_input_id]
        return model_input_name, self.target_node.model.inputs[model_input_name]

    def get_source_output(self):
        model_output_name = self.source_node.model.output_names[self.source_node_output_id]
        return model_output_name, self.source_node.model.outputs[model_output_name]

    @classmethod
    def connect(cls, target_node, target_node_input_id, source_node, source_output_id):
        connection = NodesConnection(target_node, target_node_input_id, source_node, source_output_id)
        connection.target_node.input_connections.append(connection)
        connection.source_node.output_connections.append(connection)


class NodeType(Enum):
    Input = "INPUT"
    DL_MODEL = "DL model"
    Output = "OUTPUT"
    Custom = "custom"


class Node:
    def __init__(
        self,
        name,
        model=None,
        node_type=None,
        input_names=None,
        output_names=None,
        demultiply_count=None,
        gather_from_node=None,
    ):

        if node_type is None:
            if isinstance(model, CustomNode):
                node_type = NodeType.Custom
            else:
                node_type = NodeType.DL_MODEL

        self.name = name
        self.model = model
        self.input_connections = []
        self.output_connections = []
        self.node_type = node_type
        self.input_names = input_names
        self.output_names = output_names
        self.demultiply_count = demultiply_count
        self.gather_from_node = gather_from_node

    def __str__(self):
        return self.name

    def get_input_name(self, id):
        if self.input_names:
            return self.input_names[id]
        else:
            if self.node_type == NodeType.Output:
                prefix = "output"
            else:
                prefix = "input"
            return f"{prefix}_{id}"

    def get_output_name(self, id):
        if self.output_names:
            return self.output_names[id]
        else:
            if self.node_type == NodeType.Input:
                prefix = "input"
            else:
                prefix = self.model.name
            return f"{prefix}_{id}"

    def _change_name(self, names, old_name, new_name):
        for index, name in enumerate(names):
            if name == old_name:
                names[index] = new_name
                return

    def change_input_name(self, old_name, new_name):
        self._change_name(self.input_names, old_name, new_name)

    def change_output_name(self, old_name, new_name):
        self._change_name(self.output_names, old_name, new_name)

    def get_expected_output(self, input_data, client_type: str = None):
        mapped_input_data = {}
        for input_connection in self.input_connections:
            input_name = self.model.input_names[input_connection.target_node_input_id]
            mapped_input_data[input_name] = input_connection.source_node[input_connection.source_node_output_id]

        return self.model.get_expected_output(mapped_input_data)

    def dump_config(self):
        """
        "nodes": [
          {
            "name": "node_1",
            "type": "custom",
            "inputs": [
              {
                "input_numbers": {
                  "node_name": "request",
                  "data_item": "input"
                }
              }
            ],
            "outputs": [
              {
                "data_item": "output_numbers",
                "alias": "node_1_output_0"
              }
            ],
            "library_name": "lib_node_add_sub",
            "params": {
              "add_value": "5",
              "sub_value": "4"
            }
          }
        """
        config = {"name": self.name, "type": f"{self.node_type.value}", "inputs": [], "outputs": []}

        if self.demultiply_count is not None:
            config["demultiply_count"] = self.demultiply_count

        if self.gather_from_node is not None:
            config["gather_from_node"] = self.gather_from_node

        if isinstance(self.model, CustomNode):
            config["library_name"] = self.model.name
            node_parameters = self.model.get_parameters()
            if node_parameters:
                config["params"] = node_parameters
        else:
            config["model_name"] = self.model.name

        for input_connection in self.input_connections:  # a single model input can be connected only to a single source
            input_name = self.model.input_names[input_connection.target_node_input_id]
            input_mapping = {
                "node_name": input_connection.source_node.name,
                "data_item": input_connection.get_source_data_item_name(),
            }
            config["inputs"].append({input_name: input_mapping})

        for id, model_output_name in enumerate(
            self.model.output_names
        ):  # a single model output can be connected to multiple targets
            config["outputs"].append({"data_item": model_output_name, "alias": self.get_output_name(id)})

        return config


class MediaPipeGraphNode(Node):
    def __init__(
        self,
        name,
        model=None,
        node_type=None,
        input_names=None,
        output_names=None,
        demultiply_count=None,
        gather_from_node=None,
        calculator=None,
        servable_name=None,
        servable_version=None,
        input_stream=None,
        output_stream=None,
        tag_to_input_tensor_names=None,
        tag_to_output_tensor_names=None,
    ):

        super().__init__(name, model, node_type, input_names, output_names, demultiply_count, gather_from_node)

        self.calculator = calculator
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.tag_to_input_tensor_names = tag_to_input_tensor_names
        self.tag_to_output_tensor_names = tag_to_output_tensor_names

        if self.model is not None:
            self.servable_name = servable_name if servable_name is not None else self.model.name
            self.servable_version = servable_version if servable_version is not None else str(self.model.version)
        else:
            self.servable_name = servable_name
            self.servable_version = servable_version


class PythonGraphNode(MediaPipeGraphNode):
    def __init__(
        self,
        name,
        calculator=None,
        model=None,
        input_side_packet=None,
        input_stream=None,
        output_stream=None,
        handler_path=None,
        node_options=None,
        node_type=None,
        input_names=None,
        output_names=None,
    ):

        super().__init__(name, model, node_type, input_names, output_names)

        self.calculator = calculator
        self.input_side_packet = input_side_packet
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.handler_path = handler_path
        self.node_options = node_options


class Pipeline(ModelInfo):
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.child_nodes = []
        self.config = {}
        self.inputs = {}
        self.outputs = {}
        self.demultiply_count = None  # demultiply_count could be dynamic (value: 0, -1)
        self.default_demultiply_count_value = (
            7  # real demultiply count value used in validation mechanism - generate output shape
        )
        assert kwargs.get("use_mapping", None) is not True
        self.is_mediapipe = False

    def set_expected_demultiply(self, expected_value, dynamic_mode=False):
        self.demultiply_count = -1 if dynamic_mode else expected_value
        self.default_demultiply_count_value = expected_value

    def get_demultiply_count(self):
        return self.demultiply_count

    @property
    def is_on_cloud(self):
        return False

    @abstractmethod
    def _create_nodes(self, models=None):
        raise NotImplementedError()

    def _initialize(self, models=None):
        self.child_nodes.extend(self._create_nodes(models))
        self.initialize_inputs_outputs()
        self.config_refresh()

    def initialize_inputs_outputs(self):
        input_node = self.get_input_node()
        output_names = []
        for connection in input_node.output_connections:
            _, value = connection.get_target_input()
            input_name = connection.get_source_data_item_name()
            self.inputs[input_name] = deepcopy(value)
            output_names.append(input_name)

        if input_node.output_names is None:
            input_node.output_names = output_names

        output_node = self.get_output_node()
        input_names = []
        for connection in output_node.input_connections:
            _, value = connection.get_source_output()
            input_name = connection.get_target_data_item_name()
            self.outputs[input_name] = deepcopy(value)
            input_names.append(input_name)

        if output_node.input_names is None:
            output_node.input_names = input_names

    def prepare_resources(self, base_location):
        resource_locations = []
        models = self.get_models()
        for model in models:
            resource_location_list = model.prepare_resources(base_location)
            if resource_location_list is not None:
                for location in resource_location_list:
                    if location not in resource_locations:
                        resource_locations.append(location)
        return resource_locations

    def get_resources(self):
        return [self]

    def get_input_node(self):
        return [node for node in self.child_nodes if node.node_type == NodeType.Input][0]

    def get_middle_nodes(self):
        return [
            node for node in self.child_nodes if node.node_type != NodeType.Input and node.node_type != NodeType.Output
        ]

    def get_output_node(self):
        return [node for node in self.child_nodes if node.node_type == NodeType.Output][0]

    def get_input_models(self):
        input_node = [node for node in self.child_nodes if node.node_type == NodeType.Input][0]
        input_models = []
        for connection in input_node.output_connections:
            if connection.target_node.model not in input_models:
                input_models.append(connection.target_node.model)
        return input_models

    def prepare_pipeline_input_data(self, batch_size=None, random_data=False):
        input_data = {}
        demultiply_count = (
            self.demultiply_count if self.demultiply_count is not None else self.get_input_node().demultiply_count
        )
        if demultiply_count is not None:
            number_of_batches_in_request = demultiply_count
            if demultiply_count <= 0:
                number_of_batches_in_request = (
                    self.default_demultiply_count_value
                )  # we need to set a non zero number here for data generation purpose

        if batch_size is None:
            batch_size = self.get_expected_batch_size()
        for input_model_type in self.get_input_models():
            for input_name, data in input_model_type.inputs.items():
                if batch_size is not None and data["shape"][0] == -1:
                    data["shape"][0] = batch_size

                if "dataset" in data:
                    layout = data.get("layout", None)
                    if layout is not None and ":" in layout:
                        layout_str = layout.partition(":")[0]
                    else:
                        layout_str = None
                    input_data[input_name] = data["dataset"].get_data(
                        shape=data["shape"],
                        batch_size=batch_size,
                        transpose_axes=input_model_type.transpose_axes,
                        layout=layout_str,
                    )
                    if demultiply_count is not None:
                        dumultipy_content = []
                        for i in range(number_of_batches_in_request):
                            dumultipy_content.append(input_data[input_name])
                        input_data[input_name] = np.array(dumultipy_content)
                else:
                    if demultiply_count is not None:
                        new_data = deepcopy(data["shape"])
                        new_data.insert(0, number_of_batches_in_request)
                        input_data[input_name] = np.ones(new_data, dtype=data["dtype"])
                    else:
                        input_data[input_name] = np.ones(data["shape"], dtype=data["dtype"])

        return self.map_inputs(input_data)

    def prepare_input_data(self, batch_size=None, input_key=None):
        data = self.prepare_pipeline_input_data(batch_size)
        return data

    def prepare_model_input_data(self, batch_size=None):
        return super(Pipeline, self).prepare_input_data(batch_size)

    def prepare_model_resources(self, base_location):
        return super(Pipeline, self).prepare_resources(base_location)

    def map_inputs(self, prepare_inputs: dict):
        result_dict = {}
        for key, value in self.get_pipeline_inputs_to_model_dataset_map().items():
            result_dict[key] = prepare_inputs[value]

        return result_dict

    @staticmethod
    def is_pipeline():
        return True

    def get_custom_nodes(self):
        return [node.model for node in self.child_nodes if isinstance(node.model, CustomNode)]

    def get_models(self):
        models = []
        for node in self.child_nodes:
            if node.node_type not in (NodeType.Input, NodeType.Output):
                if any(added_model.name == node.model.name for added_model in models):
                    continue

                models.append(node.model)

        return models

    def get_regular_models(self):
        return [model for model in self.get_models() if not isinstance(model, CustomNode)]

    def get_pipeline_inputs_to_model_dataset_map(self):
        inputs_to_models_map = {}
        for pipeline_input_name, model_input_name in zip(self.input_names, self.get_input_models()[0].input_names):
            inputs_to_models_map[pipeline_input_name] = model_input_name
        return inputs_to_models_map

    def config_refresh(self):
        refreshed_config = {"name": self.name}
        if self.demultiply_count is not None:
            refreshed_config["demultiply_count"] = self.demultiply_count

        refreshed_config.update({"inputs": self.input_names, "nodes": [], "outputs": []})

        nodes = self.child_nodes
        regular_nodes = [
            node for node in nodes if node.node_type != NodeType.Input and node.node_type != NodeType.Output
        ]
        for node in regular_nodes:
            refreshed_config["nodes"].append(node.dump_config())

        output_node = [node for node in nodes if node.node_type == NodeType.Output][0]
        for input_connection_of_output_node in output_node.input_connections:
            output_map = {
                "node_name": input_connection_of_output_node.source_node.name,
                "data_item": input_connection_of_output_node.get_source_data_item_name(),
            }
            output_name = input_connection_of_output_node.get_target_data_item_name()
            refreshed_config["outputs"].append({output_name: output_map})
        self.config = refreshed_config
        return self.config

    def build_pipeline_config(
        self,
        config,
        custom_nodes,
        config_custom_nodes,
        models=None,
        use_custom_graphs=False,
        mediapipe_models=None,
        use_subconfig=False,
        custom_graph_paths=None,
    ):
        config[Config.PIPELINE_CONFIG_LIST].append(self.config)
        config_custom_nodes = self.build_config_custom_nodes(custom_nodes, config_custom_nodes)
        return config, config_custom_nodes

    def build_config_custom_nodes(self, custom_nodes, config_custom_nodes):
        if custom_nodes is None:
            custom_node_list = self.get_unique_custom_node_list()
            for custom_node in custom_node_list:
                if type(custom_node) not in [type(x) for x in config_custom_nodes]:
                    config_custom_nodes.append(custom_node)
        return config_custom_nodes

    def map_model_output_to_pipeline_output(self, model_output):
        result = {}
        for node in self.child_nodes:
            if node.node_type == NodeType.Output:
                for connection in node.input_connections:
                    target_name = self.output_names[connection.target_node_input_id]
                    model = connection.source_node.model
                    source_name = model.output_names[connection.source_node_output_id]
                    result[target_name] = model_output[source_name]
                return result

        assert False, "Output node not found"

    def get_unique_custom_node_list(self):
        result = []
        for custom_node in self.get_custom_nodes():
            if type(custom_node) not in [type(x) for x in result]:
                result.append(custom_node)
        return result

    def has_custom_nodes(self):
        return len(self.get_custom_nodes()) > 0

    def change_input_name(self, old_name, new_name):
        super().change_input_name(old_name, new_name)
        self.get_input_node().change_output_name(old_name, new_name)

    def change_output_name(self, old_name, new_name):
        super().change_output_name(old_name, new_name)
        self.get_output_node().change_input_name(old_name, new_name)


class MediaPipe(Pipeline):
    name = "MediaPipe"
    is_mediapipe = True
    is_python_custom_node = False
    pbtxt_name = None

    def __init__(self, model=None, pipeline=None, demultiply_count=None, **kwargs):
        if pipeline is not None:
            pipeline = pipeline(model, demultiply_count, **kwargs)
            self.__dict__.update(pipeline.__dict__)
        self.is_mediapipe = True
        self.calculators = []
        self.graphs = []
        self.regular_models = []
        self.create_header = True

    def _initialize(self, models=None):
        self.child_nodes = []
        self.child_nodes.extend(self._create_nodes(models))
        self.initialize_inputs_outputs()
        self.graph_refresh()

    @staticmethod
    def get_mediapipe_names(config):
        return [elem["name"] for elem in config[Config.MEDIAPIPE_CONFIG_LIST]]

    def prepare_input_data(self, batch_size=None, input_key=None):
        data = self.prepare_pipeline_input_data(batch_size)
        new_data = {}
        for i, key in enumerate(list(data.keys()), start=0):
            new_input_key = input_key if input_key is not None else "input"
            new_data.update({new_input_key: data[key]})
        return new_data

    def build_pipeline_config(
        self,
        config,
        custom_nodes,
        config_custom_nodes,
        models,
        use_custom_graphs=False,
        mediapipe_models=None,
        use_subconfig=False,
        custom_graph_paths=None,
    ):
        # Mediapipe config.json example:
        # {
        #   "model_config_list": [...],
        #   "pipeline_config_list": [...],
        #   "custom_loader_config_list": [...],
        #   "mediapipe_config_list": [
        #     {
        #       "name": "pipe1",
        #       "base_path": "/models/pipe1",
        #       "graph_path": "/models/pipe1/graphdummy.pbtxt"
        #     }
        #   ]
        # }
        if self.config:
            config[Config.PIPELINE_CONFIG_LIST].append(self.config)
        config_custom_nodes = self.build_config_custom_nodes(custom_nodes, config_custom_nodes)
        if use_custom_graphs:
            config = self.prepare_custom_graphs_mediapipe_config_list(config, use_subconfig, custom_graph_paths)
        else:
            mediapipe_models = [self] if mediapipe_models is None else mediapipe_models
            config = self.add_mediapipe_graphs_to_config(config, use_subconfig, mediapipe_models)

        return config, config_custom_nodes

    def prepare_custom_graphs_mediapipe_config_list(self, config, use_subconfig=False, custom_graph_paths=None):
        # Mediapipe config.json example:
        # {
        #   "model_config_list": [...],
        #   "pipeline_config_list": [...],
        #   "custom_loader_config_list": [...],
        #   "mediapipe_config_list": [
        #     {
        #       "name": "pipe1",
        #       "base_path": "/models/pipe1",
        #       "graph_path": "/models/pipe1/graphdummy.pbtxt",
        #       "subconfig": "/models/pipe1/subconfig.json"
        #     }
        #   ]
        # }
        config[Config.MEDIAPIPE_CONFIG_LIST] = []
        mediapipe_base_path = str(Path(Paths.MODELS_PATH_INTERNAL, self.name))
        for i, calculator in enumerate(custom_graph_paths):
            proto_dict = {
                "name": self.name,
                "base_path": mediapipe_base_path,
                "graph_path": os.path.join(mediapipe_base_path, os.path.basename(calculator)),
            }
            config[Config.MEDIAPIPE_CONFIG_LIST].append(proto_dict)
            if use_subconfig and "subconfig" not in str(proto_dict):
                proto_dict["subconfig"] = os.path.join(mediapipe_base_path, Paths.SUBCONFIG_FILE_NAME)
                config[Config.MEDIAPIPE_CONFIG_LIST][i].update(proto_dict)
        return config

    def add_mediapipe_graphs_to_config(self, config, use_subconfig=False, mediapipe_models=None):
        # Mediapipe config.json example:
        # {
        #   "model_config_list": [...],
        #   "pipeline_config_list": [...],
        #   "custom_loader_config_list": [...],
        #   "mediapipe_config_list": [
        #     {
        #       "name": "pipe1",
        #       "base_path": "/models/pipe1/",
        #       "graph_path": "/models/pipe1/graphdummy.pbtxt",
        #       "subconfig": "/models/pipe1/subconfig.json"
        #     }
        #   ]
        # }

        config[Config.MEDIAPIPE_CONFIG_LIST] = (
            [] if config.get(Config.MEDIAPIPE_CONFIG_LIST) is None else config[Config.MEDIAPIPE_CONFIG_LIST]
        )
        for i, model in enumerate(mediapipe_models):
            model_name = model.name
            graph_name = model_name if (not model.is_generative and model.pbtxt_name is None) \
                else getattr(model, "pbtxt_name", None)
            graph_filename = f"{graph_name}.pbtxt"
            mediapipe_base_path = str(Path(Paths.MODELS_PATH_INTERNAL, model_name))
            graph_path = str(Path(mediapipe_base_path, graph_filename)) if not model.use_relative_paths \
                else graph_filename
            proto_dict = {
                "name": model_name,
                "base_path": mediapipe_base_path,
                "graph_path": graph_path,
            }
            if proto_dict not in config[Config.MEDIAPIPE_CONFIG_LIST]:
                config[Config.MEDIAPIPE_CONFIG_LIST].append(proto_dict)

            for regular_model in model.regular_models:
                if use_subconfig and "subconfig" not in str(proto_dict):
                    subconfig_filename = f"subconfig_{regular_model.name}.json"
                    proto_dict["subconfig"] = (
                        os.path.join(mediapipe_base_path, subconfig_filename)
                        if not model.use_relative_paths
                        else subconfig_filename
                    )
                    config[Config.MEDIAPIPE_CONFIG_LIST][i].update(proto_dict)
        return config

    def graph_refresh(self):
        nodes = []
        for child_node in self.child_nodes:
            model = child_node.model
            if getattr(child_node, "calculator", None) is not None:
                content = child_node.calculator.create_proto_content(
                    model=model,
                    input_stream=child_node.input_stream,
                    output_stream=child_node.output_stream,
                    create_header=self.create_header,
                )
                nodes.append(content)

        calculator_class = PythonCalculator if self.is_python_custom_node else MediaPipeCalculator
        header = calculator_class.create_proto_header(
            model=None,
            input_stream=self.get_input_node().output_names,
            output_stream=self.get_output_node().input_names,
        )
        full_content = header + " \n\n".join(nodes)
        self.graphs = [full_content]
