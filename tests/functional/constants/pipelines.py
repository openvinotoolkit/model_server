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

from tests.functional.config import datasets_path
from tests.functional.constants.model_dataset import RandomDataset
from tests.functional.constants.models import (
    AgeGender,
    ArgMax,
    CrnnTf,
    Dummy,
    DummyAdd2Inputs,
    DummyIncrement,
    DummyIncrementDecrement,
    EastFp32,
    Emotion,
    FaceDetectionRetail,
    GoogleNetV2Fp32,
    Increment4d,
    ModelInfo,
    Resnet,
    ResnetWrongInputShapeDim,
    ResnetWrongInputShapes,
    VehicleAttributesRecognition,
    VehicleDetection,
)
from tests.functional.constants.ovms import Config
from tests.functional.constants.paths import Paths
from tests.functional.object_model.custom_node import (
    CustomNode,
    CustomNodeAddSub,
    CustomNodeChooseMaximum,
    CustomNodeDemultiply,
    CustomNodeDemultiplyGather,
    CustomNodeDifferentOperations,
    CustomNodeDynamicDemultiplex,
    CustomNodeEastOcr,
    CustomNodeElastic1T,
    CustomNodeFaces,
    CustomNodeImageTransformation,
    CustomNodeVehicles,
)
from tests.functional.object_model.mediapipe_calculators import (
    CorruptedFileCalculator,
    MediaPipeCalculator,
    OpenVINOInferenceCalculator,
    OpenVINOModelServerSessionCalculator,
    OVMSOVCalculator,
    PythonCalculator,
)


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


class SimplePipeline(Pipeline):

    def __init__(self, model=None, demultiply_count=None, name=None, **kwargs):
        name = "single_model_pipeline" if name is None else f"single_model_pipeline_{name}"
        super().__init__(name=name, **kwargs)
        self.demultiply_count = demultiply_count

        if model is None:
            model = Resnet()

        self._initialize([model])

    def _create_nodes(self, models):
        model = models[0]
        node1 = Node("node_1", model)

        request = Node("request", node_type=NodeType.Input, output_names=["input"])
        output = Node("output", node_type=NodeType.Output, input_names=["output"])

        NodesConnection.connect(node1, 0, request, 0)
        NodesConnection.connect(output, 0, node1, 0)

        return [request, node1, output]


class MultipleInputsOutputsPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("multiple_inputs_outputs_pipeline", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        node1 = Node("node_1", DummyIncrementDecrement())

        request = Node("request", node_type=NodeType.Input)
        output = Node("output", node_type=NodeType.Output)

        NodesConnection.connect(node1, 0, request, 0)
        NodesConnection.connect(node1, 1, request, 1)
        NodesConnection.connect(output, 0, node1, 0)
        NodesConnection.connect(output, 1, node1, 1)

        nodes = [request, node1, output]
        return nodes

    def get_expected_output(self, input_data: dict, client_type: str = None):
        model_output = self.get_models()[0].get_expected_output(input_data)
        return self.map_model_output_to_pipeline_output(model_output)


class InputNotConnectedPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("single_input_not_nonnected_pipeline", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        node1 = Node("node_1", DummyIncrementDecrement())

        request = Node("request", node_type=NodeType.Input)
        output = Node("output", node_type=NodeType.Output)

        NodesConnection.connect(node1, 0, request, 0)
        NodesConnection.connect(output, 0, node1, 0)
        NodesConnection.connect(output, 1, node1, 1)

        nodes = [request, node1, output]
        return nodes


class ImageClassificationPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("image_classification_pipeline", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        resnet_node = Node("resnet_node", Resnet())
        googlenet_node = Node("googlenet_node", GoogleNetV2Fp32())
        argmax_node = Node("argmax_node", ArgMax())

        request = Node("request", node_type=NodeType.Input)
        output = Node("output", node_type=NodeType.Output)

        NodesConnection.connect(googlenet_node, 0, request, 0)
        NodesConnection.connect(resnet_node, 0, request, 0)
        NodesConnection.connect(argmax_node, 0, googlenet_node, 0)
        NodesConnection.connect(argmax_node, 1, resnet_node, 0)
        NodesConnection.connect(output, 0, argmax_node, 0)

        nodes = [request, googlenet_node, resnet_node, argmax_node, output]
        return nodes


class ComplexDummyPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("complex_pipeline", **kwargs)
        self._initialize()

    def get_pipeline_transpose_axes(self):
        return None

    def get_pipeline_datasets(self):
        return {
            "input1": os.path.join(datasets_path, RandomDataset.name),
            "input2": os.path.join(datasets_path, RandomDataset.name),
        }

    def _create_nodes(self, models=None):
        node1 = Node("node_1", DummyIncrementDecrement())
        node2 = Node("node_2", DummyIncrement())
        node3 = Node("node_3", DummyAdd2Inputs())
        node4 = Node("node_4", DummyAdd2Inputs())
        node5 = Node("node_5", DummyIncrement())
        node6 = Node("node_6", DummyIncrement())

        request = Node("request", node_type=NodeType.Input)
        output = Node("output", node_type=NodeType.Output)

        NodesConnection.connect(node1, 0, request, 0)
        NodesConnection.connect(node1, 1, request, 1)
        NodesConnection.connect(node2, 0, node1, 0)
        NodesConnection.connect(node3, 0, node1, 1)
        NodesConnection.connect(node3, 1, node6, 0)
        NodesConnection.connect(node4, 0, node1, 0)
        NodesConnection.connect(node4, 1, node2, 0)
        NodesConnection.connect(node5, 0, request, 1)
        NodesConnection.connect(node6, 0, request, 1)
        NodesConnection.connect(output, 0, node4, 0)
        NodesConnection.connect(output, 1, node3, 0)
        NodesConnection.connect(output, 2, node5, 0)
        NodesConnection.connect(output, 3, node1, 0)

        nodes = [request, node1, node2, node3, node4, node5, node6, output]
        return nodes


class SameModelsPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("multiple_versions_of_the_same_model_pipeline", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        node1 = Node("node_1", DummyIncrement())
        node2 = Node("node_2", DummyIncrement())

        request = Node("request", node_type=NodeType.Input)
        output = Node("output", node_type=NodeType.Output)

        NodesConnection.connect(node1, 0, request, 0)
        NodesConnection.connect(node2, 0, node1, 0)
        NodesConnection.connect(output, 0, node2, 0)

        nodes = [request, node1, node2, output]
        return nodes


class NodeReferringMultipleOutputsFromPreviousNodePipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("pipeline_multiple_inputs_from_the_same_model", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        node1 = Node("node_1", DummyIncrementDecrement())
        node2 = Node("node_2", DummyAdd2Inputs())

        request = Node("request", node_type=NodeType.Input)
        output = Node("output", node_type=NodeType.Output)

        NodesConnection.connect(node1, 0, request, 0)
        NodesConnection.connect(node1, 1, request, 1)
        NodesConnection.connect(node2, 0, node1, 0)
        NodesConnection.connect(node2, 1, node1, 1)
        NodesConnection.connect(output, 0, node2, 0)

        nodes = [request, node1, node2, output]
        return nodes

    def get_expected_output(self, input_data, client_type: str = None):
        models = self.get_regular_models()
        input_model = models[0]
        output_model = models[1]
        input_model_output = input_model.get_expected_output(input_data)
        node2 = {
            output_model.input_names[0]: input_model_output[input_model.output_names[0]],
            output_model.input_names[1]: input_model_output[input_model.output_names[1]],
        }
        output_model_output = output_model.get_expected_output(node2)
        pipeline_output = self.map_model_output_to_pipeline_output(output_model_output)
        return pipeline_output


class EastAndOcrPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("east_and_ocr_pipeline", **kwargs)
        self._initialize()
        self.set_expected_output_shape()

    def set_expected_output_shape(self):
        self.outputs["texts"]["shape"].insert(0, 0)

    def _create_nodes(self, models=None):
        east_node = Node("east_node", EastFp32(), output_names=["scores", "geometry"])
        extract_node = Node("extract_node", CustomNodeEastOcr(), NodeType.Custom, demultiply_count=0)
        crnn_model = CrnnTf()
        crnn_model.inputs["input"]["layout"] = "NHWC:NCHW"
        crnn_node = Node("crnn_node", crnn_model)

        request = Node("request", node_type=NodeType.Input)
        output = Node(
            "output",
            node_type=NodeType.Output,
            input_names=["text_images", "text_coordinates", "confidence_levels", "texts"],
        )
        NodesConnection.connect(east_node, 0, request, 0)
        NodesConnection.connect(extract_node, 0, request, 0)
        NodesConnection.connect(extract_node, 1, east_node, 0)
        NodesConnection.connect(extract_node, 2, east_node, 1)
        NodesConnection.connect(crnn_node, 0, extract_node, 0)
        NodesConnection.connect(output, 0, extract_node, 0)
        NodesConnection.connect(output, 1, extract_node, 1)
        NodesConnection.connect(output, 2, extract_node, 2)
        NodesConnection.connect(output, 3, crnn_node, 0)

        nodes = [request, east_node, extract_node, crnn_node, output]
        return nodes


class DemultiplyPipeline(Pipeline):

    def __init__(self, demultiply_value, **kwargs):
        super().__init__("demultiply_pipeline", **kwargs)
        self.demultiply_node = Node(
            "demultiply", CustomNodeDemultiply(demultiply_value), NodeType.Custom, demultiply_count=-1
        )
        self.resnet_node = Node("resnet", Resnet())
        self._initialize()
        self.update_demultiply_value(demultiply_value)

    def update_demultiply_value(self, new_demultiply_value):
        self.demultiply_node.model.demultiply_size = new_demultiply_value
        self.set_expected_output_shape()

    def set_expected_output_shape(self):
        self.outputs["result"]["shape"] = [self.demultiply_node.model.demultiply_size] + self.resnet_node.model.outputs[
            "softmax_tensor"
        ]["shape"]

    def _create_nodes(self, models=None):
        request = Node("request", node_type=NodeType.Input)
        output = Node("output", node_type=NodeType.Output, input_names=["result"])

        NodesConnection.connect(self.demultiply_node, 0, request, 0)
        NodesConnection.connect(self.resnet_node, 0, self.demultiply_node, 0)
        NodesConnection.connect(output, 0, self.resnet_node, 0)

        return [request, self.demultiply_node, self.resnet_node, output]


class ElasticPipeline(Pipeline):

    def __init__(self, input_shape, output_shape, demultiply_count=None, **kwargs):
        super().__init__("elastic_pipeline", **kwargs)
        self.custom_node = Node(
            "elastic_node",
            CustomNodeElastic1T(input_shape, output_shape),
            NodeType.Custom,
            demultiply_count=demultiply_count,
        )
        self._request = Node("request", node_type=NodeType.Input)
        self.model_node = Node("resnet", Resnet())
        for key in self.model_node.model.inputs:
            self.model_node.model.inputs[key]["shape"] = None
        self._initialize()
        self.set_expected_output_shape()

    def set_expected_output_shape(self):
        demultiply_value = self.custom_node.model.outputs["tensor_out"]["shape"][0]
        self.outputs["result"]["shape"] = [demultiply_value] + self.model_node.model.outputs["softmax_tensor"]["shape"]

    def _create_nodes(self, models):
        output = Node("output", node_type=NodeType.Output, input_names=["result"])

        NodesConnection.connect(self.custom_node, 0, self._request, 0)
        NodesConnection.connect(self.model_node, 0, self.custom_node, 0)
        NodesConnection.connect(output, 0, self.model_node, 0)
        return [self._request, self.custom_node, self.model_node, output]


class ElasticBatchSizePipeline(Pipeline):
    def __init__(self, node_batch_configuration_list, **kwargs):
        super().__init__(name="misconfigurated_pipeline", **kwargs)
        self.node_batch_cfg_list = node_batch_configuration_list
        self._initialize()

    def _create_nodes(self, models):
        request = Node("request", node_type=NodeType.Input, output_names=["input"])
        output = Node("output", node_type=NodeType.Output, input_names=["output"])

        nodes = [request]
        for node_name, batch_size in self.node_batch_cfg_list:
            model = Dummy(batch_size=batch_size)
            model.name = f"{model.name}_{node_name}"
            nodes.append(Node(node_name, model))
        nodes.append(output)

        for i in range(1, len(nodes)):
            NodesConnection.connect(nodes[i], 0, nodes[i - 1], 0)
        return nodes


class CustomNodesConnectedToEachOtherPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("custom_nodes_connected_to_each_other_pipeline", **kwargs)
        self.custom_node_a = Node("node_1", CustomNodeAddSub(1.5, 0.7), NodeType.Custom)
        self.custom_node_b = Node("node_2", CustomNodeAddSub(2.4, 1.2), NodeType.Custom)
        self._initialize()

    def set_expected_output_shape(self):
        arg_name = list(self.custom_node_a.model.outputs.keys())[0]
        self.outputs["output_0"]["shape"] = self.custom_node_a.model.outputs[arg_name]["shape"]

    def _create_nodes(self, models=None):
        request = Node("request", node_type=NodeType.Input)
        output = Node("output", node_type=NodeType.Output)
        nodes = [request, self.custom_node_a, self.custom_node_b, output]

        for i in range(1, len(nodes)):
            NodesConnection.connect(nodes[i], 0, nodes[i - 1], 0)
        return nodes

    def get_expected_output(self, input_data, client_type: str = None):
        custom_nodes = self.get_custom_nodes()
        node1_out = custom_nodes[0].get_expected_output(input_data)
        node2_out = custom_nodes[1].get_expected_output(node1_out)
        return self.map_model_output_to_pipeline_output(node2_out)


class CustomNodeNotAllOutputsConnectedPipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__("custom_node_not_all_outputs_connected_pipeline", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        node1 = Node("node_1", CustomNodeDifferentOperations(), NodeType.Custom)

        request = Node("request", node_type=NodeType.Input, output_names=self.input_names)
        output = Node("output", node_type=NodeType.Output, input_names=self.output_names)

        NodesConnection.connect(node1, 0, request, 0)
        NodesConnection.connect(node1, 1, request, 1)
        NodesConnection.connect(output, 0, node1, 0)

        nodes = [request, node1, output]
        return nodes


class CustomNodeNotAllInputsConnectedPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("custom_node_not_all_inputs_connected_pipeline", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        node1 = Node("node_1", CustomNodeDifferentOperations(), NodeType.Custom)

        request = Node("request", node_type=NodeType.Input, output_names=self.input_names)
        output = Node("output", node_type=NodeType.Output, input_names=self.output_names)

        NodesConnection.connect(node1, 1, request, 1)
        NodesConnection.connect(output, 0, node1, 0)
        NodesConnection.connect(output, 1, node1, 1)

        nodes = [request, node1, output]
        return nodes


class CyclicGraphPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("cyclic_graph_pipeline", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        model_two_inputs_two_outputs = DummyIncrementDecrement()
        model_two_inputs_one_output = DummyAdd2Inputs()

        node1 = Node("node_1", model_two_inputs_one_output)
        node2 = Node("node_2", model_two_inputs_two_outputs)

        request = Node("request", node_type=NodeType.Input, output_names=self.input_names)
        output = Node("output", node_type=NodeType.Output, input_names=self.output_names)

        NodesConnection.connect(node1, 0, request, 0)
        NodesConnection.connect(node1, 1, node2, 1)
        NodesConnection.connect(output, 0, node2, 0)

        nodes = [request, node1, node2, output]
        return nodes


class AgeGenderAndEmotionPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("combined-recognition", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        model_gender = AgeGender()
        model_gender.set_input_shape_for_ovms([1, 3, 64, 64])

        model_emotion = Emotion()

        age_gender_node = Node("age_gender", model_gender, output_names=["age", "gender"])
        emotion_node = Node("emotion_node", model_emotion, output_names=["emotion"])

        request = Node("request", node_type=NodeType.Input, output_names=["image"])
        output = Node("output", node_type=NodeType.Output, input_names=["age", "gender", "emotion"])

        NodesConnection.connect(age_gender_node, 0, request, 0)
        NodesConnection.connect(emotion_node, 0, request, 0)

        NodesConnection.connect(output, 0, age_gender_node, 0)
        NodesConnection.connect(output, 1, age_gender_node, 1)
        NodesConnection.connect(output, 2, emotion_node, 0)

        nodes = [request, age_gender_node, emotion_node, output]
        return nodes


class VehiclesAnalysisPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("multiple_vehicle_recognition", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        detection_model = VehicleDetection()
        recognition_model = VehicleAttributesRecognition()
        vehicles_custom_node = CustomNodeVehicles()

        vehicle_detection_node = Node("vehicle_detection_node", detection_model, output_names=["detection_out"])
        extract_node = Node(
            "extract_node",
            vehicles_custom_node,
            NodeType.Custom,
            demultiply_count=0,
            output_names=["vehicle_images", "vehicle_coordinates", "confidence_levels"],
        )
        vehicle_recognition_node = Node("vehicle_recognition_node", recognition_model, output_names=["color", "type"])

        request = Node("request", node_type=NodeType.Input, output_names=["image"])
        output = Node(
            "output",
            node_type=NodeType.Output,
            input_names=["vehicle_images", "vehicle_coordinates", "confidence_levels", "colors", "types"],
        )

        NodesConnection.connect(vehicle_detection_node, 0, request, 0)

        NodesConnection.connect(extract_node, 0, request, 0)
        NodesConnection.connect(extract_node, 1, vehicle_detection_node, 0)

        NodesConnection.connect(vehicle_recognition_node, 0, extract_node, 0)

        NodesConnection.connect(output, 0, extract_node, 0)
        NodesConnection.connect(output, 1, extract_node, 1)
        NodesConnection.connect(output, 2, extract_node, 2)

        NodesConnection.connect(output, 3, vehicle_recognition_node, 0)
        NodesConnection.connect(output, 4, vehicle_recognition_node, 1)

        nodes = [request, vehicle_detection_node, extract_node, vehicle_recognition_node, output]
        return nodes


class FacesAnalysisPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("find_face_images", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        model_face = FaceDetectionRetail()
        model_face.input_shapes = [1, 3, 400, 600]
        model_face.set_input_shape_for_ovms([1, 3, 400, 600])

        model_gender = AgeGender()
        model_gender.input_shapes = [1, 3, 64, 64]
        model_gender.set_input_shape_for_ovms([1, 3, 64, 64])

        model_emotion = Emotion()
        model_emotion.input_shapes = [1, 3, 64, 64]
        model_emotion.set_input_shape_for_ovms([1, 3, 64, 64])

        faces_custom_node = CustomNodeFaces()

        model_gender.inputs["data"]["shape"] = deepcopy(model_emotion.inputs["data"]["shape"])
        model_face.inputs["data"]["shape"] = [1, 3, 400, 600]

        face_detection_node = Node("face_detection_node", model_face, output_names=["detection_out"])
        extract_node = Node(
            "extract_node",
            faces_custom_node,
            NodeType.Custom,
            demultiply_count=0,
            output_names=["face_images", "face_coordinates", "confidence_levels"],
        )
        age_gender_recognition_node = Node("age_gender_recognition_node", model_gender, output_names=["age", "gender"])
        emotion_recognition_node = Node("emotion_recognition_node", model_emotion, output_names=["emotion"])

        request = Node("request", node_type=NodeType.Input, output_names=["image"])
        output = Node(
            "output",
            node_type=NodeType.Output,
            input_names=["face_images", "face_coordinates", "confidence_levels", "ages", "genders", "emotions"],
        )

        NodesConnection.connect(face_detection_node, 0, request, 0)

        NodesConnection.connect(extract_node, 0, request, 0)
        NodesConnection.connect(extract_node, 1, face_detection_node, 0)

        NodesConnection.connect(age_gender_recognition_node, 0, extract_node, 0)

        NodesConnection.connect(emotion_recognition_node, 0, extract_node, 0)

        NodesConnection.connect(output, 0, extract_node, 0)
        NodesConnection.connect(output, 1, extract_node, 1)
        NodesConnection.connect(output, 2, extract_node, 2)

        NodesConnection.connect(output, 3, age_gender_recognition_node, 0)
        NodesConnection.connect(output, 4, age_gender_recognition_node, 1)
        NodesConnection.connect(output, 5, emotion_recognition_node, 0)

        nodes = [
            request,
            face_detection_node,
            extract_node,
            age_gender_recognition_node,
            emotion_recognition_node,
            output,
        ]
        return nodes


class TenDummySerialPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("ten_dummy_serial", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        dummy = Dummy()
        dummy.inputs["b"]["shape"] = [1, 150528]

        node1 = Node("node_1", dummy)
        node2 = Node("node_2", dummy)
        node3 = Node("node_3", dummy)
        node4 = Node("node_4", dummy)
        node5 = Node("node_5", dummy)
        node6 = Node("node_6", dummy)
        node7 = Node("node_7", dummy)
        node8 = Node("node_8", dummy)
        node9 = Node("node_9", dummy)
        node10 = Node("node_10", dummy)

        request = Node("request", node_type=NodeType.Input)
        output = Node("output", node_type=NodeType.Output)

        NodesConnection.connect(node1, 0, request, 0)
        NodesConnection.connect(node2, 0, node1, 0)
        NodesConnection.connect(node3, 0, node2, 0)
        NodesConnection.connect(node4, 0, node3, 0)
        NodesConnection.connect(node5, 0, node4, 0)
        NodesConnection.connect(node6, 0, node5, 0)
        NodesConnection.connect(node7, 0, node6, 0)
        NodesConnection.connect(node8, 0, node7, 0)
        NodesConnection.connect(node9, 0, node8, 0)
        NodesConnection.connect(node10, 0, node9, 0)
        NodesConnection.connect(output, 0, node10, 0)

        nodes = [request, node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, output]
        return nodes


class DummyDiffOpsMaxPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("dummy_diff_ops_max_pipeline", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        request = Node("request", node_type=NodeType.Input, output_names=self.input_names)
        dummy = Dummy()
        node_diff = Node("node_diff", CustomNodeDifferentOperations(), NodeType.Custom, demultiply_count=4)
        node_dummy_1 = Node("node_d1", dummy)
        node_max = Node("node_max", CustomNodeChooseMaximum(), NodeType.Custom, gather_from_node="node_diff")
        node_max.model.selection_criteria = CustomNodeChooseMaximum.Method.MAXIMUM_MAXIMUM
        node_dummy_2 = Node("node_d2", dummy, output_names=self.input_names)
        output = Node("output", node_type=NodeType.Output, input_names=self.output_names)

        NodesConnection.connect(node_diff, 0, request, 0)
        NodesConnection.connect(node_diff, 1, request, 1)
        NodesConnection.connect(node_dummy_1, 0, node_diff, 0)
        NodesConnection.connect(node_max, 0, node_dummy_1, 0)
        NodesConnection.connect(node_dummy_2, 0, node_max, 0)
        NodesConnection.connect(output, 0, node_dummy_2, 0)
        nodes = [request, node_dummy_1, node_dummy_2, node_max, node_diff, output]
        return nodes


class DummyDynamicDemuxPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("dummy_dag_pipeline", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        self.demultiply_count = -1

        request = Node("request", node_type=NodeType.Input, output_names=self.input_names)
        output = Node("output", node_type=NodeType.Output, input_names=self.output_names)

        dummy = Dummy()
        node_dummy = Node("node_d", dummy)

        NodesConnection.connect(node_dummy, 0, request, 0)
        NodesConnection.connect(output, 0, node_dummy, 0)
        nodes = [request, node_dummy, output]

        return nodes


class DifferentDemultiplyValuesPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("dynamic_demultiplex_pipeline", **kwargs)
        self.demux_node = Node(
            "diff_node", CustomNodeDynamicDemultiplex(), node_type=NodeType.Custom, demultiply_count=-1
        )
        self._initialize()
        self.set_expected_output_shape()

    def set_expected_output_shape(self):
        self.outputs["output_0"]["shape"] = self.demux_node.model.outputs["dynamic_demultiplex_results"]["shape"]

    def _create_nodes(self, models=None):
        request = Node("request", node_type=NodeType.Input, input_names=self.input_names)
        dummy_node = Node("dummy_node", Dummy())
        output = Node("output", node_type=NodeType.Output, output_names=self.output_names)

        NodesConnection.connect(self.demux_node, 0, request, 0)
        NodesConnection.connect(dummy_node, 0, self.demux_node, 0)
        NodesConnection.connect(output, 0, dummy_node, 0)

        nodes = [request, self.demux_node, dummy_node, output]
        return nodes


class DemultiplexerAndGatherPipeline(Pipeline):

    def __init__(self, demultiply_count, **kwargs):
        super().__init__("pipeline_with_demultiplexer_and_gather", **kwargs)
        self._demultiply_count = demultiply_count
        self._initialize()

    def _create_nodes(self, models=None):
        diff_node = Node(
            "diff_node", CustomNodeDifferentOperations(), NodeType.Custom, demultiply_count=self._demultiply_count
        )
        dummy_node = Node("dummy_node", Dummy())
        demultiply_gather_node = Node(
            "demultiply_gather_node",
            CustomNodeDemultiplyGather(),
            NodeType.Custom,
            demultiply_count=self._demultiply_count,
            gather_from_node="diff_node",
        )

        request = Node("request", node_type=NodeType.Input, output_names=self.input_names)
        output = Node("output", node_type=NodeType.Output, input_names=self.output_names)

        NodesConnection.connect(diff_node, 0, request, 0)
        NodesConnection.connect(diff_node, 1, request, 1)
        NodesConnection.connect(dummy_node, 0, diff_node, 0)
        NodesConnection.connect(demultiply_gather_node, 0, dummy_node, 0)
        NodesConnection.connect(output, 0, demultiply_gather_node, 0)

        nodes = [request, dummy_node, demultiply_gather_node, diff_node, output]
        return nodes


class ImageTransformationPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__("image_transformation_test", **kwargs)
        self._initialize()

    def _create_nodes(self, models=None):
        self.demultiply_count = 0
        image_transformation_node = Node(
            "image_transformation_node", CustomNodeImageTransformation(), NodeType.Custom, output_names=["image"]
        )
        resnet_node = Node("resnet_node", Resnet())

        request = Node("request", node_type=NodeType.Input, output_names=["image"])
        output = Node("output", node_type=NodeType.Output, input_names=["image_0", "image_1"])

        NodesConnection.connect(image_transformation_node, 0, request, 0)
        NodesConnection.connect(resnet_node, 0, image_transformation_node, 0)
        NodesConnection.connect(output, 0, resnet_node, 0)

        NodesConnection.connect(output, 1, image_transformation_node, 0)

        nodes = [request, image_transformation_node, resnet_node, output]
        return nodes


class SingleLevelPipeline(Pipeline):
    def __init__(self, list_of_models, predict_shape, **kwargs):
        super().__init__("single_level_pipeline", **kwargs)
        self.predict_shape = predict_shape
        self.models = list_of_models
        self._initialize()

    def _create_nodes(self, model=None):
        names = [f"img_{i}" for i in range(len(self.models))]
        request = Node("request", node_type=NodeType.Input, output_names=["img"])
        output = Node("output", node_type=NodeType.Output, input_names=names)
        model_nodes = []
        for idx, model in enumerate(self.models):
            model_node = Node(f"model_{idx}", model)
            NodesConnection.connect(model_node, 0, request, 0)
            NodesConnection.connect(output, idx, model_node, 0)
            model_nodes.append(model_node)
        return [request] + model_nodes + [output]

    def prepare_input_data(self, batch_size=None, random_data=False, input_key=None):
        result = {}
        for in_name, in_data in self.models[0].inputs.items():
            shape = self.predict_shape.copy()
            if batch_size is not None:
                shape[0] = batch_size
            result[in_name] = np.ones(shape, dtype=in_data["dtype"])

        return self.map_inputs(result)


class MultiLevelPipeline(Pipeline):
    def __init__(self, shape_model_list, **kwargs):
        super().__init__("multi_level_pipeline", **kwargs)
        self._vertical_shape_list = shape_model_list
        self._initialize()

    def _create_nodes(self, model=None):
        request = Node("request", node_type=NodeType.Input, output_names=["img"])

        model_nodes = []
        for idx, shape in enumerate(self._vertical_shape_list):
            model = Increment4d()
            model.name = f"{model.name}_{idx}"
            model.update_shapes(shape)
            model.set_input_shape_for_ovms(shape)
            model_nodes.append(Node(f"model_{idx}", model))

        output = Node("output", node_type=NodeType.Output, input_names=[model_nodes[-1].name])
        node_list = [request] + model_nodes + [output]

        for i in range(1, len(node_list)):
            NodesConnection.connect(node_list[i], 0, node_list[i - 1], 0)

        return node_list


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
            graph_name = model_name if (not model.is_llm and model.pbtxt_name is None) \
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


class SimpleMediaPipe(MediaPipe):
    def __init__(self, model=None, demultiply_count=None, **kwargs):
        pipeline = SimplePipeline
        if model is None:
            model = Resnet()
        super().__init__(model, pipeline, demultiply_count, **kwargs)
        self.calculators = [OpenVINOModelServerSessionCalculator(model=self), OpenVINOInferenceCalculator(model=self)]
        self._initialize([model])
        self.regular_models = self.get_regular_models()
        assert not self.name.endswith("_mediapipe")
        if kwargs.get("name") is None:
            self.name += "_mediapipe"
        else:
            self.name = kwargs.get("name")

    def _create_nodes(self, models=None):
        session_calculator = self.calculators[0]
        inference_calculator = self.calculators[1]

        model = models[0]
        session_node = MediaPipeGraphNode("node1", model, calculator=session_calculator)
        inference_node = MediaPipeGraphNode(
            "node2", model, calculator=inference_calculator, input_stream="input", output_stream="output"
        )

        request = MediaPipeGraphNode("request", node_type=NodeType.Input, output_names=["input"])
        output = MediaPipeGraphNode("output", node_type=NodeType.Output, input_names=["output"])

        NodesConnection.connect(session_node, 0, inference_node, 0)
        NodesConnection.connect(inference_node, 0, request, 0)
        NodesConnection.connect(output, 0, inference_node, 0)

        return [request, session_node, inference_node, output]


class FailedToLoadModelMediaPipe(SimpleMediaPipe):
    def __init__(self, model=None, demultiply_count=None, **kwargs):
        super().__init__(model, demultiply_count, **kwargs)
        self.regular_models = self.get_regular_models()
        side_feed_calculator = OpenVINOInferenceCalculator()
        api_session_calculator = OpenVINOModelServerSessionCalculator(
            model=self, session=MediaPipeCalculator.get_valid_model_name(self.regular_models[0]), model_name=self.name
        )
        self.calculators = [side_feed_calculator, api_session_calculator]


class ImageClassificationMediaPipe(MediaPipe):
    def __init__(self, **kwargs):
        Pipeline.__init__(self, "image_classification_pipeline", **kwargs)
        super().__init__()
        self._initialize()
        self.regular_models = self.get_regular_models()

    def _create_nodes(self, models=None):
        session_calculator = OpenVINOModelServerSessionCalculator()
        inference_calculator = OpenVINOInferenceCalculator()

        googlenet_session_node = MediaPipeGraphNode(
            "googlenet_session_node", GoogleNetV2Fp32(), calculator=session_calculator
        )
        resnet_session_node = MediaPipeGraphNode("resnet_session_node", Resnet(), calculator=session_calculator)
        argmax_session_node = MediaPipeGraphNode("argmax_session_node", ArgMax(), calculator=session_calculator)

        googlenet_inference_node = MediaPipeGraphNode(
            "googlenet_inference_node",
            GoogleNetV2Fp32(),
            calculator=inference_calculator,
            input_stream="GOOGLE_INPUT:input_0",
            output_stream="GOOGLE_OUTPUT:google_output",
        )
        resnet_inference_node = MediaPipeGraphNode(
            "resnet_inference_node",
            Resnet(),
            calculator=inference_calculator,
            input_stream="RESNET_INPUT:input_0",
            output_stream="RESNET_OUTPUT:resnet_output",
        )
        argmax_inference_node = MediaPipeGraphNode(
            "argmax_inference_node",
            ArgMax(),
            calculator=inference_calculator,
            input_stream=["ARGMAX_INPUT1:google_output", "ARGMAX_INPUT2:resnet_output"],
            output_stream="ARGMAX_OUTPUT:argmax_0",
        )

        request = MediaPipeGraphNode("request", node_type=NodeType.Input, output_names=["input_0"])
        output = MediaPipeGraphNode("output", node_type=NodeType.Output, input_names=["argmax_0"])

        NodesConnection.connect(googlenet_session_node, 0, googlenet_inference_node, 0)
        NodesConnection.connect(resnet_session_node, 0, resnet_inference_node, 0)
        NodesConnection.connect(argmax_session_node, 0, argmax_inference_node, 0)

        NodesConnection.connect(googlenet_inference_node, 0, request, 0)
        NodesConnection.connect(resnet_inference_node, 0, request, 0)
        NodesConnection.connect(argmax_inference_node, 0, googlenet_inference_node, 0)
        NodesConnection.connect(argmax_inference_node, 1, resnet_inference_node, 0)
        NodesConnection.connect(output, 0, argmax_inference_node, 0)

        nodes = [
            request,
            googlenet_session_node,
            resnet_session_node,
            argmax_session_node,
            googlenet_inference_node,
            resnet_inference_node,
            argmax_inference_node,
            output,
        ]
        return nodes


class SimpleModelMediaPipe(MediaPipe):
    def __init__(self, model=None, use_mapping=False, batch_size=None, single_mediapipe_model_mode=False,
                 pbtxt_name=None):
        model = Resnet(batch_size=batch_size) if model is None else model
        self.__dict__.update(model.__dict__)
        super().__init__(model)
        self.calculators = [OpenVINOInferenceCalculator(), OpenVINOModelServerSessionCalculator()]
        self.regular_models = model.get_regular_models()
        assert not self.name.endswith("_mediapipe")
        self.name += "_mediapipe"
        self.pbtxt_name = pbtxt_name
        self.single_mediapipe_model_mode = single_mediapipe_model_mode
        if self.single_mediapipe_model_mode:
            self.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, self.name)

    def replace_input_data_names(self, input_data, input_key=None):
        new_data = {}
        for i, key in enumerate(list(input_data.keys()), start=0):
            new_input_key = input_key if input_key is not None else f"in_{i}"
            new_data.update({new_input_key: input_data[key]})
        return new_data

    def prepare_input_data(self, batch_size=None, input_key=None):
        data = super().prepare_model_input_data(batch_size)
        new_data = self.replace_input_data_names(data, input_key)
        return new_data

    def prepare_input_data_from_model_datasets(self, batch_size=None, input_key=None):
        result = ModelInfo.prepare_input_data_from_model_datasets(self, batch_size)
        new_data = self.replace_input_data_names(result, input_key)
        return new_data

    def get_expected_model_output_data(self):
        expected_model_output_data = {}
        for i, key in enumerate(list(self.outputs.keys()), start=0):
            new_output_key = f"out_{i}"
            expected_model_output_data.update({new_output_key: self.outputs[key]})
        return expected_model_output_data

    def validate_outputs(self, outputs, expected_output_shapes=None, provided_input=None):
        assert outputs, "Prediction returned no output"
        if expected_output_shapes is None:
            expected_output_shapes = list(self.output_shapes.values())
        for i, shape in enumerate(expected_output_shapes):  # Check for dynamic shape
            for j, val in enumerate(shape):
                if val == -1:
                    expected_output_shapes[i][j] = 1

        expected_outputs = self.get_expected_model_output_data()
        for output_name in expected_outputs:
            assert (
                output_name in outputs
            ), f"Incorrect output name, expected: {output_name}, found: {', '.join(outputs.keys())}"
        output_shapes = [list(o.shape) for o in outputs.values()]
        assert any(
            shape in expected_output_shapes for shape in output_shapes
        ), f"Incorrect output shape, expected: {expected_output_shapes}, found: {output_shapes}."

    @staticmethod
    def is_pipeline():
        return False

    def get_models(self):
        return [self]

    def prepare_resources(self, base_location):
        return super().prepare_model_resources(base_location)

    def get_demultiply_count(self):
        return None


class SimpleDynamicModelMediaPipe(SimpleModelMediaPipe):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        self._initialize(models=[model])

    def _create_nodes(self, models=None):
        nodes = []
        if models is not None:
            assert len(models) == 1, f"Currently only single model in {self.__class__.__name__} is supported."
            model = models[0]
            session_calculator = OpenVINOModelServerSessionCalculator()
            inference_calculator = OpenVINOInferenceCalculator()
            valid_model_name = MediaPipeCalculator.get_upper_model_name(model)
            dummy_node = MediaPipeGraphNode(model.name, model, calculator=session_calculator)
            dummy_inference_node = MediaPipeGraphNode(
                f"{model.name}_inference_node",
                model,
                calculator=inference_calculator,
                input_stream=f"{valid_model_name}:in_0",
                output_stream=f"{valid_model_name}:out_0",
            )

            request = MediaPipeGraphNode("request", node_type=NodeType.Input, output_names=[f"{valid_model_name}:in_0"])
            output = MediaPipeGraphNode("output", node_type=NodeType.Output, input_names=[f"{valid_model_name}:out_0"])

            NodesConnection.connect(dummy_node, 0, dummy_inference_node, 0)

            nodes = [request, dummy_node, dummy_inference_node, output]
        return nodes


class SimpleModelMediaPipeResnetWrongInputShapes(SimpleModelMediaPipe):
    def __init__(self, model=None, use_mapping=False, batch_size=None):
        model = ResnetWrongInputShapes()
        super().__init__(model, use_mapping, batch_size)


class SimpleModelMediaPipeResnetWrongInputShapeDim(SimpleModelMediaPipe):
    def __init__(self, model=None, use_mapping=False, batch_size=None):
        model = ResnetWrongInputShapeDim()
        super().__init__(model, use_mapping, batch_size)


class CorruptedFileModelMediaPipe(SimpleModelMediaPipe):
    def __init__(self, model=None):
        super().__init__(model)
        self.calculators = [CorruptedFileCalculator()]


class SimpleOneCalculatorMediaPipe(SimpleModelMediaPipe):
    def __init__(self, model=None):
        super().__init__(model)
        self.calculators = [OVMSOVCalculator()]
        assert not self.name.endswith("_mediapipe")
        self.name += "_mediapipe"


class SameModelsMediaPipe(MediaPipe):
    def __init__(self, **kwargs):
        Pipeline.__init__(self, "same_models_mediapipe", **kwargs)
        super().__init__()
        self._initialize()
        self.regular_models = self.get_regular_models()

    def _create_nodes(self, models=None):
        model = Dummy()
        dummy_session_name = f"{model.name}_session"
        session1_calculator = OpenVINOModelServerSessionCalculator(session=dummy_session_name)
        inference1_calculator = OpenVINOInferenceCalculator(session=dummy_session_name)
        inference2_calculator = OpenVINOInferenceCalculator(session=dummy_session_name)

        dummy_session_node = MediaPipeGraphNode("dummy_session_node", model, calculator=session1_calculator)

        dummy1_inference_node = MediaPipeGraphNode(
            "dummy1_inference_node",
            model,
            calculator=inference1_calculator,
            input_stream="DUMMY1_INPUT:input",
            output_stream="DUMMY1_OUTPUT:dummy1_output",
        )
        dummy2_inference_node = MediaPipeGraphNode(
            "dummy2_inference_node",
            model,
            calculator=inference2_calculator,
            input_stream="DUMMY2_INPUT:dummy1_output",
            output_stream="DUMMY2_OUTPUT:output",
        )

        request = MediaPipeGraphNode("request", node_type=NodeType.Input, output_names=["input"])
        output = MediaPipeGraphNode("output", node_type=NodeType.Output, input_names=["output"])

        NodesConnection.connect(dummy_session_node, 0, dummy1_inference_node, 0)
        NodesConnection.connect(dummy_session_node, 0, dummy2_inference_node, 0)

        NodesConnection.connect(dummy1_inference_node, 0, request, 0)
        NodesConnection.connect(dummy2_inference_node, 0, dummy1_inference_node, 0)
        NodesConnection.connect(output, 0, dummy2_inference_node, 0)

        nodes = [request, dummy_session_node, dummy1_inference_node, dummy2_inference_node, output]
        return nodes


class ModelsChainMediaPipe(MediaPipe):
    def __init__(self, models=None, demultiply_count=None, **kwargs):
        Pipeline.__init__(self, "models_chain_mediapipe", **kwargs)
        super().__init__()
        if models is None:
            models = [Dummy()]
        self.chain_length = len(models)
        self.demultiply_count = demultiply_count
        self._initialize(models)
        self.regular_models = self.get_regular_models()

    def _create_nodes(self, models=None):
        model = models[0]
        inference_nodes = []

        model_name = MediaPipeCalculator.get_valid_model_name(model)
        final_input_name = "input"
        final_output_name = "output"
        session_calculator = OpenVINOModelServerSessionCalculator(session=model_name)
        session_node = MediaPipeGraphNode(f"{model_name}_session_node", model, calculator=session_calculator)
        request = MediaPipeGraphNode("request", node_type=NodeType.Input, output_names=[final_input_name])
        output = MediaPipeGraphNode("output", node_type=NodeType.Output, input_names=[final_output_name])
        inference_calculators = [OpenVINOInferenceCalculator(session=model_name) for i in range(self.chain_length)]

        for i, inf_calc in enumerate(inference_calculators):
            output_stream = f"{model_name}_{i}_output"
            inf_node_name = f"{model_name}_{i}_inference_node"
            if i == 0:
                inf_node = MediaPipeGraphNode(
                    inf_node_name,
                    model,
                    calculator=inf_calc,
                    input_stream=final_input_name,
                    output_stream=output_stream,
                )
            elif i == (len(inference_calculators) - 1):
                inf_node = MediaPipeGraphNode(
                    inf_node_name,
                    model,
                    calculator=inf_calc,
                    input_stream=inference_nodes[i - 1].output_stream,
                    output_stream=final_output_name,
                )
            elif 0 < i < len(inference_calculators):
                inf_node = MediaPipeGraphNode(
                    inf_node_name,
                    model,
                    calculator=inf_calc,
                    input_stream=inference_nodes[i - 1].output_stream,
                    output_stream=output_stream,
                )
            inference_nodes.append(inf_node)

        for i, inf_node in enumerate(inference_nodes):
            NodesConnection.connect(session_node, 0, inf_node, 0)
            if i == 0:
                NodesConnection.connect(inf_node, 0, request, 0)
            elif i == (len(inference_nodes) - 1):
                NodesConnection.connect(output, 0, inference_nodes[i - 1], 0)
            elif 0 < i < len(inference_nodes):
                NodesConnection.connect(inf_node, 0, inference_nodes[i - 1], 0)

        nodes = [request, output, session_node] + inference_nodes
        return nodes
