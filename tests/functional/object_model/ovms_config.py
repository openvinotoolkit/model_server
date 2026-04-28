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
from pathlib import Path
from typing import List

from tests.functional.utils.logger import get_logger
from tests.functional.constants.os_type import OsType
from tests.functional.config import enable_plugin_config_target_device
from tests.functional.constants.custom_loader import CustomLoaderConsts
from tests.functional.constants.models import ModelInfo
from tests.functional.constants.ovms import Config, CurrentOvmsType, Ovms, set_plugin_config_boolean_value
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.paths import Paths
from tests.functional.constants.pipelines import Pipeline
from tests.functional.object_model.custom_loader import CustomLoader
from tests.functional.object_model.custom_node import CustomNode
from tests.functional.object_model.ovms_params import MetricsPolicy, OvmsParams
from tests.functional.object_model.test_environment import TestEnvironment
from tests.functional.utils.remote_test_environment import copy_custom_lib_to_host

logger = get_logger(__name__)


class OvmsConfig(object):

    @staticmethod
    def generate(name, models, **kwargs):
        params = OvmsParams(models=models)
        return OvmsConfig.generate_from_parameters(name, params, **kwargs)

    @staticmethod
    def generate_from_parameters(name, parameters, resource_dir=None):
        models = parameters.models
        pipelines = []
        regular_models = []
        regular_models_used_in_pipelines = []
        for model in models or []:
            if isinstance(model, Pipeline) and model.is_pipeline():
                pipelines.append(model)
                regular_models_used_in_pipelines.extend(model.get_regular_models())
            else:
                regular_models.append(model)

        for model in regular_models_used_in_pipelines:
            model_already_exists = False
            for regular_model in regular_models:
                if regular_model.name == model.name:
                    model_already_exists = True
            if not model_already_exists:
                regular_models.append(model)

        config_dict = OvmsConfig.build(
            models=regular_models,
            pipelines=pipelines,
            resource_dir=resource_dir,
            use_subconfig=parameters.use_subconfig,
        )

        if parameters.metrics_enable in [MetricsPolicy.EnabledInConfig, MetricsPolicy.Disabled,
                                         MetricsPolicy.EnabledMetricsList]:
            enable = True if parameters.metrics_enable == MetricsPolicy.EnabledInConfig else False
            config_dict[Config.MONITORING] = {"metrics": {"enable": enable}}

        if parameters.metrics_list is not None:
            metrics = config_dict[Config.MONITORING].get("metrics", {})
            metrics["metrics_list"] = parameters.metrics_list

        if enable_plugin_config_target_device:
            plugin_config_target_device = Ovms.PLUGIN_CONFIG[parameters.target_device]
            if plugin_config_target_device:
                for model_config in config_dict[Config.MODEL_CONFIG_LIST]:
                    plugin_config = model_config[Config.CONFIG].get(Config.PLUGIN_CONFIG, None)
                    if plugin_config is not None:
                        model_config[Config.CONFIG][Config.PLUGIN_CONFIG] = {
                            **plugin_config,
                            **plugin_config_target_device,
                        }
                    else:
                        model_config[Config.CONFIG][Config.PLUGIN_CONFIG] = plugin_config_target_device
        return OvmsConfig.save(name, config_dict), config_dict

    @staticmethod
    def save(name, config_dict: dict, config_path: str = None):
        config_json = json.dumps(config_dict, indent=2)
        config_json = set_plugin_config_boolean_value(config_json, config_file=True)
        config_path = (
            os.path.join(TestEnvironment.current.base_dir, name, Paths.MODELS_PATH_NAME, Paths.CONFIG_FILE_NAME)
            if config_path is None
            else config_path
        )
        logger.info("Saving config file to {}, content:\n{}".format(config_path, config_json))

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(os.path.join(config_path), "w") as outfile:
            outfile.write(config_json)

        return Paths.CONFIG_PATH_INTERNAL

    @staticmethod
    def save_without_encoding(config_path, config_dict: dict):
        logger.info("Saving config file to {}, content:\n{}".format(config_path, str(config_dict)))
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(os.path.join(config_path), "w") as outfile:
            outfile.write(str(config_dict))

        return Paths.CONFIG_PATH_INTERNAL

    @staticmethod
    def build(
        models: List[ModelInfo] = [],
        pipelines: List[Pipeline] = None,
        custom_nodes: List[CustomNode] = None,
        metrics_enable=MetricsPolicy.NotDefined,
        resource_dir=None,
        mediapipe_models=None,
        use_custom_graphs=False,
        use_subconfig=False,
        custom_graph_paths=None,
    ) -> dict:
        config = OvmsConfig.build_ovms_config(
            models,
            pipelines,
            custom_nodes,
            metrics_enable,
            resource_dir,
            mediapipe_models,
            use_custom_graphs,
            use_subconfig,
            custom_graph_paths,
        )
        return config

    @staticmethod
    def build_ovms_config(
        models: List[ModelInfo] = [],
        pipelines: List[Pipeline] = None,
        custom_nodes: List[CustomNode] = None,
        metrics_enable=MetricsPolicy.NotDefined,
        resource_dir=None,
        mediapipe_models=None,
        use_custom_graphs=False,
        use_subconfig=False,
        custom_graph_paths=None,
    ) -> dict:
        if use_subconfig:
            config = {Config.MODEL_CONFIG_LIST: []}
        else:
            config = {Config.MODEL_CONFIG_LIST: [model.get_config() for model in models]}
            if all([c is None for c in config[Config.MODEL_CONFIG_LIST]]):
                config = {Config.MODEL_CONFIG_LIST: []}
        if resource_dir and CurrentOvmsType.ovms_type in [
            OvmsType.CAPI,
            OvmsType.BINARY,
        ]:  # Add `resource_dir` prefix to default `base_path`
            for model in config[Config.MODEL_CONFIG_LIST]:
                if not model["config"]["base_path"].startswith(resource_dir):
                    model_config_base_path = model["config"]["base_path"]
                    model["config"]["base_path"] = os.path.join(
                        resource_dir,
                        *model_config_base_path.split(os.path.sep)[1:]
                    )
        config_custom_nodes = []
        if pipelines is not None:
            config[Config.PIPELINE_CONFIG_LIST] = []
            for pipeline in pipelines:
                config, config_custom_nodes = pipeline.build_pipeline_config(
                    config,
                    custom_nodes,
                    config_custom_nodes,
                    models,
                    use_custom_graphs,
                    mediapipe_models,
                    use_subconfig,
                    custom_graph_paths,
                )

        if config_custom_nodes:
            config[Config.CUSTOM_NODE_LIBRARY_CONFIG_LIST] = [
                custom_node.get_config() for custom_node in config_custom_nodes
            ]
            for model in config[Config.CUSTOM_NODE_LIBRARY_CONFIG_LIST]:
                if (
                    CurrentOvmsType.ovms_type in [OvmsType.CAPI, OvmsType.BINARY]
                    and resource_dir
                    and CurrentOvmsType.ovms_type in [OvmsType.CAPI, OvmsType.BINARY]
                    and not model["base_path"].startswith(resource_dir)
                ):
                    model["base_path"] = os.path.join(resource_dir, f'./{model["base_path"]}')

        loader_configs = set([model.custom_loader.loader_config for model in models if model.custom_loader is not None])
        for loader in loader_configs:
            if (
                resource_dir
                and CurrentOvmsType.ovms_type in [OvmsType.CAPI, OvmsType.BINARY]
                and not loader["config"]["library_path"].startswith(resource_dir)
            ):
                loader["config"]["library_path"] = os.path.join(resource_dir, f"./{loader['config']['library_path']}")
        config[CustomLoader.PARENT_KEY] = list(loader_configs)

        if metrics_enable in [MetricsPolicy.EnabledInConfig, MetricsPolicy.Disabled]:
            enabled = True if metrics_enable == MetricsPolicy.EnabledInConfig else False
            config[Config.MONITORING] = {"metrics": {"enable": enabled}}

        # If MediaPipe pipelines (e.g. SimpleMediaPipe, ImageClassificationMediaPipe) are not defined, we use SimpleModelMediaPipe models
        mediapipe_models = [model for model in models if model.is_mediapipe]
        if not use_custom_graphs and len(mediapipe_models) > 0:
            # Scenario for any model without specifying custom_graph_paths (e.g. test_mediapipe_various_models)
            config = mediapipe_models[0].add_mediapipe_graphs_to_config(config, use_subconfig, mediapipe_models)

        if use_custom_graphs:
            for model in mediapipe_models:
                # Scenario with any graph path (custom_graph_paths) for model (e.g. test_mediapipe_dummy_basic__)
                config = model.prepare_custom_graphs_mediapipe_config_list(config, use_subconfig, custom_graph_paths)

        return config

    @staticmethod
    def load(config_path):
        with open(config_path, "r") as f:
            try:
                config_json = f.read()
                config_dict = json.loads(config_json)
            except ValueError as e:
                logger.error("Error while loading json: {}".format(config_json))
                raise e
        return config_dict

    @staticmethod
    def replace_config_models_paths_for_binary(context, config_path, resources_dir, name, **kwargs):
        config_dict = OvmsConfig.load(config_path)
        if config_dict is not None:
            if kwargs.get("replace_config_models_paths_for_binary", True):
                for model in config_dict[Config.MODEL_CONFIG_LIST]:
                    new_base_path = model["config"]["base_path"].replace(
                        Paths.MODELS_PATH_INTERNAL, os.path.join(resources_dir, Paths.MODELS_PATH_NAME), 1
                    )
                    model["config"]["base_path"] = new_base_path
                    if "graph_path" in model["config"]:
                        model["config"]["graph_path"] = model["config"]["graph_path"].replace(
                            Paths.MODELS_PATH_INTERNAL, os.path.join(resources_dir, Paths.MODELS_PATH_NAME), 1
                        )

            if kwargs.get("replace_config_custom_loader_paths_for_binary", True):
                for custom_loader in config_dict.get(Config.CUSTOM_LOADER_CONFIG_LIST, ""):
                    custom_loader_library_path = custom_loader["config"]["library_path"]
                    new_library_path = os.path.join(resources_dir, Paths.CUSTOM_LOADER_PATH_NAME,
                                                    CustomLoaderConsts.SAMPLE_CUSTOM_LOADER_NAME,
                                                    CustomLoaderConsts.SAMPLE_CUSTOM_LOADER_LIB_NAME)
                    custom_loader["config"]["library_path"] = new_library_path
                    if context.base_os == OsType.Windows:
                        raise NotImplementedError("Custom resources are not implemented for Windows")
                    copy_custom_lib_to_host(context.ovms_test_image, custom_loader_library_path, new_library_path)

            if kwargs.get("replace_config_custom_nodes_paths_for_binary", True):
                for custom_node in config_dict.get(Config.CUSTOM_NODE_LIBRARY_CONFIG_LIST, ""):
                    custom_node_library_path = custom_node["base_path"]
                    new_library_path = os.path.join(resources_dir, custom_node_library_path)
                    custom_node["base_path"] = new_library_path
                    if context.base_os == OsType.Windows:
                        raise NotImplementedError("Custom resources are not implemented for Windows")
                    copy_custom_lib_to_host(context.ovms_test_image, custom_node_library_path, new_library_path)

            if kwargs.get("replace_config_mediapipe_paths_for_binary", True):
                for mediapipe_config in config_dict.get(Config.MEDIAPIPE_CONFIG_LIST, []):
                    if "graph_path" in mediapipe_config:
                        mediapipe_config["graph_path"] = mediapipe_config["graph_path"].replace(
                            Paths.MODELS_PATH_INTERNAL, os.path.join(resources_dir, Paths.MODELS_PATH_NAME), 1
                        )
                    if "base_path" in mediapipe_config:
                        mediapipe_config["base_path"] = mediapipe_config["base_path"].replace(
                            Paths.MODELS_PATH_INTERNAL, os.path.join(resources_dir, Paths.MODELS_PATH_NAME), 1
                        )
                    if "subconfig" in mediapipe_config:
                        mediapipe_config["subconfig"] = mediapipe_config["subconfig"].replace(
                            Paths.MODELS_PATH_INTERNAL, os.path.join(resources_dir, Paths.MODELS_PATH_NAME), 1
                        )

        OvmsConfig.save(name, config_dict)

    @staticmethod
    def replace_subconfig_paths(name, subconfig_path, resources_dir):
        subconfig_dict = json.loads(Path(subconfig_path).read_text())
        for i, model in enumerate(subconfig_dict["model_config_list"]):
            subconfig_dict["model_config_list"][i]["config"]["base_path"] = model["config"]["base_path"].replace(
                Paths.MODELS_PATH_INTERNAL, os.path.join(resources_dir, Paths.MODELS_PATH_NAME)
            )
        OvmsConfig.save(name, subconfig_dict, subconfig_path)
        logger.info("Subconfig paths replaced")

    @staticmethod
    def create_subconfig(name, parameters, config_path_on_host):
        config_dict = None
        if parameters.custom_config is not None:
            config_dict = parameters.custom_config
        else:
            config_path = Path(os.path.join(config_path_on_host, Paths.CONFIG_FILE_NAME))
            if config_path.exists():
                config_dict = json.loads(config_path.read_text())

        subconfig_dict = {Config.MODEL_CONFIG_LIST: []}
        mediapipe_model = [model for model in parameters.models if model.is_mediapipe][0]
        feature_extraction_models = \
            [
                model for model in parameters.models
                if hasattr(model, "is_feature_extraction") and model.is_feature_extraction
            ]
        rerank_models = \
            [
                model for model in parameters.models
                if hasattr(model, "is_rerank") and model.is_rerank
            ]
        regular_models = [model for model in mediapipe_model.regular_models]

        filename = Paths.SUBCONFIG_FILE_NAME
        subconfigs = [os.path.basename(elem.get("subconfig", ""))
                      for elem in config_dict[Config.MEDIAPIPE_CONFIG_LIST]] if config_dict is not None else []
        if Paths.SUBCONFIG_FILE_NAME in subconfigs:
            for model in regular_models:
                subconfig_dict[Config.MODEL_CONFIG_LIST].append(model.get_config())
        else:
            for model in regular_models:
                subconfig_dict[Config.MODEL_CONFIG_LIST].append(model.get_config())
                filename = (
                    f"subconfig_{model.name}.json"
                    if config_dict is not None and all(["subconfig" in elem
                                                        for elem in config_dict[Config.MEDIAPIPE_CONFIG_LIST]])
                    else Paths.SUBCONFIG_FILE_NAME
                )
        mediapipe_resources_path = os.path.join(config_path_on_host, mediapipe_model.name)
        Path(mediapipe_resources_path).mkdir(parents=True, exist_ok=True)
        subconfig_path = os.path.join(mediapipe_resources_path, filename)
        OvmsConfig.save(name, subconfig_dict, subconfig_path)
        return subconfig_dict, subconfig_path
