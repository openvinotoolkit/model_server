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

from tests.functional import config
from tests.functional.models.models_library import ModelsLib
# from tests.functional.utils.hooks import CURRENT_TARGET_DEVICE_DICT
from tests.functional.constants.os_type import OsType
from tests.functional.constants.ovms import (
    BASE_OS_PARAM_NAME,
    OVMS_TYPE_PARAM_NAME,
    TARGET_DEVICE_PARAM_NAME,
    USES_MAPPING_PARAM_NAME,
)
from tests.functional.constants.ovms_images import calculate_ovms_image_name
from tests.functional.constants.target_device import TargetDevice
from tests.functional.object_model.ovms_info import OvmsInfo
from tests.functional.utils.environment_info import EnvironmentInfo
from tests.functional.utils.logger import get_logger
from tests.functional.utils.marks import MarkTestParameters

logger = get_logger(__name__)


CURRENT_TARGET_DEVICE_DICT = {}


def get_marker_args(metafunc, marker_name):
    _marker = [marker for marker in metafunc.definition.own_markers if marker.name == marker_name]
    if _marker:
        return _marker[0].args
    return


def get_ids_with_target_device(parameter, func):
    # return id for target_device
    if parameter in vars(TargetDevice).values():
        return CURRENT_TARGET_DEVICE_DICT.get(parameter, parameter)
    # return custom id
    return func(parameter)


def parametrize_model_type(metafunc):
    args = get_marker_args(metafunc, MarkTestParameters.MODEL_TYPE)
    if args is None:
        parametrize_target_device(metafunc)
        return
    if isinstance(args[0], dict):
        params_list = [
            (device_type, result) for device_type in config.target_devices for result in args[0][device_type]
        ]
    else:
        params_list = [(device_type, result) for device_type in config.target_devices for result in args[0]]
    ids_list = [
        f"{CURRENT_TARGET_DEVICE_DICT.get(device_type, device_type)}-{model_type.__name__}"
        for device_type, model_type in params_list
    ]
    metafunc.parametrize(f"{TARGET_DEVICE_PARAM_NAME}, {MarkTestParameters.MODEL_TYPE}", params_list, ids=ids_list)


def parametrize_model_aux_type(metafunc):
    """Parametrize auxiliary (second) model for tests that need two models simultaneously.

    Requires model_type to also be present in the test — model_type handles target_device parametrization,
    so model_aux_type only parametrizes the model class itself (no device cross-product).
    """
    assert MarkTestParameters.MODEL_TYPE in metafunc.fixturenames, (
        f"model_aux_type requires model_type to also be a fixture in test {metafunc.function.__name__}"
    )
    args = get_marker_args(metafunc, MarkTestParameters.MODEL_AUX_TYPE)
    if args is None:
        return
    if isinstance(args[0], dict):
        params_list = [model for device_type in config.target_devices for model in args[0][device_type]]
    else:
        params_list = list(args[0])
    ids_list = [model.__name__ for model in params_list]
    metafunc.parametrize(MarkTestParameters.MODEL_AUX_TYPE, params_list, ids=ids_list)


def parametrize_all_models(metafunc):
    args = get_marker_args(metafunc, MarkTestParameters.ALL_MODELS)
    if args is None:
        parametrize_target_device(metafunc, config.target_devices)
        return
    params_list = []
    for device_type in config.target_devices:
        for _models in args[0]:
            if isinstance(_models, dict):
                params_list.append((device_type, _models[device_type]))
            else:
                params_list.append((device_type, _models))
    ids_list = lambda i: get_ids_with_target_device(
        i, lambda x: x[-1].name if len(x) > 0 and hasattr(x[-1], "name") else "empty"
    )
    metafunc.parametrize(f"{TARGET_DEVICE_PARAM_NAME}, {MarkTestParameters.ALL_MODELS}", params_list, ids=ids_list)


def parametrize_many_models(metafunc):
    args = get_marker_args(metafunc, MarkTestParameters.MANY_MODELS)
    if args is None:
        parametrize_target_device(metafunc)
        return
    # (ModelsLib.get_many_models, 8, 100) -> ("CPU", (ModelsLib.get_many_models(8), 100))
    params_list = [
        (device_type, (arg[0](device_type, arg[1]), arg[2])) for device_type in config.target_devices for arg in args
    ]
    ids_list = lambda i: get_ids_with_target_device(i, lambda x: f"count={len(x[0])}-iters={x[1]}")
    metafunc.parametrize(f"{TARGET_DEVICE_PARAM_NAME}, {MarkTestParameters.MANY_MODELS}", params_list, ids=ids_list)


def parametrize_iteration_info(metafunc):
    args = get_marker_args(metafunc, MarkTestParameters.ITERATION_INFO)
    if args is None:
        parametrize_target_device(metafunc)
        return
    # [(ModelsLib.various_models, 0, False, True)] -> ("CPU", ([(ModelsLib.various_models["CPU"][0], False, True)])
    params_list = []
    for device_type in config.target_devices:
        for _iteration_info_item in args:
            iteration_info = []
            for _iteration_info in _iteration_info_item:
                iteration_info.append(
                    (_iteration_info[0][device_type][_iteration_info[1]], _iteration_info[2], _iteration_info[3])
                )
            params_list.append((device_type, iteration_info))
    ids_list = lambda i: get_ids_with_target_device(i, ModelsLib.generate_ids_for_iteration_info)
    metafunc.parametrize(f"{TARGET_DEVICE_PARAM_NAME}, {MarkTestParameters.ITERATION_INFO}", params_list, ids=ids_list)


def parametrize_input_shape(metafunc):
    params_list = []
    args = get_marker_args(metafunc, MarkTestParameters.INPUT_SHAPE)
    if args is not None:
        # (ModelsLib.create_input_shapes_for_auto_reshape_tests, ModelsLib.reshapeable_model, ModelType.ONNX) ->
        # ("CPU", ModelsLib.create_input_shapes_for_auto_reshape_tests(ModelsLib.reshapeable_model["CPU"][ModelType.ONNX]))
        params_list = [
            (device_type, shape)
            for device_type in config.target_devices
            for shape in args[0](args[1][device_type][args[2]])
        ]
    args = get_marker_args(metafunc, MarkTestParameters.INPUT_SHAPE_NO_AUTO)
    if args is not None:
        params_list = [(device_type, shape) for device_type in config.target_devices for shape in args[0][device_type]]
    if params_list:
        ids_list = lambda i: get_ids_with_target_device(i, ModelsLib.generate_model_shape_ids)
        metafunc.parametrize(f"{TARGET_DEVICE_PARAM_NAME}, {MarkTestParameters.INPUT_SHAPE}", params_list, ids=ids_list)
    else:
        parametrize_target_device(metafunc)


def parametrize_plugin_config(metafunc):
    args = get_marker_args(metafunc, MarkTestParameters.PLUGIN_CONFIG)
    if args is None:
        parametrize_target_device(metafunc)
        return
    params_list = [
        (device_type, plugin_config) for device_type in config.target_devices for plugin_config in args[0][device_type]
    ]
    ids_list = lambda i: get_ids_with_target_device(i, lambda x: "-".join(map(lambda y: "%s=%s" % y, x.items())))
    metafunc.parametrize(f"{TARGET_DEVICE_PARAM_NAME}, {MarkTestParameters.PLUGIN_CONFIG}", params_list, ids=ids_list)


def parametrize_target_device(metafunc):
    ids = [CURRENT_TARGET_DEVICE_DICT.get(x, x) for x in config.target_devices]
    metafunc.parametrize(TARGET_DEVICE_PARAM_NAME, config.target_devices, ids=ids)


def parametrize_ovms_type(metafunc):
    metafunc.parametrize(OVMS_TYPE_PARAM_NAME, config.ovms_types)


def parametrize_uses_mapping(metafunc):
    value_to_id = {
        True: "use_mapping",
        False: "no_mapping",
        None: "default_model_mapping",
    }
    ids = [value_to_id[x] for x in config.uses_mapping]
    metafunc.parametrize(USES_MAPPING_PARAM_NAME, config.uses_mapping, ids=ids)


def parametrize_base_os(metafunc):
    ids = []
    params = config.base_os
    for _os in config.base_os:
        if OsType.Windows in config.base_os:
            os_name = OsType.Windows
            if len(config.base_os) > 1:
                raise NotImplementedError("Iterating with Windows OS is not supported")
        elif any(_os.lower() == value for key, value in vars(OsType).items() if not key.startswith("__")):
            image = calculate_ovms_image_name(config.target_devices[0], _os)
            env_info = EnvironmentInfo.get_instance(class_info=OvmsInfo, image=image)
            dist_name = env_info.get_os_distname()
            if dist_name.startswith("Ubuntu 22.04") or dist_name.startswith(OsType.Ubuntu22):
                os_name = OsType.Ubuntu22
            elif dist_name.startswith("Ubuntu 24.04") or dist_name.startswith(OsType.Ubuntu24):
                os_name = OsType.Ubuntu24
            elif dist_name.startswith("Red Hat") or dist_name.startswith(OsType.Redhat):
                os_name = OsType.Redhat
        else:
            raise Exception("Unexpected OS")
        if config.is_nginx_mtls:
            os_name = f"{os_name}_NGINX"
        ids.append(os_name.upper())
    metafunc.parametrize(BASE_OS_PARAM_NAME, params, ids=ids)


def log_configuration_variables():
    logger.info("============== configuration variables ==============")
    pt_env_vars = list(filter(lambda x: x[0].startswith("TT_"), os.environ.items()))
    pt_env_vars.sort()
    for env_var in pt_env_vars:
        logger.info("{}={}".format(*env_var))
