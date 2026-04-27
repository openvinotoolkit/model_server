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
import shutil
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import pytest
from grpc._channel import _InactiveRpcError
from tensorflow_serving.apis import get_model_status_pb2

from tests.functional.utils.assertions import UnexpectedResponseError, assert_raises_grpc_exception
from tests.functional.utils.context import Context
from tests.functional.utils.inference.communication. grpc import GRPC
from tests.functional.utils.inference.communication.rest import REST, RestCommunicationInterface
from tests.functional.utils.inference.serving import KFS, TFS
from tests.functional.utils.logger import get_logger, step

from tests.functional.config import is_nginx_mtls
from tests.functional.constants.components import OvmsComponents
from tests.functional.constants.models import Resnet
from tests.functional.constants.models_library import ModelsLib
from tests.functional.constants.ovms import Ovms, get_model_base_path
from tests.functional.constants.ovms_messages import OvmsMessages
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.paths import Paths
from tests.functional.constants.pipelines import MediaPipe, SimpleModelMediaPipe
from tests.functional.constants.requirements import Requirements
from tests.functional.fixtures.server import start_ovms
from tests.functional.object_model.inference_helpers import (
    InferenceBuilder,
    InferenceInfo,
    InferenceRequest,
    ensure_predict,
    get_and_validate_model_status,
    get_model_status,
    predict_and_assert,
    prepare_requests,
    prepare_requests_and_run_predict,
    wait_for_model_status,
)
from tests.functional.object_model.ovms_docker import OvmsDockerParams
from tests.functional.object_model.ovms_params import OvmsParams
from tests.functional.object_model.test_helpers import run_in_loop_during

logger = get_logger(__name__)


@pytest.mark.components(OvmsComponents.OVMS)
class TestPredictModel:

    class State(Enum):
        METHOD_START = 1
        NEW_VERSION_ADDING_START = 2
        NEW_VERSION_ADDING_END = 3
        OLD_VERSION_REMOVING_START = 4
        OLD_VERSION_REMOVING_END = 5
        OLD_VERSION_RESTORING_START = 6
        OLD_VERSION_RESTORING_END = 7
        NEW_VERSION_REMOVING_START = 8
        NEW_VERSION_REMOVING_END = 9
        METHOD_END = 10

    def validate_model_status_for_current_models_repo_state(self, flags, inference):

        status = None
        if flags.get("state") == TestPredictModel.State.METHOD_START:
            status = [{
                "version": 1,
                "accepted_states": [get_model_status_pb2.ModelVersionStatus.State.AVAILABLE],
                "accepted_error_messages": "OK",
            }]
            if flags.get("iteration", 0) > 0:
                status.append({
                    "version": 2,
                    "accepted_states": [get_model_status_pb2.ModelVersionStatus.State.END],
                    "accepted_error_messages": "OK",
                })
        elif flags.get("state") == TestPredictModel.State.NEW_VERSION_ADDING_END:
            status = [
                {
                    "version": 1,
                    "accepted_states": [
                        get_model_status_pb2.ModelVersionStatus.State.END,
                        get_model_status_pb2.ModelVersionStatus.State.UNLOADING,
                    ],
                    "accepted_error_messages": "OK",
                },
                {
                    "version": 2,
                    "accepted_states": [get_model_status_pb2.ModelVersionStatus.State.AVAILABLE],
                    "accepted_error_messages": "OK",
                },
            ]
        elif flags.get("state") == TestPredictModel.State.OLD_VERSION_REMOVING_END:
            status = [
                {
                    "version": 1,
                    "accepted_states": [get_model_status_pb2.ModelVersionStatus.State.END],
                    "accepted_error_messages": "OK",
                },
                {
                    "version": 2,
                    "accepted_states": [get_model_status_pb2.ModelVersionStatus.State.AVAILABLE],
                    "accepted_error_messages": "OK",
                },
            ]
        elif flags.get("state") == TestPredictModel.State.OLD_VERSION_RESTORING_END:
            status = [
                {
                    "version": 1,
                    "accepted_states": [get_model_status_pb2.ModelVersionStatus.State.END],
                    "accepted_error_messages": "OK",
                },
                {
                    "version": 2,
                    "accepted_states": [get_model_status_pb2.ModelVersionStatus.State.AVAILABLE],
                    "accepted_error_messages": "OK",
                },
            ]
        elif flags.get("state") == TestPredictModel.State.NEW_VERSION_REMOVING_END:
            status = [
                {
                    "version": 1,
                    "accepted_states": [get_model_status_pb2.ModelVersionStatus.State.AVAILABLE],
                    "accepted_error_messages": "OK",
                },
                {
                    "version": 2,
                    "accepted_states": [get_model_status_pb2.ModelVersionStatus.State.END],
                    "accepted_error_messages": "OK",
                },
            ]

        get_and_validate_model_status(inference, status)

    def update_model_version_and_restore(self, inference, ovms, model, flags, context: Context):
        container_folder = ovms.container_folder
        flags["state"] = TestPredictModel.State.METHOD_START

        if inference is not None:
            self.validate_model_status_for_current_models_repo_state(flags, inference)

        flags["state"] = TestPredictModel.State.NEW_VERSION_ADDING_START
        ovms_log_monitor = ovms.create_log(False)
        new_model = model.create_new_version(container_folder, model.version + 1)
        step(f"Created version: {new_model.version} for model: {new_model.name}")
        ovms_log_monitor.models_loaded([new_model])

        flags["state"] = TestPredictModel.State.NEW_VERSION_ADDING_END
        if inference is not None:
            self.validate_model_status_for_current_models_repo_state(flags, inference)

        flags["state"] = TestPredictModel.State.OLD_VERSION_REMOVING_START
        step(f"Removing version {model.version} of model {model.name}")
        model.delete_version(container_folder)
        ovms_log_monitor.models_unloaded([model])

        flags["state"] = TestPredictModel.State.OLD_VERSION_REMOVING_END
        if inference is not None:
            self.validate_model_status_for_current_models_repo_state(flags, inference)

        flags["state"] = TestPredictModel.State.OLD_VERSION_RESTORING_START
        step(f"Restoring version {model.version} of model {model.name}")
        model.prepare_resources(container_folder)

        flags["state"] = TestPredictModel.State.OLD_VERSION_RESTORING_END
        if inference is not None:
            self.validate_model_status_for_current_models_repo_state(flags, inference)

        flags["state"] = TestPredictModel.State.NEW_VERSION_REMOVING_START
        step(f"Removing version {new_model.version} of model {new_model.name}")
        ovms_log_monitor_unloaded = ovms.create_log(False)
        ovms_log_monitor_loaded = ovms.create_log(False)
        new_model.delete_version(container_folder)
        ovms_log_monitor_unloaded.models_unloaded([new_model])
        ovms_log_monitor_loaded.models_loaded([model])

        flags["state"] = TestPredictModel.State.NEW_VERSION_REMOVING_END
        if inference is not None:
            self.validate_model_status_for_current_models_repo_state(flags, inference)

        flags["iteration"] = flags.get("iteration", 0) + 1

    def update_nireq_in_config(self, model, ovms, nireq_to_test):
        for nireq in nireq_to_test:
            model.nireq = nireq
            ovms_log_monitor = ovms.create_log(False)
            ovms.update_model_list_and_config(ovms.name, [model])
            ovms_log_monitor.action_on_models(
                models_loaded=[model], custom_msg_list=[f"No of InferRequests: {nireq}"], timeout=60
            )

    @pytest.mark.priority_high
    @pytest.mark.reqids(Requirements.parity)
    @pytest.mark.model_type(ModelsLib.predict_models_and_mediapipe)
    @pytest.mark.api_on_commit
    def test_predict_various_models(self, context: Context, api_type, model_type):
        """
        <b>Description:</b>
        Run prediction on several models listed on:
        https://github.com/opencv/open_model_zoo/tree/master/models/public
        The IR versions can be found at (ssh access, user: guest):
        movilab-us-qaserver1.ra.intel.com:/home/Share/IR_2020.2

        <b>Input data:</b>
        1. Model to run inference on
        2. Input data for each model

        <b>Expected results:</b>
        Test passes when it is possible to run basic prediction using batch_size=1.

        <b>Steps:</b>
        1. Start OVMS with multiple models.
        2. Prepare inference input data.
        3. Run predict using gRPC or REST.
        4. Check if the response contains the expected output.
        """

        step("Start OVMS with multiple models.")
        model = model_type()
        result = start_ovms(context, OvmsParams(models=[model]))

        step("Prepare inference input data")
        input_data = model.prepare_input_data_from_model_datasets()
        port = result.ovms.get_port(api_type)

        step(f"Run predict using {api_type} with model name: {model.name}")
        inference = api_type(port=port, model=model)
        inference_request = inference.prepare_request(input_objects=input_data)
        outputs = inference.predict(inference_request)

        step("Check if the response contains the expected output.")
        model.validate_outputs(outputs, provided_input=input_data)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.sdl)
    @pytest.mark.parametrize(
        "version_generator", [lambda m: m.version + 1, lambda m: -1], ids=["version_1", "version_-1"]
    )
    @pytest.mark.api_regression
    @pytest.mark.model_type([Resnet])
    def test_predict_models_non_existing_version_grpc(
            self, context: Context, version_generator, grpc_api_type, model_type
    ):
        """
        <b>Description:</b>
        Run prediction on one model but with a non-existing version specified (gRPC interface)

        <b>SDL</b>
        1. SDL523-23: Verify that errors and exceptions are detected and handled appropriately

        <b>Input data:</b>
        1. Model with a specified name and version
        2. Non-existing model version [model.version + 1, -1]

        <b>Expected results:</b>
        Test passes when ovms returns an error specifying the reason of the failure.

        <b>Steps:</b>
        1. Start OVMS with at least one model with a specified version
        2. Prepare inference input
        3. Run predict using gRPC for this model but by specifying a non existing version
        4. Check if ovms returns an error with a code/message saying that the specified version does not exist
        """

        step("Start OVMS with at least one model with a specified version")
        model = model_type()
        result = start_ovms(context, OvmsParams(models=[model]))

        step("Prepare inference input")
        input_data = model.prepare_input_data()
        port = result.ovms.get_port(grpc_api_type)

        step(f"Run predict using {grpc_api_type} by specifying a non existing model name")
        inference = grpc_api_type(
            port=port,
            model_name=model.name,
            model_version=version_generator(model),
            batch_size=Ovms.BATCHSIZE,
            input_data_types=model.input_types,
            model_meta_from_serving=False,
        )
        request = inference.prepare_request(input_objects=input_data)

        step(
            "Check if ovms returns an error with a "
            "code/message saying that a model with the specified name does not exist"
        )
        assert_raises_grpc_exception(
            grpc_api_type.NOT_FOUND, OvmsMessages.MODEL_VERSION_NOT_FOUND, inference.predict, request=request
        )

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.sdl)
    @pytest.mark.parametrize("request_format", Ovms.TFS_REST_LAYOUT_TYPES)
    @pytest.mark.parametrize(
        "version_generator, message, status",
        [
            (lambda m: m.version + 1, OvmsMessages.MODEL_VERSION_NOT_FOUND, RestCommunicationInterface.NOT_FOUND),
            (lambda m: -1, OvmsMessages.ERROR_INVALID_URL, RestCommunicationInterface.INVALID_ARGUMENT),
        ],
        ids=["version_1", "version_-1"],
    )
    @pytest.mark.api_regression
    def test_predict_models_non_existing_version_rest(
        self, context: Context, request_format, tfs_rest_api_type, version_generator, message, status
    ):
        """
        <b>Description:</b>
        Run prediction on one model but with a non-existing version specified (REST interface)

        <b>SDL</b>
        1. SDL523-23: Verify that errors and exceptions are detected and handled appropriately

        <b>Input data:</b>
        1. Model with a specified name and version
        2. Non-existing model version [model.version + 1, -1]

        <b>Expected results:</b>
        Test passes when ovms returns an error specifying the reason of the failure.

        <b>Steps:</b>
        1. Start OVMS with at least one model with a specified version (fixture)
        2. Prepare inference input
        3. Run predict using REST for this model but by specifying a non existing version
        4. Check if ovms returns an error with a code/message saying that the specified version does not exist
        """
        step("Start OVMS (fixture)")
        model = Resnet()
        result = start_ovms(context, OvmsParams(models=[model]))

        step("Prepare inference input")
        input_data = model.prepare_input_data()

        step(f"Run predict using {tfs_rest_api_type} by specifying a non existing model name")
        port = result.ovms.get_port(tfs_rest_api_type)
        inference = tfs_rest_api_type(port=port, model_name=model.name, model_version=version_generator(model))
        request = inference.prepare_body_format(input_objects=input_data, request_format=request_format)

        step(
            "Check if ovms returns an error with a "
            "code/message saying that a model with the specified name does not exist"
        )
        tfs_rest_api_type.assert_raises_exception(status, message, inference.predict, request=request)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.sdl)
    @pytest.mark.api_regression
    @pytest.mark.model_type([Resnet, SimpleModelMediaPipe])
    def test_predict_non_existing_model_name(self, context: Context, api_type, model_type):
        """
        <b>Description:</b>
        Run prediction by specifying a non-existing model name.

        <b>SDL</b>
        1. SDL523-23: Verify that errors and exceptions are detected and handled appropriately

        <b>Input data:</b>
        1. Non-existing model name

        <b>Expected results:</b>
        Test passes when ovms returns an error specifying the reason of the failure.

        <b>Steps:</b>
        1. Start OVMS (fixture)
        2. Prepare inference input
        3. Run predict using gRPC or REST by specifying a non existing model name
        4. Check if ovms returns an error with a code/message saying that a model with the specified name does not exist
        """
        step("Start OVMS (fixture)")
        model = model_type()
        result = start_ovms(context, OvmsParams(models=[model]))
        port = result.ovms.get_port(api_type)

        step("Prepare inference input")
        data = model.prepare_input_data()

        step("Run predict using gRPC or REST by specifying a non existing model name")
        inference = api_type(
            port=port,
            model_name="fake_model_name",
            batch_size=Ovms.BATCHSIZE,
            input_data_types=model.input_types,
            model_meta_from_serving=False,
            is_mediapipe=issubclass(model_type, MediaPipe),
        )
        request = inference.prepare_request(input_objects=data)

        step(
            "Check if ovms returns an error with a "
            "code/message saying that a model with the specified name does not exist"
        )
        if api_type.serving == KFS:
            error_message = OvmsMessages.MEDIAPIPE_GRAPH_NOT_FOUND
        elif api_type.communication == REST and api_type.serving == TFS:
            error_message = OvmsMessages.ERROR_MODEL_NOT_FOUND
        elif api_type.communication == GRPC and api_type.serving == TFS:
            error_message = OvmsMessages.ERROR_PIPELINE_NOT_FOUND
        else:
            raise NotImplementedError
        api_type.assert_raises_exception(
            api_type.NOT_FOUND, error_message, inference.predict, request=request
        )

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.sdl)
    @pytest.mark.api_stress_and_load
    @pytest.mark.timeout(4000)
    def test_predict_model_deleted_in_loop(self, context: Context, api_type):
        """
        <b>Description:</b>
        Run predict in the loop, delete its model on the disc and check if predict error is not returned

        <b>SDL</b>
        1. SDL523-23: Verify that errors and exceptions are detected and handled appropriately

        <b>Input data:</b>
        1. Model name
        2. Input data

        <b>Expected results:</b>
        Test passes when ovms does not start returning error messages after the model was deleted.

        <b>Steps:</b>
        1. Start OVMS with at least one resnet-50 model configured
        2. Prepare inference input data
        3. Start predictions in the loop, hundreds of calls
        4. Check if they return correct results
        5. Delete the model's folder on the disc
        6. Check if subsequent predictions does not start to report errors due to missing model
        7. Restore the model folder on the disk and check if OVMS still reports correct results
        8. Repeat steps 2 - 6 ten times.
        9. Stop the prediction loop
        """

        step("Start OVMS with at least one resnet-50 model configured")
        model = Resnet()
        result = start_ovms(context, OvmsParams(models=[model]))

        step("Prepare inference input data")
        input_data = model.prepare_input_data_from_model_datasets()
        port = result.ovms.get_port(api_type)
        client = api_type(port=port, model=model)
        client.create_inference()

        container_model_folder = os.path.join(result.ovms.container_folder, Paths.MODELS_PATH_NAME, model.name)
        tmp_model_folder = os.path.join(result.ovms.container_folder, "tmp")

        def remove_model_and_restore():
            shutil.copytree(container_model_folder, tmp_model_folder)
            step(f"Delete model {model.name}")
            ovms_log_monitor = result.ovms.create_log(False)
            shutil.rmtree(container_model_folder)

            ovms_log_monitor.unloaded([model])
            step(f"Restore model {model.name}")
            ovms_log_monitor = result.ovms.create_log(False)
            shutil.copytree(tmp_model_folder, container_model_folder)

            ovms_log_monitor.reloaded([model])

        step(f"Run predict in loop using {api_type} with model name: {model.name}")
        for i in range(10):
            executor = ThreadPoolExecutor(max_workers=1)
            executor.submit(remove_model_and_restore)

            iterations = 250
            inference_info = InferenceInfo.create(
                client, model, 30, input_data, InferenceRequest(model=model, api_type=api_type, ovms=result.ovms)
            )
            for j in range(iterations):
                logger.debug(f"Range: {i}; Iteration: {j}")
                ensure_predict(inference_info)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.status, Requirements.online_modification)
    @pytest.mark.api_stress_and_load
    def test_get_status_during_config_update(self, context: Context, tfs_api_type):
        """
        <b>Description:</b>
        Run get status in the loop and during that time change nireq in config

        <b>Input data:</b>
        1. Model name
        2. Input data

        <b>Expected results:</b>
        Test passes when ovms starts and all get status requests return correct result.

        <b>Steps:</b>
        1. Start OVMS.
        2. Start sending GetModelStatus requests in the loop and during that time start updating nireq in config.
        3. Check if GetModelStatus requests return correct results.
        """

        step("Start OVMS")
        model = Resnet()
        result = start_ovms(context, OvmsParams(models=[model], use_config=True))
        port = result.ovms.get_port(tfs_api_type)
        inference = tfs_api_type(port=port, model=model)

        step("Start sending GetModelStatus requests in the loop and during that time start updating nireq in config.")
        accepted_model_states = [Ovms.ModelStatus.AVAILABLE, Ovms.ModelStatus.UNLOADING, Ovms.ModelStatus.LOADING]

        def get_model_status_during_reload(inference, accepted_model_states):
            try:
                get_model_status(inference, accepted_model_states)
            except (UnexpectedResponseError, _InactiveRpcError) as e:
                logger.warning(f"Getting model status returned error: {e}")

        run_in_loop_during(
            lambda: get_model_status_during_reload(inference, accepted_model_states),
            lambda: self.update_nireq_in_config(model, result.ovms, range(1, 6)),
            5,
        )

        step("Check if GetModelStatus requests return correct results.")
        get_and_validate_model_status(
            inference,
            [{
                "version": 1,
                "accepted_states": [get_model_status_pb2.ModelVersionStatus.State.AVAILABLE],
                "accepted_error_messages": "OK",
            }],
        )

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.metadata, Requirements.online_modification)
    @pytest.mark.api_stress_and_load
    def test_get_metadata_during_config_update(self, context: Context, api_type):
        """
        <b>Description:</b>
        Run get metadata in the loop and during that time change nireq in config.

        <b>Input data:</b>
        1. Model name
        2. Input data

        <b>Expected results:</b>
        Test passes when ovms starts and all get metadata requests return correct result.

        <b>Steps:</b>
        1. Start OVMS.
        2. Start sending GetModelMetadata requests in the loop and during that time start updating nireq in config.
           Check if GetModelMetadata requests return correct results.
        """

        step("Start OVMS")
        model = Resnet()
        result = start_ovms(context, OvmsParams(models=[model], use_config=True))
        port = result.ovms.get_port(api_type)
        inference = api_type(port=port, model=model)

        step("Start sending GetModelMetadata requests in the loop and during that time start updating nireq in config.")

        def get_and_validate_metadata_during_reload(inference, model):
            try:
                inference.get_and_validate_metadata(model)
            except (UnexpectedResponseError, _InactiveRpcError) as e:
                logger.warning(f"Getting model status returned error: {e}")

        run_in_loop_during(
            lambda: get_and_validate_metadata_during_reload(inference, model),
            lambda: self.update_nireq_in_config(model, result.ovms, range(1, 6)),
            5,
        )
        get_and_validate_metadata_during_reload(inference, model)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.online_modification)
    @pytest.mark.api_stress_and_load
    def test_predict_model_updated_in_loop(self, context: Context, api_type):
        """
        <b>Description:</b>
        Run predict in the loop, create new version of model and delete original on the disc and check if predict has
        not returned any errors

        <b>Input data:</b>
        1. Model name
        2. Input data

        <b>Expected results:</b>
        Test passes if ovms has not returned any error messages after the model was replaced with new version.

        <b>Steps:</b>
        1. Start OVMS with at least one resnet-50 model configured
        2. Prepare inference input data
        3. Start predictions in the loop, hundreds of calls
        4. Check if they return correct results
        5. Copy another model to the new numerical model's subfolder
        6. Check if the subsequent prediction on a newer version returns no errors
        7. Restore the model folder on the disk and check if OVMS starts to report correct results
        8. Repeat steps 2 - 6  ten times.
        9. Stop the prediction loop
        """

        step("Start OVMS with at least one resnet-50 model configured")
        model = Resnet()
        result = start_ovms(context, OvmsParams(models=[model]))

        step("Prepare inference input data")
        model_request = Resnet()  # duplicate model for request to allow for model's version to be resolved by OVMS
        model_request.version = None
        inference_infos = prepare_requests([InferenceRequest(model=model_request, api_type=api_type, ovms=result.ovms)])

        flags = {}
        step("Start predictions in the loop, hundreds of calls")
        run_in_loop_during(
            lambda: predict_and_assert(inference_infos),
            lambda: self.update_model_version_and_restore(None, result.ovms, model, flags, context),
            runs=10,
        )

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.status, Requirements.online_modification, Requirements.sdl)
    @pytest.mark.parametrize("num_of_iterations", [100])
    @pytest.mark.api_stress_and_load
    def test_get_status_model_updated_in_loop(self, context: Context, api_type, num_of_iterations):
        """
        <b>Description:</b>
        Run GetModelStatus in the loop, delete its model on the disc and check if predict error is returned

        <b>SDL</b>
        1. SDL523-23: Verify that errors and exceptions are detected and handled appropriately

        <b>Input data:</b>
        1. Model name
        2. Input data

        <b>Expected results:</b>
        Test passes when ovms starts returning error messages after the model was deleted.

        <b>Steps:</b>
        1. Start OVMS with at least one resnet-50 model configured
        2. Start sending GetModelStatus requests in the loop, hundreds of calls
        3. Check if they return correct results
        4. Copy another model to the new numerical model's subfolder
        5. Check if the subsequent prediction on a newer version returns no errors
        6. Restore the model folder on the disk and check if OVMS starts to report correct results
        7. Repeat steps 2 - 6  ten times.
        8. Stop the prediction loop
        """

        step("Start OVMS with at least one resnet-50 model configured")
        model = Resnet()
        result = start_ovms(context, OvmsParams(models=[model]))
        port = result.ovms.get_port(api_type)
        inference = api_type(
            port=port,
            model_name=model.name,
            batch_size=Ovms.BATCHSIZE,
            input_data_types=model.input_types,
            input_names=model.input_names,
            output_names=model.output_names,
            model_meta_from_serving=False,
        )

        step("Start sending GetModelStatus requests in the loop, hundreds of calls")
        flags = {}
        run_in_loop_during(
            lambda: self.validate_model_status_for_current_models_repo_state(flags, inference),
            lambda: self.update_model_version_and_restore(inference, result.ovms, model, flags, context),
            num_of_iterations,
        )

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.metadata, Requirements.online_modification, Requirements.sdl)
    @pytest.mark.parametrize("num_of_iterations", [100])
    @pytest.mark.api_stress_and_load_single
    def test_get_metadata_model_updated_in_loop(self, context: Context, api_type, num_of_iterations):
        """
        <b>Description:</b>
        Run GetModelMetadata in the loop, delete its model on the disc and check if predict error is returned

        <b>SDL</b>
        1. SDL523-23: Verify that errors and exceptions are detected and handled appropriately

        <b>Input data:</b>
        1. Model name
        2. Input data

        <b>Expected results:</b>
        Test passes when ovms starts returning error messages after the model was deleted.

        <b>Steps:</b>
        1. Start OVMS with at least one resnet-50 model configured
        2. Start sending GetModelMetadata requests in the loop, hundreds of calls
        3. Check if they return correct results
        4. Copy another model to the new numerical model's subfolder
        5. Check if the subsequent prediction on a newer version returns no errors
        6. Restore the model folder on the disk and check if OVMS starts to report correct results
        7. Repeat steps 2 - 6  ten times.
        8. Stop the prediction loop
        """

        step("Start OVMS with at least one resnet-50 model configured")
        model_class = Resnet
        result = start_ovms(context, OvmsParams(models=[model_class()]))
        model = result.models[0]

        port = result.ovms.get_port(api_type)
        inference = api_type(
            port=port,
            model_name=model.name,
            batch_size=Ovms.BATCHSIZE,
            input_data_types=model.input_types,
            input_names=model.input_names,
            output_names=model.output_names,
            model_meta_from_serving=False,
        )

        def get_and_validate_metadata_changing_version(inference, model_class, flags):
            if flags.get("state") in [
                TestPredictModel.State.NEW_VERSION_ADDING_START,
                TestPredictModel.State.NEW_VERSION_REMOVING_START,
                TestPredictModel.State.METHOD_END,
            ]:
                pass
            elif flags.get("state") in [
                None,
                TestPredictModel.State.METHOD_START,
                TestPredictModel.State.NEW_VERSION_REMOVING_END,
            ]:
                inference.get_and_validate_metadata(model_class(version=1))
            else:
                inference.get_and_validate_metadata(model_class(version=2))

        step("Start sending GetModelMetadata requests in the loop, hundreds of calls")
        flags = {}
        run_in_loop_during(
            lambda: get_and_validate_metadata_changing_version(inference, model_class, flags),
            lambda: self.update_model_version_and_restore(None, result.ovms, model, flags, context),
            num_of_iterations,
        )

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.sdl)
    @pytest.mark.api_regression
    def test_predict_model_deleted(self, context: Context, api_type):
        """
        <b>Description:</b>
        Run predict, delete its model on the disc and check if predict error is not returned

        <b>SDL</b>
        1. SDL523-23: Verify that errors and exceptions are detected and handled appropriately

        <b>Input data:</b>
        1. Model name
        2. Input data

        <b>Expected results:</b>
        Test passes when ovms does not start returning error messages after the model was deleted.

        <b>Steps:</b>
        1. Start OVMS with at least one resnet-50 model configured
        2. Run prediction
        3. Check if prediction return correct results
        4. Delete the model's folder on the disk
        5. Check if the subsequent prediction does not report an error due to missing model
        """

        step("Start OVMS with at least one resnet-50 model configured")
        model = Resnet()
        result = start_ovms(context, OvmsParams(models=[model]))

        step("Run prediction")
        input_data = model.prepare_input_data()
        port = result.ovms.get_port(api_type)
        inference = api_type(port=port, model=model)
        request = inference.prepare_request(input_objects=input_data)
        outputs = inference.predict(request)

        step("Check if predict returns correct results")
        model.validate_outputs(outputs)

        step("Delete the model's folder on the disk")
        ovms_log_monitor = result.ovms.create_log(False)
        model.delete(result.ovms.container_folder)
        model_path = get_model_base_path(model.base_path, context, result)
        ovms_log_monitor.ensure_contains_messages(
            [OvmsMessages.ERROR_DIRECTORY_DOES_NOT_EXISTS.format(model_path)]
        )

        step("Check if the subsequent prediction does not report an error due to missing model")
        outputs = inference.predict(request)
        model.validate_outputs(outputs)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.parity)
    @pytest.mark.api_regression
    def test_predict_model_updated(self, context: Context, api_type):
        """
        <b>Description:</b>
        Run predict, update its model on the disk, check if subsequent prediction is run on new model

        <b>Input data:</b>
        1. Model name
        2. Input data
        3. New model

        <b>Expected results:</b>
        Test passes when ovms starts returning prediction results of a new model

        <b>Steps:</b>
        1. Start OVMS with at least one resnet-50 model configured
        2. Run prediction
        3. Check if prediction return correct results
        4. Copy another  model to the new numerical model's subfolder
        5. Check if the subsequent prediction on a newer version returns no errors
        """

        step("Start OVMS with at least one resnet-50 model configured")
        model = Resnet()
        result = start_ovms(context, OvmsParams(models=[model]))

        step("Run prediction")
        input_data = model.prepare_input_data()
        port = result.ovms.get_port(api_type)
        inference = api_type(port=port, model=model)
        request = inference.prepare_request(input_objects=input_data)
        outputs = inference.predict(request)

        step("Check if predict returns correct results")
        model.validate_outputs(outputs)

        step("Copy another  model to the new numerical model's subfolder")
        ovms_log = result.ovms.create_log(False)
        newer_version = 2
        newer_resnet = model.create_new_version(result.ovms.container_folder, newer_version)
        ovms_log.models_loaded([newer_resnet])

        step("Check if the subsequent prediction on a newer version returns no errors")
        inference.model.version = newer_version
        request = inference.prepare_request(input_objects=input_data)
        outputs = inference.predict(request)
        for output_name in model.output_names:
            assert output_name in outputs, (
                f"Incorrect output name, expected: {output_name}, " f"found: {', '.join(outputs.keys())}."
            )

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.parity)
    @pytest.mark.parametrize("nr_of_versions", [4])
    @pytest.mark.parametrize("model_type", [Resnet])
    @pytest.mark.api_stress_and_load
    @pytest.mark.ovms_types_supported_for_test(OvmsType.DOCKER, OvmsType.DOCKER_CMD_LINE)
    def test_many_version_support_stress_and_load(self, context: Context, api_type, nr_of_versions, model_type):
        """
        <b>Description:</b>
        Check if OVMS can handle support of many models (same model different versions)

        <b>Input data:</b>
        1. Nr of different model versions

        <b>Expected results:</b>
        Predict operation of each model will work without any errors and problems

        <b>Steps:</b>
        1. Create set of model versions (number of different versions: <nr_of_versions>)
        2. Set model_version_policy={"latest": {"num_versions": <nr_of_versions>}}
        3. Validate ovms server log
        4. Execute simple predict operation to make sure that loaded models are working
        """

        step("Create set of model versions (number of different versions: <nr_of_versions>)")
        model = model_type()
        ovms_models = [model]
        model.model_version_policy = {"latest": {"num_versions": nr_of_versions}}

        original_cmd = None

        def get_command(docker_params):
            nonlocal original_cmd
            original_cmd = " ".join(docker_params["command"].to_list())
            docker_params["command"] = ["inf"]
            return docker_params

        ovms_params = OvmsDockerParams(models=[model], privileged=True, use_config=True, process_params=get_command)
        result = start_ovms(context, ovms_params, entrypoint="sleep", ensure_started=False)

        for version in range(2, nr_of_versions + 1):
            tmp_model = model.create_new_version(result.ovms.container_folder, version)
            tmp_model.model_path_on_host = os.path.join(result.ovms.container_folder, str(version))
            ovms_models.append(tmp_model)

        if is_nginx_mtls:
            original_cmd = f"/usr/bin/dumb-init -- /ovms_wrapper {original_cmd}"
        result.ovms.execute_command(cmd=original_cmd, stream=True)

        step("Execute simple predict operation to make sure that loaded models are working")
        for tmp_model in ovms_models:
            client = InferenceBuilder(tmp_model).create_client(
                api_type, result.ovms.get_port(api_type), model_version=tmp_model.version
            )
            wait_for_model_status(client, [Ovms.ModelStatus.AVAILABLE], timeout=60)
            prepare_requests_and_run_predict([InferenceRequest(ovms=result.ovms, model=tmp_model, api_type=api_type)])

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.parity)
    @pytest.mark.model_type(ModelsLib.models_with_encoded_names)
    @pytest.mark.api_regression
    def test_predict_model_encoded_name(self, context: Context, api_type, model_type):
        """
        <b>Description:</b>
        Run prediction on models with names that include slash or whitespace. Encode names for REST:
        %20 (whitespace) and %2f (/)

        <b>Input data:</b>
        1. Model to run inference on
        2. Input data for each model

        <b>Expected results:</b>
        Test passes when it is possible to run basic prediction using batch_size=1.

        <b>Steps:</b>
        1. Start OVMS with multiple models.
        2. Prepare inference input data.
        3. Run predict using gRPC or REST.
        4. Check if the response contains the expected output.
        """
        step("Start OVMS with multiple models.")
        model = model_type()
        result = start_ovms(context, OvmsParams(models=[model], use_config=True))

        step("Prepare inference input data")
        input_data = model.prepare_input_data_from_model_datasets()
        port = result.ovms.get_port(api_type)

        step(f"Run predict using {api_type} with model name: {model.name}")
        if api_type.communication == REST:  # encode model names for REST
            model.name = model.name.replace("/", "%2F").replace(" ", "%20")
        inference = api_type(port=port, model=model)
        inference_request = inference.prepare_request(input_objects=input_data)
        outputs = inference.predict(inference_request)

        step("Check if the response contains the expected output.")
        model.validate_outputs(outputs, provided_input=input_data)
