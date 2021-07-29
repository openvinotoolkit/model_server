#
# Copyright (c) 2018-2019 Intel Corporation
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
from logging import FileHandler

import grpc  # noqa
import pytest
from _pytest._code import ExceptionInfo, filter_traceback  # noqa
from _pytest.outcomes import OutcomeException

from constants import MODEL_SERVICE, PREDICTION_SERVICE
from utils.cleanup import clean_hanging_docker_resources, delete_test_directory, \
    get_containers_with_tests_suffix, get_docker_client
from utils.logger import get_logger
from tensorflow_serving.apis import prediction_service_pb2_grpc, \
    model_service_pb2_grpc  # noqa
from utils.files_operation import get_path_friendly_test_name
from utils.parametrization import get_tests_suffix
from config import test_dir, test_dir_cleanup, artifacts_dir

logger = get_logger(__name__)


pytest_plugins = [
    'fixtures.model_download_fixtures',
    'fixtures.model_conversion_fixtures',
    'fixtures.server_detection_model_fixtures',
    'fixtures.server_for_update_fixtures',
    'fixtures.server_local_models_fixtures',
    'fixtures.server_multi_model_fixtures',
    'fixtures.server_remote_models_fixtures',
    'fixtures.server_with_batching_fixtures',
    'fixtures.server_with_version_policy_fixtures',
    'fixtures.test_files_fixtures',
    ]


def pytest_sessionstart():
    for item in os.environ.items():
        logger.debug(item)


@pytest.fixture(scope="session")
def get_docker_context(request):
    client = get_docker_client()
    request.addfinalizer(client.close)
    return client


@pytest.fixture()
def create_grpc_channel():
    def _create_channel(address: str, service: int):
        channel = grpc.insecure_channel(address)
        if service == MODEL_SERVICE:
            return model_service_pb2_grpc.ModelServiceStub(channel)
        elif service == PREDICTION_SERVICE:
            return prediction_service_pb2_grpc.PredictionServiceStub(channel)
        return None

    return _create_channel


def pytest_configure():
    # Perform initial configuration.
    init_conf_logger = get_logger("init_conf")

    container_names = get_containers_with_tests_suffix()
    if container_names:
        init_conf_logger.info("Possible conflicting container names: {} "
                              "for given tests_suffix: {}".format(container_names, get_tests_suffix()))

    if artifacts_dir:
        os.makedirs(artifacts_dir, exist_ok=True)


def pytest_unconfigure():
    # Perform cleanup.
    cleanup_logger = get_logger("cleanup")

    cleanup_logger.info("Cleaning hanging docker resources with suffix: {}".format(get_tests_suffix()))
    clean_hanging_docker_resources()

    if test_dir_cleanup:
        cleanup_logger.info("Deleting test directory: {}".format(test_dir))
        delete_test_directory()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call():
    __tracebackhide__ = True
    try:
        outcome = yield
    finally:
        pass
    exception_catcher("call", outcome)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup():
    __tracebackhide__ = True
    try:
        outcome = yield
    finally:
        pass
    exception_catcher("setup", outcome)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown():
    __tracebackhide__ = True
    try:
        outcome = yield
    finally:
        pass
    exception_catcher("teardown", outcome)


def exception_catcher(when: str, outcome):
    if isinstance(outcome.excinfo, tuple):
        if len(outcome.excinfo) > 1 and isinstance(outcome.excinfo[1], OutcomeException):
            return
        exception_logger = get_logger("exception_logger")
        exception_info = ExceptionInfo.from_exc_info(outcome.excinfo)
        exception_info.traceback = exception_info.traceback.filter(filter_traceback)
        exc_repr = exception_info.getrepr(style="short", chain=False)\
            if exception_info.traceback\
            else exception_info.exconly()
        exception_logger.error('Unhandled Exception during {}: \n{}'
                               .format(when.capitalize(), str(exc_repr)))




@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_logstart(nodeid, location):
    if artifacts_dir:
        test_name = get_path_friendly_test_name(location)
        log_path = os.path.join(artifacts_dir, f"{test_name}.log")
        _root_logger = get_logger(None)
        _root_logger._test_log_handler = FileHandler(log_path)
        _root_logger.addHandler(_root_logger._test_log_handler)
    yield


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_logfinish(nodeid, location):
    if artifacts_dir:
        _root_logger = get_logger(None)
        _root_logger.removeHandler(_root_logger._test_log_handler)
    yield
