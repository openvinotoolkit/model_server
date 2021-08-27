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
import logging
import os
import pickle
from collections import defaultdict
from logging import FileHandler

import grpc  # noqa
import pytest
from _pytest._code import ExceptionInfo, filter_traceback  # noqa
from _pytest.outcomes import OutcomeException

from constants import MODEL_SERVICE, PREDICTION_SERVICE
from utils.helpers import get_xdist_worker_nr, get_xdist_worker_count
from utils.xdist_utils import OvmsCLoadScheduling
from object_model.server import Server
from utils.cleanup import clean_hanging_docker_resources, delete_test_directory, \
    get_containers_with_tests_suffix, get_docker_client
from utils.logger import init_logger
from tensorflow_serving.apis import prediction_service_pb2_grpc, \
    model_service_pb2_grpc  # noqa
from utils.files_operation import get_path_friendly_test_name
from utils.parametrization import get_tests_suffix
from config import test_dir, test_dir_cleanup, artifacts_dir, using_xdist

logger = logging.getLogger(__name__)


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
    init_logger()

    init_conf_logger = logging.getLogger("init_conf")

    container_names = get_containers_with_tests_suffix()
    if container_names:
        init_conf_logger.info("Possible conflicting container names: {} "
                              "for given tests_suffix: {}".format(container_names, get_tests_suffix()))

    if artifacts_dir:
        os.makedirs(artifacts_dir, exist_ok=True)


def pytest_keyboard_interrupt (excinfo):
    clean_hanging_docker_resources()
    Server.stop_all_instances()

def pytest_unconfigure():
    # Perform cleanup.
    cleanup_logger = logging.getLogger("cleanup")

    cleanup_logger.info("Cleaning hanging docker resources with suffix: {}".format(get_tests_suffix()))
    clean_hanging_docker_resources()

    if test_dir_cleanup:
        cleanup_logger.info("Deleting test directory: {}".format(test_dir))
        delete_test_directory()

    if len(Server.running_instances) > 0:
        logger.warning("Test got unstopped docker instances")
    Server.stop_all_instances()


def _get_server_fixtures(item):
    server_fixtures = list(filter(lambda x: "start_server_" in x, item.fixturenames))
    return server_fixtures

@pytest.hookimpl(hookwrapper=True)
def pytest_collection_finish(session):
    yield
    # Collect all fixtures that starts Docker instance
    # This map will keep fixture usages in

    server_fixtures_to_item = defaultdict(lambda: [])
    for item in session.items:
        item._server_fixtures = _get_server_fixtures(item)
        for fixture in item._server_fixtures:
            server_fixtures_to_item[fixture].append(item)
    session._server_fixtures_to_item = server_fixtures_to_item.copy()

    # Try to order test execution minimal 'start_server_*' fixtures working
    ordered_items = []

    # Choose fixture with max tests assigned to be executed first.
    most_cases_lambda = lambda x: len(x[1])
    fixture_with_most_cases = max(server_fixtures_to_item.items(), key=most_cases_lambda)[0]

    fixtures_working = [fixture_with_most_cases] # FIFO queue
    tasks_for_nodes = []
    current_node = []
    while server_fixtures_to_item:
        current_fixture = fixtures_working[0]
        for item in server_fixtures_to_item[current_fixture]:
            if item not in ordered_items:
                ordered_items.append(item)
                current_node.append(item)
                item_fixtures = _get_server_fixtures(item)
                for it in item_fixtures:
                    # Test execute multiple fixtures  with servers, add fixture to be processed next (out of order).
                    if it not in fixtures_working:
                        fixtures_working.append(it)
                    if item in server_fixtures_to_item:
                        del server_fixtures_to_item[item]
        fixtures_working.remove(current_fixture)
        del server_fixtures_to_item[current_fixture]
        if not fixtures_working and server_fixtures_to_item:
            tasks_for_nodes.append(current_node)
            current_node = []
            fixtures_working.append(max(server_fixtures_to_item.items(), key=most_cases_lambda)[0])
    session.items = ordered_items

    node_to_test = [[] for i in range(get_xdist_worker_count())]
    for tasks in tasks_for_nodes:
        idx_min_tasks = node_to_test.index(min(node_to_test, key=lambda x: len(x)))
        node_to_test[idx_min_tasks].extend(tasks)

    #assert 0
    worker = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker:
        assigned_tests_path = os.path.join(artifacts_dir, f"assigned_tests_{worker}.xdist")
        with open(assigned_tests_path, "wb") as file:
            test_ids = list(map(lambda x: x.nodeid, node_to_test[get_xdist_worker_nr()]))
            f = pickle.dump(test_ids, file)
            ff = 0
    foo = 0


# if not using_xdist:
#     @pytest.hookimpl(hookwrapper=True)
#     def pytest_runtestloop(session):
#         # Override default runtestloop in order to sort test execution by used fixtures
#         # This operation will ensure that only required containers will run and container will be cleared after all usages.
#
#         # Collect all fixtures that starts Docker instance
#         # This map will keep fixture usages in tests
#         server_fixtures_to_item = defaultdict(lambda: [])
#         for item in session.items:
#             item._server_fixtures = list(filter(lambda x: "start_server_" in x, item.fixturenames))
#             for fixture in item._server_fixtures:
#                 server_fixtures_to_item[fixture].append(item)
#
#         # Sort test items using required fixtures as key (group test by fixture)
#         sorted_items = sorted(session.items, key=lambda x: x._server_fixtures )
#
#         for i, item in enumerate(sorted_items):
#             nextitem = sorted_items[i + 1] if i + 1 < len(sorted_items) else None
#             item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)
#
#             # Test finished: remove test item for all fixtures that was used
#             for fixture in item._server_fixtures:
#                 server_fixtures_to_item[fixture].remove(item)
#                 if len(server_fixtures_to_item[fixture]) == 0:
#                     # No other tests will use this docker instance so we can close it.
#                     Server.stop_by_fixture_name(fixture)


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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    yield
    # Test finished: remove test item for all fixtures that was used
    for fixture in item._server_fixtures:
        if item in item.session._server_fixtures_to_item[fixture]:
            item.session._server_fixtures_to_item[fixture].remove(item)
        if len(item.session._server_fixtures_to_item[fixture]) == 0:
            # No other tests will use this docker instance so we can close it.
            Server.stop_by_fixture_name(fixture)



def exception_catcher(when: str, outcome):
    if isinstance(outcome.excinfo, tuple):
        if len(outcome.excinfo) > 1 and isinstance(outcome.excinfo[1], OutcomeException):
            return
        exception_logger = logging.getLogger("exception_logger")
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
        _root_logger = logging.getLogger(None)
        _root_logger._test_log_handler = FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        _root_logger._test_log_handler.setFormatter(formatter)
        _root_logger.addHandler(_root_logger._test_log_handler)
    yield


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_logfinish(nodeid, location):
    if artifacts_dir:
        _root_logger = logging.getLogger(None)
        _root_logger.removeHandler(_root_logger._test_log_handler)
    yield


def pytest_xdist_make_scheduler(config, log):
    scheduler = OvmsCLoadScheduling(config, log)
    log.debug("Created xdist scheduler")
    return scheduler
