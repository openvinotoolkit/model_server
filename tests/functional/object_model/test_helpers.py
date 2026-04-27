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

import concurrent.futures
import enum
import requests

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http import HTTPStatus
from math import prod
from retry.api import retry_call
from statistics import mean

from tests.functional.utils.assertions import InvalidReturnCodeException
from tests.functional.utils.logger import get_logger, step

from tests.functional.object_model.test_environment import TestEnvironment

logger = get_logger(__name__)


def run_in_loop_during(action_to_run_in_loop, parallel_action, runs):
    for run in range(runs):
        logger.info(f"Run: {run}")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(parallel_action)

            counter = 0
            while not future.done():
                action_to_run_in_loop()
                counter += 1

            future.result()
            logger.info(f"Main thread action executed {counter} times.")


def run_all_actions_in_loop(actions, runs, max_workers=None):
    logger.info(f"Starting n={runs} parallel actions={','.join(map(lambda x: str(x), actions))}")
    with ThreadPoolExecutor(max_workers) as executor:
        futures = []
        for run in range(runs):
            logger.debug(f"Run: {run}")
            futures.extend([executor.submit(action) for action in actions])
        for future in concurrent.futures.as_completed(futures):
            future.result()
    logger.info("All actions finished")


def run_all_actions(action, arguments_list, max_workers=None):
    result = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        thread_list = []
        for arguments in arguments_list:
            thread = executor.submit(action, *arguments)
            thread_list.append(thread)

        for thread in concurrent.futures.as_completed(thread_list):
            thread_result = thread.result()
            result.append(thread_result)
    return result


# Model Control API Support
class Endpoints(enum.Enum):
    GET_CONFIG = "/v1/config"
    RELOAD_CONFIG = "/v1/config/reload"


def send_request_to_endpoint(port, address=None, endpoint=None, expected_code=None, retry=1, timeout=60):
    address = TestEnvironment.get_server_address() if address is None else address
    url_with_endpoint = f"http://{address}:{port}{endpoint}"
    logger.info(f"Try to send request to endpoint: {url_with_endpoint}")

    if endpoint == Endpoints.RELOAD_CONFIG.value:
        func = requests.post
    elif endpoint == Endpoints.GET_CONFIG.value:
        func = requests.get
    else:
        msg = f"Not supported endpoint: {endpoint}"
        raise ValueError(msg)
    retry_setup = {"tries": int(retry), "delay": 1}
    kwargs = {"url": url_with_endpoint, "params": {}, "timeout": timeout}
    ret = retry_call(func, fkwargs=kwargs, **retry_setup)
    if expected_code is None:
        return ret

    if ret.status_code != expected_code:
        logger.warning(f"Response text:\n{ret.text}")
    msg1 = f"Received status code is {ret.status_code}."
    msg2 = f"Expected return code is {expected_code}."
    if all([ret.status_code == HTTPStatus.OK.value,
            expected_code == HTTPStatus.CREATED.value,
            endpoint == Endpoints.RELOAD_CONFIG.value]):
        # Both of those codes are accepted for REST: https://jira.devtools.intel.com/browse/CVS-159242
        logger.warning(f"{msg1} {msg2} Both of those codes are accepted.")
        return ret
    elif not ret.status_code == expected_code:
        raise InvalidReturnCodeException(f"{msg1} {msg2}")
    logger.info(msg1)
    return ret


def send_reload_request(port, address=None, expected_code=None, retry=1, timeout=60):
    address = TestEnvironment.get_server_address() if address is None else address
    endpoint = Endpoints.RELOAD_CONFIG.value
    return send_request_to_endpoint(port, address, endpoint, expected_code, retry, timeout=timeout)


def get_config_request(port, address=None, expected_code=None, retry=1):
    address = TestEnvironment.get_server_address() if address is None else address
    endpoint = Endpoints.GET_CONFIG.value
    return send_request_to_endpoint(port, address, endpoint, expected_code, retry)


def _generate_permutations(input_shape, shape_results, example_input_data_for_predict, skip_first_items=0):
    result = []
    result_predict_shape = []
    shape_count = [len(v) for _, v in shape_results.items()]
    for i in range(prod(shape_count)):
        item = {k: None for k in input_shape}
        example_array = []

        tmp_shape_count = shape_count.copy()
        tmp_shape_count.reverse()

        input_array = {}
        for in_name in input_shape:
            current_cnt = tmp_shape_count.pop()
            idx = (i // prod(tmp_shape_count)) % current_cnt
            item[in_name] = shape_results[in_name][idx]
            input_array[in_name] = example_input_data_for_predict[in_name][idx]

        result.append(item)

        example_cnt = [len(v) for _, v in input_array.items()]
        for j in range(prod(example_cnt)):
            item = defaultdict(None)
            tmp_example_cnt = example_cnt.copy()
            tmp_example_cnt.reverse()
            for in_name in input_array:
                current_cnt = tmp_example_cnt.pop()
                idx = (j // prod(tmp_example_cnt)) % current_cnt
                item[in_name] = input_array[in_name][idx]
            example_array.append(item)
        result_predict_shape.append(example_array)
    return result[skip_first_items:], result_predict_shape[skip_first_items:]


def generate_dynamic_shape_permutation(model):
    """
    Generate possible shape with -1 and example input data shape for predict operation for testing purposes (tuple)
        For example input_shape = {'in': [1, 3, 224]}.
        This example contains 3 dim so we have 2^3 -1 (shape without -1 is not dynamic)
    Output:
    Output (note len of first and second output arguments are the same):
        first output argument: {<param_name>: <config-shape>}
            {'in': '(-1,3,224)'}
            {'in': '(1,-1,224)'}
            {'in': '(-1,-1,224)'}
            {'in': '(1,3,-1)'}
            {'in': '(-1,3,-1)'}
            {'in': '(1,-1,-1)'}
            {'in': '(-1,-1,-1)'}
        second output argument: [{<param_name>: <example inference data shape>}]
            [{'in': [1, 3, 224]}, {'in': [2, 3, 224]}],
            [{'in': [1, 1, 224]}, {'in': [1, 6, 224]}],
            [{'in': [1, 1, 224]}, {'in': [2, 1, 224]}, {'in': [1, 6, 224]}, {'in': [2, 6, 224]}],
            [{'in': [1, 3, 112]}, {'in': [1, 3, 448]}],
            [{'in': [1, 3, 224]}, {'in': [2, 3, 224]}, {'in': [1, 3, 448]}, {'in': [2, 3, 448]}],
            [{'in': [1, 1, 224]}, {'in': [1, 6, 224]}, {'in': [1, 1, 448]}, {'in': [1, 6, 448]}],
            [{'in': [1, 1, 224]}, {'in': [2, 1, 224]}, {'in': [1, 6, 224]}, {'in': [2, 6, 224]},
             {'in': [1, 1, 448]}, {'in': [2, 1, 448]}, {'in': [1, 6, 448]}, {'in': [2, 6, 448]}]

        {'<param-name>': (<config-shape>, <example inference data shape>)}
        {'in': ('(-1,3,224)', [[1, 3, 224], [2, 3, 224]])
        {'in': ('(1,-1,224), [[1, 1, 224], [1, 6, 224]])
        {'in': ('(-1,-1,224), [[1, 1, 224], [2, 1, 224], [1, 6, 224], [2, 6, 224]])
        {'in': ('(1,3,-1), [[1, 3, 112], [1, 3, 448]])
        {'in': ('(-1,3,-1), [[1, 3, 224], [2, 3, 224], [1, 3, 448], [2, 3, 448]])
        {'in': ('(1,-1,-1), [[1, 1, 224], [1, 6, 224], [1, 1, 448], [1, 6, 448]])
        {'in': ('(-1,-1,-1)', [[1, 1, 224], [2, 1, 224], [1, 6, 224], [2, 6, 224],
                               [1, 1, 448], [2, 1, 448], [1, 6, 448], [2, 6, 448]])
    """
    input_shape = model.input_shapes.copy()
    shape_results = defaultdict(lambda: [])
    example_input_shape_for_predict_operation = defaultdict(lambda: [])
    for in_name, shape in input_shape.items():
        layout = None
        if len(shape) >= 4:
            layout = model.inputs[in_name].get("layout", "NCHW")

        for i in range(2 ** len(shape)):
            new_shape = shape.copy()
            shape_for_predict_list = [shape.copy()]
            for dim in range(len(shape)):
                if (i >> dim) % 2 == 1:
                    new_shape[dim] = -1

                    copy_shape_for_predict_list = deepcopy(shape_for_predict_list)
                    for for_predict, copy_for_predict in zip(shape_for_predict_list, copy_shape_for_predict_list):
                        for_predict[dim] = max(1, shape[dim] // 2)
                        if layout is not None and layout.index("C") == dim:
                            copy_for_predict[dim] = shape[dim]
                        else:
                            copy_for_predict[dim] = shape[dim] * 2
                    shape_for_predict_list += copy_shape_for_predict_list

            shape_results[in_name].append(f"({','.join([str(x) for x in new_shape])})")
            example_input_shape_for_predict_operation[in_name].append(shape_for_predict_list)
    return _generate_permutations(input_shape, shape_results, example_input_shape_for_predict_operation, 1)


def generate_range_shape_permutation(model, skip_dims=2, generate_low_range=None, generate_high_range=None):
    """
    Generate possible shape with range and example of input data shape for predict operation:
        for example input_shape = [1, 2, 3, 4].
    For dim = 4 - function generate different sequence for dim 3, 4 (skip_dims = 2 -> skip dim 1, 2)
    Function generate possible permutation for dim(input_shape) - skip_dims - in
        or example perm(2) = 2^2 - 1 (option without range is not dynamic).
    Function create range based on value from input shape:
        {generate_low_range(input_shape[dim])}:{generate_high_range(input_shape[dim])}
    Default values are:
        generate_low_range(x) = x // 2
        generate_high_range(x) = x * 2
    For example input: {'in': [1, 3, 224, 224]}
    Output (note len of first and second output arguments are the same):
        first output argument: {<param_name>: <config-shape>}
            {'in': '(1,3,112:448,224)'},
            {'in': '(1,3,224,112:448)'},
            {'in': '(1,3,112:448,112:448)'
        second output argument: [{<param_name>: <example inference data shape>}]
            [{'in': [1, 3, 112, 224]}, {'in': [1, 3, 448, 224]}],
            [{'in': [1, 3, 224, 112]}, {'in': [1, 3, 224, 448]}],
            [{'in': [1, 3, 112, 112]}, {'in': [1, 3, 448, 112]}, {'in': [1, 3, 112, 448]}, {'in': [1, 3, 448, 448]}]
    """
    input_shape = model.input_shapes.copy()
    if generate_low_range is None:
        generate_low_range = lambda x: x // 2
    if generate_high_range is None:
        generate_high_range = lambda x: x * 2

    shape_results = defaultdict(lambda: [])
    example_input_shape_for_predict_operation = defaultdict(lambda: [])
    for in_name, shape in input_shape.items():
        layout = None
        if len(shape) >= 4:
            layout = model.inputs[in_name].get("layout", "NCHW")

        for i in range(2 ** len(shape[skip_dims:])):
            new_shape = shape[skip_dims:]
            shape_for_predict_list = [shape.copy()]
            for dim in range(len(new_shape)):
                if (i >> dim) % 2 == 1:
                    high_value = generate_high_range(new_shape[dim])
                    if layout is not None and layout.index("C") == (dim + skip_dims):
                        copy_for_predict[dim] = shape[dim]

                    new_shape[dim] = f"{generate_low_range(new_shape[dim])}:{high_value}"

                    copy_shape_for_predict_list = deepcopy(shape_for_predict_list)
                    for for_predict, copy_for_predict in zip(shape_for_predict_list, copy_shape_for_predict_list):
                        for_predict[dim + skip_dims] = max(1, generate_low_range(shape[dim + skip_dims]))
                        copy_for_predict[dim + skip_dims] = generate_high_range(shape[dim + skip_dims])
                    shape_for_predict_list += copy_shape_for_predict_list

            new_item = [str(x) for x in shape[:skip_dims]] + [str(x) for x in new_shape]
            shape_results[in_name].append(f"({','.join(new_item)})")
            example_input_shape_for_predict_operation[in_name].append(shape_for_predict_list)
    return _generate_permutations(input_shape, shape_results, example_input_shape_for_predict_operation, 1)


def check_initial_memory_usage(context, result, model):
    step("Log initial memory usage")
    memory_stats = []
    errors_logged = []
    res_monitor = result.attach_resource_monitor(context, start=False)
    memory_stats.append(float(res_monitor.get_stats_by_field(res_monitor.MEMORY_USAGE).replace("M", "")))
    logger.info(f"{model.name} initial memory usage:\t{memory_stats[0]}M")
    return memory_stats, errors_logged, res_monitor


def validate_memory_usage(
        iteration,
        memory_stats,
        errors_logged,
        validate_initial_memory_usage=True,
        check_memory_usage_initial_multiplier=2,
        check_memory_usage_multiplier=1.5,
):
    if len(memory_stats) > 2:
        current_memory_usage = memory_stats[-1]
        initial_memory_usage = memory_stats[0]
        first_iter_memory_usage = memory_stats[1]

        check_memory_usage_initial = current_memory_usage < check_memory_usage_initial_multiplier * initial_memory_usage
        if validate_initial_memory_usage and not check_memory_usage_initial:
            errors_logged.append(
                f"Too much memory usage! Initial: {initial_memory_usage}M; "
                f"{iteration} iteration: {current_memory_usage}M"
            )
        check_memory_usage = current_memory_usage < check_memory_usage_multiplier * first_iter_memory_usage
        if not check_memory_usage:
            errors_logged.append(
                f"Too much memory usage! 1st iteration: {first_iter_memory_usage}M; "
                f"{iteration} iteration: {current_memory_usage}M"
            )


def log_memory_usage_stats(memory_stats, errors_logged):
    initial_memory_usage = memory_stats[0]
    average_memory_usage = mean(memory_stats[1:])
    minimum_memory_usage = min(memory_stats[1:])
    maximum_memory_usage = max(memory_stats[1:])

    logger.info(f"Initial memory usage:\t{initial_memory_usage}")
    logger.info(f"Average inference memory usage:\t{average_memory_usage}")
    logger.info(f"Minimum inference memory usage:\t{minimum_memory_usage}")
    logger.info(f"Maximum inference memory usage:\t{maximum_memory_usage}")
    assert not errors_logged, f"Too much memory usage! Errors logged: {errors_logged}"
