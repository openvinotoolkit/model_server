#
# Copyright (c) 2021 Intel Corporation
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

###
# Do you see bug? - call krzysztof.czarnecki@intel.com
###

###
# PART 1 - Import and Definitions
###

import copy
import sys
import json
import argparse
import multiprocessing
import time
import os

from ovms_benchmark_client.metrics import XMetrics
from ovms_benchmark_client.db_exporter import DBExporter
from ovms_benchmark_client.client_ovms import OVmsClient
from ovms_benchmark_client.client import BaseClient

try:
    from ovms_benchmark_client.client_nvtrt import NvTrtClient
except ImportError:
    # NvTrt is optional
    print("NvTrt is not included!")
    NvTrtClient = None

# Version used for print only...
INTERNAL_VERSION="1.12"


# client engine - used for single and multiple client configuration
def run_single_client(xargs, worker_name_or_client, index, json_flag=None):

    # choose Client import for Triton / OVMS
    if xargs["nv_triton"]:
        if NvTrtClient is None:
            raise TypeError("NvTrt is not included!")
        else: Client = NvTrtClient
    else: Client = OVmsClient

    if isinstance(worker_name_or_client, str):
        worker_name = worker_name_or_client
        client = Client(worker_name, xargs["server_address"], xargs["grpc_port"],
                        xargs["rest_port"], xargs["certs_dir"])
    elif isinstance(worker_name_or_client, Client):
        client = worker_name_or_client
    else: raise TypeError

    if json_flag is None:
        client.set_flags(xargs["json"], xargs["print_all"])
    else: client.set_flags(json_flag, xargs["print_all"])
    client.get_model_metadata(xargs["model_name"],
                              xargs["model_version"],
                              xargs["metadata_timeout"])
    stateful_id = int(xargs["stateful_id"]) + int(index)
    client.set_stateful(stateful_id, xargs["stateful_length"], 0)
    client.set_random_range(xargs["min_value"], xargs["max_value"])
    bs_list = [int(b) for bs in xargs["bs"] for b in str(bs).split("-")]

    if xargs["stateful_length"] is not None and int(xargs["stateful_length"]) > 0:
        if xargs["dataset_length"] is not None:
            factor = int(xargs["dataset_length"]) // int(xargs["stateful_length"])
            dataset_length = (factor + 1) * int(xargs["stateful_length"])
        else: dataset_length = int(xargs["stateful_length"])
    elif xargs["dataset_length"] is not None and int(xargs["dataset_length"]) > 0:
        dataset_length = int(xargs["dataset_length"])
    else: dataset_length = None

    client.prepare_data(xargs["data"], bs_list, dataset_length, xargs["shape"])
    error_limits = xargs["error_limit"], xargs["error_exposition"]
    results = client.run_workload(xargs["steps_number"],
                                  xargs["duration"],
                                  xargs["step_timeout"],
                                  error_limits,
                                  xargs["warmup"],
                                  xargs["window"],
                                  xargs["hist_base"],
                                  xargs["hist_factor"])
    return_code = 0 if client.get_status() else -1
    return return_code, results


# single client launcher
def exec_single_client(xargs, db_exporter):
    worker_id = xargs.get("id", "worker")
    # choose Client import for Triton / OVMS
    if xargs["nv_triton"]:
        if NvTrtClient is None:
            raise TypeError("NvTrt is not included!")
        else: Client = NvTrtClient
    else: Client = OVmsClient
    client = Client(f"{worker_id}", xargs["server_address"], xargs["grpc_port"],
                    xargs["rest_port"], xargs["certs_dir"])
    client.set_flags(xargs["json"], xargs["print_all"])
    if xargs["list_models"]:
        client.set_flags(xargs["json"], True)
        client.show_server_status()
        client.print_warning("Finished execution. If you want to run inference remove --list_models.")
        if xargs["model_name"] is not None:
            tout = int(xargs["metadata_timeout"])
            client.get_model_metadata(xargs["model_name"], xargs["model_version"], tout)
        return 0

    if xargs["model_name"] is None:
        client.set_flags(xargs["json"], True)
        client.show_server_status()
        raise ValueError("Model to inference is needed!")

    return_code, results = run_single_client(xargs, client, 0)
    base, factor = float(xargs["hist_base"]), float(xargs["hist_factor"])
    x_results = XMetrics(results)

    if xargs["quantile_list"] is not None:
        x_results.recalculate_quantiles("window_", base, factor, xargs["quantile_list"])
    x_results["window_hist_factor"] = factor
    x_results["window_hist_base"] = base
    db_exporter.upload_results(x_results, return_code)
    return return_code


# many client launcher
def exec_many_clients(xargs, db_exporter):
    def launcher(worker_name, queue):
        xargs2 = copy.deepcopy(xargs)
        return_code, results = run_single_client(
            xargs2, worker_name, index, False)
        queue.put((return_code, results))

    queue = multiprocessing.Queue()
    for index in range(int(xargs["concurrency"])):
        worker_name = f"{worker_id}.{index}"
        fargs = (worker_name, queue)
        job = multiprocessing.Process(target=launcher, args=fargs)
        job.start()

    final_return_code = 0
    common_results = XMetrics(submetrics=0)
    counter = int(xargs["concurrency"])
    if xargs["duration"] is not None:
        time.sleep(int(xargs["duration"]))
    while counter > 0:
        time.sleep(int(xargs["sync_interval"]))
        while queue.qsize() > 0:
            return_code, results = queue.get()
            if return_code != 0:
                final_return_code = return_code
                sys.stderr.write(f"return code:{return_code}\n")
            x_results = XMetrics(results)
            common_results += x_results
            counter -= 1
    base, factor = float(xargs["hist_base"]), float(xargs["hist_factor"])
    if xargs["quantile_list"] is not None:
        common_results.recalculate_quantiles("window_", base, factor, xargs["quantile_list"])
    common_results["window_hist_factor"] = factor
    common_results["window_hist_base"] = base
    db_exporter.upload_results(common_results, final_return_code)

    if xargs["json"]:
        jout = json.dumps(common_results)
        print(f"{BaseClient.json_prefix}###{worker_id}###STATISTICS###{jout}")
    if xargs["print_all"]:
        for key, value in common_results.items():
            sys.stdout.write(f"{worker_id}: {key}: {value}\n")
    return final_return_code

###
# PART 2 - Execution
###

if __name__ == "__main__":
    description = """
    This is benchmarking client which uses TF API to communicate with OVMS/TFS.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--id", required=False, default="worker",
                        help="client id. default: worker")
    parser.add_argument("-c", "--concurrency", required=False, default="1",
                        help="concurrency - number of parrlel clients. default: 1")
    parser.add_argument("-a", "--server_address", required=False, default=None,
                        help="url to rest/grpc OVMS service. default: None")
    parser.add_argument("-p", "--grpc_port", required=False, default=None,
                        help="port to grpc OVMS service. default: None")
    parser.add_argument("-r", "--rest_port", required=False, default=None,
                        help="port to rest OVMS service. default: None")
    parser.add_argument("-l", "--list_models", required=False, action="store_true",
                        help="check status of all models (finish after this)")
    parser.add_argument("-b", "--bs", required=False, default=[1], nargs="*",
                        help="batchsize, can be used multiple values. default: 1")
    parser.add_argument("-s", "--shape", required=False, default=None, nargs="*",
                        help="shape for data generation (bs has to be -1/0). default: None")
    parser.add_argument("-d", "--data", required=False, default=None, nargs="*",
                        help="data to inference, can be used multiple values")
    parser.add_argument("-j", "--json", required=False, action="store_true",
                        help="flag to form output in JSON format")
    parser.add_argument("-m", "--model_name", required=False, default=None,
                        help="model name to inference, default: None")
    parser.add_argument("-k", "--dataset_length", required=False, default=None,
                        help="synthetic dataset length, default: None")
    parser.add_argument("-v", "--model_version", required=False, default=None,
                        help="model version to inference, default: None")
    parser.add_argument("-n", "--steps_number", required=False, default=None,
                        help="number of iteration, default: None")
    parser.add_argument("-t", "--duration", required=False, default=None,
                        help="duration in seconds, default: None")
    parser.add_argument("-u", "--warmup", required=False, default=0,
                        help="warmup duration in seconds, default: 0")
    parser.add_argument("-w", "--window", required=False, default=None,
                        help="window duration in seconds, default: None")
    parser.add_argument("-e", "--error_limit", required=False, default=None,
                        help="counter limit of errors to break, default: None")
    parser.add_argument("-x", "--error_exposition", required=False, default=None,
                        help="counter limit of errors to show, default None")
    parser.add_argument("--max_value", required=False, default=255.0,
                        help="random maximal value, default: 255")
    parser.add_argument("--min_value", required=False, default=0.0,
                        help="random minimal value, default: 0")
    parser.add_argument("--step_timeout", required=False, default=30,
                        help="iteration timeout in seconds, default: 30")
    parser.add_argument("--metadata_timeout", required=False, default=30,
                        help="metadata timeout in seconds, default: 30")
    parser.add_argument("-y", "--db_config", required=False, default=None,
                        help="database configuration. default: None")
    parser.add_argument("--print_all", required=False, action="store_true",
                        help="flag to form output in JSON format")
    parser.add_argument("--certs_dir", required=False, default=None,
                        help="directory to certificats, default: None")
    parser.add_argument("-q", "--stateful_length", required=False, default=0,
                        help="stateful series length, default: 0")
    parser.add_argument("--stateful_id", required=False, default=1,
                        help="stateful sequence id, default: 1")
    parser.add_argument("--stateful_hop", required=False, default=0,
                        help="stateful sequence id hopsize, default: 0")
    parser.add_argument("--nv_triton", required=False, action="store_true",
                        help="flag to use NvClient to connect with Nvidia Triton Server")
    parser.add_argument("--sync_interval", required=False, default=1,
                        help="sync interval for multi-client mode, default: 1")
    parser.add_argument("--quantile_list", required=False, default=None, nargs="*",
                        help="quantile list, default: None")
    parser.add_argument("--hist_factor", required=False, default=100,
                        help="histogram factor, default: 100")
    parser.add_argument("--hist_base", required=False, default=1.5,
                        help="histogram base, default: 1.5")
    parser.add_argument("--internal_version", required=False, action="store_true",
                        help="flag to print internal version")
    xargs = vars(parser.parse_args())
    if xargs["internal_version"]:
        print(INTERNAL_VERSION)
        sys.exit(0)

    # check address is specified
    server_address = xargs["server_address"]
    assert server_address is not None

    # list models cannot be checked when concurrency > 1
    if xargs["list_models"]:
        assert xargs["concurrency"] in ("1", 1), "to list models use concurrency eq. to 1"

    # check duration is specified
    if not xargs["list_models"]:
        duration_error_flag = xargs["steps_number"] is None and xargs["duration"] is None
        assert not duration_error_flag, "Steps/duration not set!"

    # mongo exporter is optional
    db_exporter = DBExporter(xargs)

    worker_id = xargs.get("id", "worker")
    if xargs["concurrency"] in ("1", 1):
        return_code = exec_single_client(xargs, db_exporter)
    else: return_code = exec_many_clients(xargs, db_exporter)
    sys.exit(return_code)
