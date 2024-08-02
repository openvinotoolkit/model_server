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

import sys
import copy
import time
import json
import argparse
import multiprocessing

try:
    from ovms_benchmark_client.metrics import XMetrics
    from ovms_benchmark_client.client import BaseClient
    from ovms_benchmark_client.client_tfs import TFS_Client
    from ovms_benchmark_client.client_kfs import KFS_Client
    from ovms_benchmark_client.db_exporter import DBExporter
except ModuleNotFoundError:
    from metrics import XMetrics
    from client import BaseClient
    from client_tfs import TFS_Client
    from client_kfs import KFS_Client
    from db_exporter import DBExporter

def get_client(xargs):
    if xargs["api"] == "TFS": return TFS_Client
    elif xargs["api"] == "KFS": return KFS_Client
    elif xargs["api"] == "REST": raise NotImplementedError("TODO - add REST support")
    else: return TFS_Client # default client API


# Version used for print only...
INTERNAL_VERSION="2.7"

# client engine - used for single and multiple client configuration
def run_single_client(xargs, worker_name_or_client, index, json_flag=None):

    # choose Client import for Triton / OVMS
    Client = get_client(xargs)

    if isinstance(worker_name_or_client, str):
        worker_name = worker_name_or_client
        client = Client(worker_name, xargs["server_address"], xargs["grpc_port"],
                        xargs["rest_port"], xargs["certs_dir"])
    elif isinstance(worker_name_or_client, Client):
        client = worker_name_or_client
    else: raise TypeError

    if json_flag is None:
        client.set_flags(xargs["json"], xargs["print_all"], xargs["print_time"], xargs["report_warmup"])
    else: client.set_flags(json_flag, xargs["print_all"], xargs["print_time"], xargs["report_warmup"])
    client.get_model_metadata(xargs["model_name"], xargs["model_version"], xargs["metadata_timeout"])

    stateful_id = int(xargs["stateful_id"]) + int(index)
    client.set_stateful(stateful_id, xargs["stateful_length"], 0)
    client.set_random_range(xargs["min_value"], xargs["max_value"])
    client.set_xrandom_number(xargs["xrand"])
    if xargs["dump_png"]: client.set_dump_png()

    bs_list = [int(b) for bs in xargs["bs"] for b in str(bs).split("-")]
    if xargs["stateful_length"] is not None and int(xargs["stateful_length"]) > 0:
        if xargs["dataset_length"] is not None:
            factor = int(xargs["dataset_length"]) // int(xargs["stateful_length"])
            dataset_length = (factor + 1) * int(xargs["stateful_length"])
        else: dataset_length = int(xargs["stateful_length"])
    elif xargs["dataset_length"] is not None and int(xargs["dataset_length"]) > 0:
        dataset_length = int(xargs["dataset_length"])
    else: dataset_length = None

    forced_shape = {}
    if xargs["shape"] is not None:
        # --shape input-name: 1 225 225 3 input_name2: 2 3
        # --shape layer:3: 64 64
        # --shape 1 225 225 3
        curr_input = None
        forced_shape[None] = []
        for shape_item in xargs["shape"]:
            if isinstance(shape_item, str) and shape_item and shape_item[-1] == ":":
                curr_input = str(shape_item[:-1])
                forced_shape[curr_input] = []
            else:
                dim = int(shape_item)
                assert dim > 0, "size has to be positive"
                forced_shape[curr_input].append(dim)

    client.prepare_data(xargs["data"], bs_list, dataset_length, forced_shape)
    error_limits = xargs["error_limit"], xargs["error_exposition"]
    client.print_info("start workload...", force=True)
    results = client.run_workload(xargs["steps_number"],
                                  xargs["duration"],
                                  xargs["step_timeout"],
                                  error_limits,
                                  xargs["warmup"],
                                  xargs["window"],
                                  xargs["hist_base"],
                                  xargs["hist_factor"],
                                  xargs["max_throughput"],
                                  xargs["concurrency"])
    return_code = 0 if client.get_status() else -1
    return return_code, results


# single client launcher
def exec_single_client(xargs):
    worker_id = xargs.get("id", "worker")
    # choose Client import for Triton / OVMS
    Client = get_client(xargs)
    client = Client(f"{worker_id}", xargs["server_address"], xargs["grpc_port"],
                    xargs["rest_port"], xargs["certs_dir"])
    client.set_flags(xargs["json"], xargs["print_all"], xargs["print_time"], xargs["report_warmup"])
    if xargs["list_models"]:
        client.set_flags(xargs["json"], True, xargs["print_time"], False)
        client.show_server_status()
        client.print_warning("Finished execution. If you want to run inference remove --list_models.")
        if xargs["model_name"] is not None:
            tout = int(xargs["metadata_timeout"])
            client.get_model_metadata(xargs["model_name"], xargs["model_version"], tout)
        return 0, {}

    if xargs["model_name"] is None:
        client.set_flags(xargs["json"], True, xargs["print_time"], False)
        client.show_server_status()
        raise ValueError("Model to inference is needed!")

    return_code, results = run_single_client(xargs, client, 0, False)
    base, factor = float(xargs["hist_base"]), float(xargs["hist_factor"])
    x_results = XMetrics(results)
    return return_code, x_results


# many client launcher
def exec_many_clients(xargs):
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
    if xargs["duration"] is not None:
        time.sleep(int(xargs["duration"]))

    final_return_code = 0
    common_results = XMetrics(submetrics=0)
    counter = int(xargs["concurrency"])

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
    return final_return_code, common_results


class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, data):
       self.stream.writelines(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)


###
# PART 2 - Execution
###

if __name__ == "__main__":
    description = """
    This is benchmarking client which uses TFS/KFS API to communicate with OVMS/TFS/KFS-based-services.
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
    parser.add_argument("--max_throughput", required=False, default=None,
                        help="max throughput in Sa per second, default: None")
    parser.add_argument("--max_value", required=False, default=255.0,
                        help="random maximal value, default: 255")
    parser.add_argument("--min_value", required=False, default=0.0,
                        help="random minimal value, default: 0")
    parser.add_argument("--xrand", required=False, default=8,
                        help="xrandom value, default: 8")
    parser.add_argument("--dump_png", required=False, action="store_true",
                        help="flag to dump PNG data")
    parser.add_argument("--step_timeout", required=False, default=30,
                        help="iteration timeout in seconds, default: 30")
    parser.add_argument("--metadata_timeout", required=False, default=45,
                        help="metadata timeout in seconds, default: 45")
    parser.add_argument("-Y", "--db_endpoint", required=False, default=None,
                        help="database endpoint configuration. default: None")
    parser.add_argument("-y", "--db_metadata", required=False, default=None, nargs="*",
                        help="database metadata configuration. default: None")
    parser.add_argument("--print_all", required=False, action="store_true",
                        help="flag to print all output")
    parser.add_argument("-ps", "--print_summary", required=False, action="store_true",
                        help="flag to print output summary")
    parser.add_argument("--print_time", required=False, action="store_true",
                        help="flag to print datetime next to each output line")
    parser.add_argument("--report_warmup", required=False, action="store_true",
                        help="flag to report warmup statistics")
    parser.add_argument("--certs_dir", required=False, default=None,
                        help="directory to certificates, default: None")
    parser.add_argument("-q", "--stateful_length", required=False, default=0,
                        help="stateful series length, default: 0")
    parser.add_argument("--stateful_id", required=False, default=1,
                        help="stateful sequence id, default: 1")
    parser.add_argument("--stateful_hop", required=False, default=0,
                        help="stateful sequence id hopsize, default: 0")
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
    parser.add_argument("--unbuffered", required=False, action="store_true",
                        help="flag to print stdout/stderr immediately rather than buffer")
    parser.add_argument("--api", required=False, default="TFS", choices=["TFS", "KFS", "REST"],
                        help="flag to choose which API to use")
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

    # buffering
    if xargs["unbuffered"]:
        sys.stdout = Unbuffered(sys.stdout)
        sys.stderr = Unbuffered(sys.stderr)

    # mongo exporter is optional
    worker_id = xargs.get("id", "worker")
    db_exporter = DBExporter(xargs, worker_id)

    # workload
    if xargs["concurrency"] in ("1", 1):
        return_code, common_results = exec_single_client(xargs)
    else: return_code, common_results = exec_many_clients(xargs)
    if not common_results:
        sys.exit(return_code)

    base, factor = float(xargs["hist_base"]), float(xargs["hist_factor"])
    if xargs["quantile_list"] is not None:
        common_results.recalculate_quantiles("window_", base, factor, xargs["quantile_list"])
    common_results["window_hist_factor"] = factor
    common_results["window_hist_base"] = base

    # exporting results
    db_exporter.upload_results(common_results, return_code)
    if xargs["json"]:
        jout = json.dumps(common_results)
        print(f"{BaseClient.json_prefix}###{worker_id}###STATISTICS###{jout}")

    if xargs["print_all"]:
        for key, value in common_results.items():
            sys.stdout.write(f"{worker_id}: {key}: {value}\n")

    if xargs["print_summary"]:
        sys.stdout.write("\n### Benchmark Parameters ###\n")
        if xargs['model_name'] is not None:
            model_name = xargs['model_name']
            sys.stdout.write(f" Model: {model_name}\n")
        if xargs['shape']:
            inp_shape = xargs['shape']
            sys.stdout.write(f" Input shape: {inp_shape}\n")
        if 'submetrics' in common_results:
            total_clients = common_results["submetrics"]
        else:
            total_clients = 1

        sys.stdout.write(f" Request concurrency: {total_clients}\n")
        if xargs['duration']:
            total_t = float(xargs['duration'])
            sys.stdout.write(f" Test Duration (s): Total (t): {total_t:.2f}")
        if xargs['warmup']:
            warm_up = float(xargs['warmup'])
            sys.stdout.write(f" | Warmup (u): {warm_up:.2f}")
        if xargs['window']:
            window = float(xargs['window'])
            sys.stdout.write(f" | Window (w): {window:.2f}\n")


        sys.stdout.write("\n### Benchmark Summary ###\n")
        sys.stdout.write(" ## General Metrics ##\n")

        sys.stdout.write(f" Duration(s): Total: {common_results['total_duration']:.2f}")
        sys.stdout.write(f" | Window: {common_results['window_total_duration']:.2f}\n")

        sys.stdout.write(f" Batches: Total: {common_results['total_batches']}")
        sys.stdout.write(f" | Window: {common_results['window_total_batches']}\n")

        if total_clients:
            sys.stdout.write("\n ## Latency Metrics (ms) ##\n")
            sys.stdout.write(f" Mean: {common_results['window_mean_latency']*1000:.2f}")
            sys.stdout.write(f" | stdev: {common_results['window_stdev_latency']*1000:.2f}")

            base, factor = float(xargs["hist_base"]), float(xargs["hist_factor"])
            xargs["quantile_list"] = [0.5, 0.9, 0.95]

            common_results.recalculate_quantiles("window_", base, factor, xargs["quantile_list"])

            for idx, v in enumerate(xargs["quantile_list"]):
                # Convert string to float
                try:
                    quantile_value = float(v)
                except ValueError:
                    # case where the string cannot be converted to a float
                    sys.stdout.write(f"Invalid quantile value: {v}")
                    continue
                # float to percentage
                q = str(int(quantile_value * 100))
                p = str("p") + q
                qv = str("qos_latency_") + str(idx)

                sys.stdout.write(f" | {p}: {common_results[qv]*1000:.2f}")
            sys.stdout.write("\n")
            sys.stdout.write("\n ## Throughput Metrics (fps) ##\n")

            # Brutto: Total number of frames processed or produced per second by a system or application.
            # It doesn't take into account any overhead or inefficiencies in the system
            # Netto: Effective or net frame rate, which is the frame rate adjusted for any overhead,
            # delays, or processing inefficiencies in the system
            sys.stdout.write(f" Frame Rate (FPS): Brutto: {common_results['brutto_frame_rate']:.2f}")
            sys.stdout.write(f" | Netto: {common_results['netto_frame_rate']:.2f} \n")
            sys.stdout.write(f" Batch Rate (batches/s): Brutto: {common_results['brutto_batch_rate']:.2f}")
            sys.stdout.write(f" | Netto: {common_results['netto_batch_rate']:.2f}\n")

    sys.exit(return_code)
