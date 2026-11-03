"""Benchmark OVMS KServe REST or gRPC inference using the Triton client."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import importlib
import json
import math
import multiprocessing
from pathlib import Path
import sys
import threading
import time

import numpy as np
from tritonclient.utils import triton_to_np_dtype

_worker_gate = None


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "-m", required=True, help="Served model name")
    parser.add_argument("--model-version", default="", help="Default: server-selected version")
    parser.add_argument("--endpoint", choices=("rest", "grpc"), default="grpc")
    parser.add_argument("--host", default="localhost", help="Hostname or IP, without a URL scheme")
    parser.add_argument("--port", type=positive_int, help="Default: 8000 for REST, 9000 for gRPC")
    parser.add_argument("--concurrency", type=positive_int, default=1)
    parser.add_argument("--requests", "--request-count", type=positive_int, default=5000)
    parser.add_argument("--timeout", type=positive_seconds, default=60.0, help="Request timeout in seconds")
    parser.add_argument("--shape", action="append", default=[], metavar="NAME:DIMS",
                        help="Concrete full input shape, e.g. image:1,224,224,3; repeat for multiple inputs")
    parser.add_argument("--input-file", action="append", default=[], metavar="NAME=PATH",
                        help="Raw file for BYTES, or .npy for numeric data; repeat for multiple inputs")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON report")
    args = parser.parse_args(argv)
    if args.port is None:
        args.port = 8000 if args.endpoint == "rest" else 9000
    if args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def named_values(values, names, separator):
    result = {}
    for value in values:
        name, found, content = value.partition(separator)
        if not found:
            if len(names) != 1:
                raise ValueError(f"Specify an input name using NAME{separator}VALUE for multiple inputs")
            name, content = names[0], value
        if name not in names:
            raise ValueError(f"Unknown input {name!r}; model inputs: {', '.join(names)}")
        if name in result:
            raise ValueError(f"Duplicate option for input {name!r}")
        if not content:
            raise ValueError(f"Missing value for input {name!r}")
        result[name] = content
    return result


def concrete_shape(value):
    if value == "scalar":
        return ()
    shape = tuple(int(dimension) for dimension in value.split(","))
    if any(dimension <= 0 for dimension in shape):
        raise ValueError("Input shape dimensions must be positive; use 'scalar' for rank zero")
    return shape


def prepare_inputs(metadata, shape_options, file_options):
    """Resolve metadata, shape overrides, and files into reusable input arrays."""
    names = [item["name"] for item in metadata["inputs"]]
    shapes = named_values(shape_options, names, ":")
    files = named_values(file_options, names, "=")
    prepared = []
    for item in metadata["inputs"]:
        name, datatype = item["name"], item["datatype"]
        model_shape = tuple(int(dimension) for dimension in item["shape"])
        path = Path(files[name]) if name in files else None
        binary_file = path is not None and (datatype == "BYTES" or path.suffix.lower() != ".npy")
        encoded_image = binary_file and datatype != "BYTES"
        data = None
        if path is not None and not binary_file:
            data = np.load(path, allow_pickle=False)
            if not isinstance(data, np.ndarray):
                raise ValueError(f"{name}: expected a single .npy array")
        if name in shapes:
            shape = concrete_shape(shapes[name])
        elif encoded_image:
            shape = (1,)
        elif data is not None:
            shape = data.shape
        else:
            shape = model_shape
        if any(dimension <= 0 for dimension in shape):
            raise ValueError(f"{name}: dynamic shape {model_shape}; supply --shape {name}:DIMS")
        if not encoded_image and (len(shape) != len(model_shape) or any(
                expected > 0 and expected != actual for expected, actual in zip(model_shape, shape))):
            raise ValueError(f"{name}: shape {shape} does not match model shape {model_shape}")
        if binary_file:
            payload = path.read_bytes()
            data = np.empty(shape, dtype=object)
            data.fill(payload)
            datatype = "BYTES"
        elif datatype == "BYTES":
            raise ValueError(f"{name}: BYTES input requires --input-file {name}=PATH")
        else:
            dtype = triton_to_np_dtype(datatype)
            if dtype is None:
                raise ValueError(f"{name}: unsupported input datatype {datatype}")
            if data is None:
                data = np.zeros(shape, dtype=dtype)
            elif data.shape != shape or data.dtype != np.dtype(dtype):
                raise ValueError(f"{name}: .npy data must have shape {shape} and dtype {np.dtype(dtype)}")
        prepared.append((name, datatype, np.ascontiguousarray(data).reshape(shape)))
    return prepared


def make_client(args, protocol):
    host = args.host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    url = f"{host}:{args.port}"
    if args.endpoint == "rest":
        return protocol.InferenceServerClient(
            url, connection_timeout=args.timeout, network_timeout=args.timeout)
    return protocol.InferenceServerClient(url)


def initialize_worker(gate):
    global _worker_gate
    _worker_gate = gate


def start_measurement(started):
    started.value = time.perf_counter()


def run_worker(args, protocol_name, prepared, worker_index, worker_count):
    latencies = []
    failures = 0
    errors = []
    intervals = []
    try:
        protocol = importlib.import_module(protocol_name)
        with make_client(args, protocol) as client:
            inputs = []
            for name, datatype, data in prepared:
                tensor = protocol.InferInput(name, list(data.shape), datatype)
                tensor.set_data_from_numpy(data)
                inputs.append(tensor)
            infer_options = {"client_timeout": args.timeout} if args.endpoint == "grpc" else {}
            _worker_gate.wait(timeout=max(60.0, args.timeout))
            finished = time.perf_counter()
            for request_index in range(worker_index, args.requests, worker_count):
                started = time.perf_counter()
                try:
                    client.infer(args.model, inputs, model_version=args.model_version,
                                 request_id=str(request_index), **infer_options)
                except Exception as error:
                    failures += 1
                    if len(errors) < 3:
                        errors.append(str(error))
                else:
                    latencies.append((time.perf_counter() - started) * 1000)
                finally:
                    finished = time.perf_counter()
                    intervals.append((started, finished))
            return latencies, failures, errors, finished, intervals
    except Exception:
        _worker_gate.abort()
        raise


def maximum_concurrency(results):
    events = []
    for result in results:
        for started, finished in result[4]:
            events.append((started, 1))
            events.append((finished, -1))
    active = maximum = 0
    for _, change in sorted(events):
        active += change
        maximum = max(maximum, active)
    return maximum


def summarize(results, started, maximum):
    latencies = [latency for result in results for latency in result[0]]
    failures = sum(result[1] for result in results)
    successes = len(latencies)
    total = successes + failures
    elapsed = max(result[3] for result in results) - started
    latency = dict.fromkeys(("mean", "median", "p90", "p99", "stddev"))
    if latencies:
        latency.update(mean=float(np.mean(latencies)), median=float(np.median(latencies)),
                       p90=float(np.percentile(latencies, 90)), p99=float(np.percentile(latencies, 99)),
                       stddev=float(np.std(latencies)))
    return {
        "requests": total,
        "successful_requests": successes,
        "failed_requests": failures,
        "success_rate_percent": 100 * successes / total if total else 0.0,
        "latency_ms": latency,
        "elapsed_seconds": elapsed,
        "throughput_requests_per_second": successes / elapsed if elapsed > 0 else 0.0,
        "max_concurrency": maximum,
        "errors": list(dict.fromkeys(error for result in results for error in result[2]))[:3],
    }


def benchmark(args, protocol, prepared):
    worker_count = min(args.concurrency, args.requests)
    context = multiprocessing.get_context("spawn")
    started = context.Value("d", 0.0, lock=False)
    gate = context.Barrier(worker_count + 1, action=partial(start_measurement, started))
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context,
                             initializer=initialize_worker, initargs=(gate,)) as executor:
        futures = [executor.submit(run_worker, args, protocol.__name__, prepared, worker_index,
                                   worker_count)
                   for worker_index in range(worker_count)]
        try:
            gate.wait(timeout=max(60.0, args.timeout))
        except threading.BrokenBarrierError:
            for future in futures:
                try:
                    future.result()
                except threading.BrokenBarrierError:
                    continue
            raise RuntimeError("Benchmark worker initialization failed") from None
        results = [future.result() for future in futures]
    report = summarize(results, started.value, maximum_concurrency(results))
    report.update(model=args.model, endpoint=args.endpoint, host=args.host, port=args.port,
                  requested_concurrency=args.concurrency)
    return report


def print_report(report):
    print(f"Number of requests: {report['requests']}")
    print(f"Successful requests: {report['successful_requests']}")
    print(f"Failed requests: {report['failed_requests']}")
    print(f"Success rate: {report['success_rate_percent']:.2f}%")
    for name, label in (("mean", "mean (average)"), ("median", "median (p50)"),
                        ("p90", "p90"), ("p99", "p99"), ("stddev", "stddev")):
        value = report["latency_ms"][name]
        display = f"{value:.3f} ms" if value is not None else "N/A (no successful requests)"
        print(f"Latency {label}: {display}")
    print(f"Throughput: {report['throughput_requests_per_second']:.3f} successful requests/s")
    print(f"Elapsed time: {report['elapsed_seconds']:.3f} s")
    print(f"Max concurrency: {report['max_concurrency']} (requested {report['requested_concurrency']})")
    for error in report["errors"]:
        print(f"Request error: {error}", file=sys.stderr)


def main(argv=None):
    args = parse_args(argv)
    if args.endpoint == "rest":
        import tritonclient.http as protocol
    else:
        import tritonclient.grpc as protocol
    try:
        with make_client(args, protocol) as client:
            options = {"as_json": True, "client_timeout": args.timeout} if args.endpoint == "grpc" else {}
            metadata = client.get_model_metadata(args.model, model_version=args.model_version, **options)
        prepared = prepare_inputs(metadata, args.shape, args.input_file)
        report = benchmark(args, protocol, prepared)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print_report(report)
    return 0 if report["failed_requests"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())