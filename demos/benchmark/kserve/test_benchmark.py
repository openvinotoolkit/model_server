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
"""Focused tests for benchmark inputs, measurement, and worker coordination."""

from contextlib import redirect_stderr
import io
import multiprocessing
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

import benchmark


class SpawnTestClient:
    def __init__(self, url, **kwargs):
        if url.startswith("startup-fail:"):
            raise ValueError("initialization failed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def infer(self, *args, **kwargs):
        if multiprocessing.current_process().name == "MainProcess":
            raise RuntimeError("Inference must run in a child process")


InferenceServerClient = SpawnTestClient


class InputTests(unittest.TestCase):
    def metadata(self, shape, datatype="FP32", name="image"):
        return {"inputs": [{"name": name, "datatype": datatype, "shape": shape}]}

    def test_static_numeric_input(self):
        inputs = benchmark.prepare_inputs(self.metadata([1, 2]), [], [])
        self.assertEqual(inputs[0][:2], ("image", "FP32"))
        np.testing.assert_array_equal(inputs[0][2], np.zeros((1, 2), dtype=np.float32))

    def test_dynamic_shape_requires_override(self):
        with self.assertRaisesRegex(ValueError, "dynamic shape"):
            benchmark.prepare_inputs(self.metadata([-1, 2]), [], [])
        inputs = benchmark.prepare_inputs(self.metadata(["-1", "2"]), ["3,2"], [])
        self.assertEqual(inputs[0][2].shape, (3, 2))

    def test_static_dimension_cannot_be_overridden(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            benchmark.prepare_inputs(self.metadata([-1, 2]), ["image:1,3"], [])

    def test_multiple_inputs(self):
        metadata = self.metadata([-1, 2])
        metadata["inputs"].append({"name": "mask", "datatype": "INT64", "shape": [-1]})
        inputs = benchmark.prepare_inputs(metadata, ["image:3,2", "mask:3"], [])
        self.assertEqual([item[2].shape for item in inputs], [(3, 2), (3,)])
        self.assertEqual(inputs[1][2].dtype, np.int64)
        with self.assertRaisesRegex(ValueError, "Specify an input name"):
            benchmark.prepare_inputs(metadata, ["3,2"], [])

    def test_unknown_and_duplicate_inputs(self):
        for shapes in (["unknown:1"], ["image:1,2", "image:1,2"]):
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                benchmark.prepare_inputs(self.metadata([1, 2]), shapes, [])

    def test_invalid_shapes(self):
        for shape in ("1,0", "-1,2", "1,x", ""):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                benchmark.prepare_inputs(self.metadata([-1, 2]), [shape], [])

    def test_bytes_file_preserves_nulls_and_repeats_elements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpeg"
            payload = b"\xff\xd8image\x00\x00"
            path.write_bytes(payload)
            inputs = benchmark.prepare_inputs(self.metadata([-1], "BYTES"), ["2"], [str(path)])
            self.assertEqual(inputs[0][2].tolist(), [payload, payload])
            self.assertEqual(inputs[0][2].dtype, object)

    def test_multiple_bytes_files(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.bin"
            second = Path(directory) / "second.bin"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            metadata = self.metadata([1], "BYTES", "first")
            metadata["inputs"].extend(self.metadata([1], "BYTES", "second")["inputs"])
            inputs = benchmark.prepare_inputs(metadata, [], [f"first={first}", f"second={second}"])
            self.assertEqual(inputs[0][2].item(), b"first")
            self.assertEqual(inputs[1][2].item(), b"second")

    def test_encoded_image_uses_bytes_and_batch_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpeg"
            path.write_bytes(b"jpeg")
            inputs = benchmark.prepare_inputs(self.metadata([1, 224, 224, 3]), [], [str(path)])
            self.assertEqual(inputs[0][1], "BYTES")
            self.assertEqual(inputs[0][2].shape, (1,))

    def test_bytes_requires_file(self):
        with self.assertRaisesRegex(ValueError, "requires --input-file"):
            benchmark.prepare_inputs(self.metadata([1], "BYTES"), [], [])

    def test_scalar_shape(self):
        inputs = benchmark.prepare_inputs(self.metadata([], "INT32"), ["scalar"], [])
        self.assertEqual(inputs[0][2].shape, ())

    def test_numeric_file_shape_and_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.npy"
            np.save(path, np.ones((3, 2), dtype=np.float32))
            inputs = benchmark.prepare_inputs(self.metadata([-1, 2]), [], [str(path)])
            self.assertEqual(inputs[0][2].shape, (3, 2))
            np.save(path, np.ones((3, 2), dtype=np.int32))
            with self.assertRaisesRegex(ValueError, "dtype float32"):
                benchmark.prepare_inputs(self.metadata([-1, 2]), [], [str(path)])


class MeasurementTests(unittest.TestCase):
    def test_metrics_and_partial_failure(self):
        report = benchmark.summarize([([1.0, 2.0, 3.0], 1, ["failed"], 12.0)], 10.0, 2)
        self.assertEqual(report["requests"], 4)
        self.assertEqual(report["success_rate_percent"], 75.0)
        self.assertEqual(report["throughput_requests_per_second"], 1.5)
        self.assertEqual(report["latency_ms"]["mean"], 2.0)
        self.assertEqual(report["latency_ms"]["median"], 2.0)
        self.assertAlmostEqual(report["latency_ms"]["p90"], 2.8)
        self.assertAlmostEqual(report["latency_ms"]["p99"], 2.98)
        self.assertAlmostEqual(report["latency_ms"]["stddev"], np.std([1, 2, 3]))
        self.assertEqual(report["max_concurrency"], 2)

    def test_all_failures_have_no_latency(self):
        report = benchmark.summarize([([], 3, ["failed"], 11.0)], 10.0, 1)
        self.assertEqual(report["success_rate_percent"], 0.0)
        self.assertEqual(report["throughput_requests_per_second"], 0.0)
        self.assertTrue(all(value is None for value in report["latency_ms"].values()))

    def test_spawn_workers_and_request_count(self):
        args = benchmark.parse_args(["-m", "test", "--requests", "6", "--concurrency", "3"])
        report = benchmark.benchmark(args, sys.modules[__name__], [])
        self.assertEqual(report["requests"], 6)
        self.assertEqual(report["successful_requests"], 6)
        self.assertGreaterEqual(report["max_concurrency"], 1)
        self.assertLessEqual(report["max_concurrency"], 3)

    def test_peak_concurrency_from_request_intervals(self):
        results = [([], 0, [], 6.0, [(1.0, 4.0), (5.0, 6.0)]),
                   ([], 0, [], 3.0, [(2.0, 3.0)]),
                   ([], 0, [], 5.0, [(2.5, 4.0), (4.0, 5.0)])]
        self.assertEqual(benchmark.maximum_concurrency(results), 3)

    def test_concurrency_is_capped_by_request_count(self):
        args = benchmark.parse_args(["-m", "test", "--requests", "1", "--concurrency", "20"])
        report = benchmark.benchmark(args, sys.modules[__name__], [])
        self.assertEqual(report["requests"], 1)
        self.assertEqual(report["max_concurrency"], 1)
        self.assertEqual(report["requested_concurrency"], 20)

    def test_request_exceptions_are_counted(self):
        client = unittest.mock.MagicMock()
        client.__enter__.return_value = client
        client.infer.side_effect = [None, RuntimeError("failed"), None]
        args = benchmark.parse_args(["-m", "test", "--requests", "3"])
        benchmark.initialize_worker(threading.Barrier(1))
        with patch.object(benchmark, "make_client", return_value=client):
            result = benchmark.run_worker(args, "tritonclient.grpc", [], 0, 1)
        report = benchmark.summarize([result], result[4][0][0], benchmark.maximum_concurrency([result]))
        self.assertEqual(report["requests"], 3)
        self.assertEqual(report["successful_requests"], 2)
        self.assertEqual(report["failed_requests"], 1)
        self.assertEqual(report["errors"], ["failed"])
        self.assertEqual(len(result[4]), 3)

    def test_worker_initialization_failure_does_not_deadlock(self):
        args = benchmark.parse_args(["-m", "test", "--host", "startup-fail",
                                     "--requests", "2", "--concurrency", "2"])
        with self.assertRaisesRegex(ValueError, "initialization failed"):
            benchmark.benchmark(args, sys.modules[__name__], [])

    def test_invalid_cli_limits(self):
        for flag, value in (("--requests", "0"), ("--concurrency", "-1"),
                            ("--port", "65536"), ("--timeout", "nan")):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark.parse_args(["-m", "test", flag, value])


if __name__ == "__main__":
    unittest.main()