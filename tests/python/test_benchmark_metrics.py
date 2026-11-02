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

import pytest

from demos.benchmark.python.metrics import XMetrics, XSeries


@pytest.mark.parametrize("prefix", ["", "window_", "warmup_"])
@pytest.mark.parametrize(
    "passed_latencies,failed_latencies,expected_pass_cv,expected_fail_cv",
    [
        ([2.0, 2.0], [1.0, 3.0], 0.0, 0.5),
        ([1.0, 3.0], [2.0, 2.0], 0.5, 0.0),
        ([], [1.0, 3.0], 0.0, 0.5),
        ([1.0, 3.0], [], 0.5, 0.0),
    ],
)
def test_aggregated_latency_variation_uses_matching_request_outcomes(
    prefix, passed_latencies, failed_latencies, expected_pass_cv, expected_fail_cv
):
    combined = XMetrics(submetrics=0)
    for _ in range(2):
        worker_metrics = {}
        for series_prefix in ("", "window", "warmup"):
            series = XSeries(series_prefix)
            series.start_timestamp = 0.0
            series.stop_timestamp = 1.0
            for latency in passed_latencies:
                series.add(True, latency, 1)
            for latency in failed_latencies:
                series.add(False, latency, 1)
            worker_metrics.update(series.analyze())
        combined += XMetrics(worker_metrics)

    assert combined[f"{prefix}pass_cv_latency"] == pytest.approx(expected_pass_cv)
    assert combined[f"{prefix}fail_cv_latency"] == pytest.approx(expected_fail_cv)
