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
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
httpclient = pytest.importorskip("tritonclient.http")


@pytest.mark.parametrize(
    "image_count,batch_size,iterations,expected_requests",
    [(4, 1, 1, 1), (6, 2, 0, 3), (4, 1, 6, 6), (4, 1, 0, 4)],
    ids=["short-run", "batched", "partial-last-round", "full-round"],
)
def test_http_async_consumes_each_submitted_result_once(
    monkeypatch, tmp_path, image_count, batch_size, iterations, expected_requests
):
    root = Path(__file__).resolve().parents[2]
    samples = root / "client" / "python" / "kserve-api" / "samples"
    sample = samples / "http_async_infer_resnet.py"
    image_file = tmp_path / "images.npy"
    np.save(
        image_file, np.arange(image_count * 3, dtype=np.float32).reshape(-1, 1, 1, 3)
    )
    requests = []
    consumed = []

    def async_infer(**kwargs):
        index = len(requests)
        requests.append(kwargs)

        def get_result():
            consumed.append(index)
            output = np.tile([[1.0, 0.0]], (batch_size, 1))
            return SimpleNamespace(as_numpy=lambda name: output)

        return SimpleNamespace(get_result=get_result)

    monkeypatch.setattr(
        httpclient,
        "InferenceServerClient",
        lambda **kwargs: SimpleNamespace(async_infer=async_infer),
    )
    monkeypatch.syspath_prepend(str(root / "demos" / "common" / "python"))
    monkeypatch.chdir(samples)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(sample),
            "--images_numpy_path",
            str(image_file),
            "--iterations",
            str(iterations),
            "--batchsize",
            str(batch_size),
            "--transpose_input",
            "False",
        ],
    )
    runpy.run_path(str(sample), run_name="__main__")

    assert len(requests) == expected_requests
    assert consumed == list(range(expected_requests))
    assert all(request["inputs"][0].shape()[0] == batch_size for request in requests)
