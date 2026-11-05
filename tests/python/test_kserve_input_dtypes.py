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


@pytest.mark.parametrize("protocol", ["grpc", "http"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("dtype", ["uint8", "float64", "float32"])
@pytest.mark.parametrize("transpose", [False, True])
def test_numpy_sample_sends_fp32(
    monkeypatch, tmp_path, protocol, asynchronous, dtype, transpose
):
    client = pytest.importorskip(f"tritonclient.{protocol}")
    root = Path(__file__).resolve().parents[2]
    samples = root / "client" / "python" / "kserve-api" / "samples"
    sample = samples / f"{protocol}_{'async_' if asynchronous else ''}infer_resnet.py"
    images = np.arange(36, dtype=dtype).reshape(3, 2, 2, 3)
    image_file = tmp_path / "images.npy"
    np.save(image_file, images)
    sent = []
    set_data = client.InferInput.set_data_from_numpy

    def record_input(tensor, array, *args, **kwargs):
        set_data(tensor, array, *args, **kwargs)
        sent.append(array.copy())

    class InputAccepted(Exception):
        pass

    def stop_before_network(**kwargs):
        raise InputAccepted

    monkeypatch.setattr(client.InferInput, "set_data_from_numpy", record_input)
    monkeypatch.setattr(
        client,
        "InferenceServerClient",
        lambda **kwargs: SimpleNamespace(
            infer=stop_before_network, async_infer=stop_before_network
        ),
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
            "1",
            "--transpose_input",
            str(transpose),
        ],
    )
    with pytest.raises(InputAccepted):
        runpy.run_path(str(sample), run_name="__main__")

    assert len(sent) == 1
    assert sent[0].dtype == np.float32
    expected = images.astype(np.float32) / 35 * 255
    if transpose:
        expected = expected.transpose(0, 3, 1, 2)
    np.testing.assert_allclose(sent[0], expected[:1])
