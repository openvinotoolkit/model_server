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

import json
import os
import runpy
import shlex
import sys
from pathlib import Path

import pytest

pytest.importorskip("jinja2")
pytest.importorskip("huggingface_hub")


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
def test_image_generation_export_honors_overwrite(
    monkeypatch, tmp_path, cached, overwrite
):
    repository = tmp_path / "models"
    destination = repository / "image-model"
    destination.mkdir(parents=True)
    model_index = destination / "model_index.json"
    if cached:
        model_index.write_text("{}")

    commands = []

    def export_model(command):
        commands.append(shlex.split(command, posix=os.name != "nt"))
        model_index.write_text("{}")
        return 0

    monkeypatch.setattr(os, "system", export_model)
    script = (
        Path(__file__).resolve().parents[2]
        / "demos/common/export_models/export_model.py"
    )
    config = tmp_path / "config.json"
    argv = [
        str(script),
        "image_generation",
        "--source_model",
        "example/image-model",
        "--model_name",
        "image-model",
        "--model_repository_path",
        str(repository),
        "--config_file_path",
        str(config),
    ]
    if overwrite:
        argv.append("--overwrite_models")
    monkeypatch.setattr(sys, "argv", argv)

    runpy.run_path(str(script), run_name="__main__")

    if overwrite or not cached:
        assert commands == [
            [
                "optimum-cli",
                "export",
                "openvino",
                "--model",
                "example/image-model",
                "--weight-format",
                "int8",
                str(destination),
            ]
        ]
    else:
        assert commands == []
    assert (destination / "graph.pbtxt").is_file()
    assert (
        json.loads(config.read_text())["model_config_list"][0]["config"]["name"]
        == "image-model"
    )
