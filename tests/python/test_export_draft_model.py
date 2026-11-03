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


@pytest.mark.parametrize(
    "source_model,draft_source_model,expected_tool",
    [
        ("OpenVINO/target", "OpenVINO/draft", "huggingface-cli"),
        ("example/target", "OpenVINO/draft", "huggingface-cli"),
        ("local", "OpenVINO/draft", "huggingface-cli"),
        ("OpenVINO/target", "example/draft", "optimum-cli"),
        ("example/target", "example/draft", "optimum-cli"),
        ("OpenVINO/target", "OpenVINO/draft", None),
    ],
    ids=[
        "converted-pair",
        "converted-draft",
        "local-target",
        "unconverted-draft",
        "unconverted-pair",
        "cached-draft",
    ],
)
def test_export_uses_draft_source_model(
    monkeypatch, tmp_path, source_model, draft_source_model, expected_tool
):
    model_repository = tmp_path / "models"
    model_repository.mkdir()
    if source_model == "local":
        local_model = tmp_path / "local"
        local_model.mkdir()
        (local_model / "openvino_model.xml").touch()
        source_model = str(local_model)

    draft_directory = (
        model_repository / "servable" / draft_source_model.replace("/", "-")
    )
    if expected_tool is None:
        draft_directory.mkdir(parents=True)
        (draft_directory / "openvino_model.xml").touch()

    commands = []

    def record_command(command):
        commands.append(shlex.split(command, posix=os.name != "nt"))
        return 0

    monkeypatch.setattr(os, "system", record_command)
    script = (
        Path(__file__).resolve().parents[2]
        / "demos/common/export_models/export_model.py"
    )
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "text_generation",
            "--source_model",
            source_model,
            "--draft_source_model",
            draft_source_model,
            "--model_name",
            "servable",
            "--model_repository_path",
            str(model_repository),
            "--config_file_path",
            str(config_path),
        ],
    )
    runpy.run_path(str(script), run_name="__main__")

    if expected_tool is None:
        assert commands == [
            [
                "huggingface-cli",
                "download",
                source_model,
                "--local-dir",
                str(model_repository / "servable"),
            ]
        ]
    elif expected_tool == "huggingface-cli":
        assert commands[-1] == [
            "huggingface-cli",
            "download",
            draft_source_model,
            "--local-dir",
            str(draft_directory),
        ]
    else:
        assert commands[-1] == [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            draft_source_model,
            "--weight-format",
            "int8",
            "--trust-remote-code",
            str(draft_directory),
        ]

    graph = (model_repository / "servable" / "graph.pbtxt").read_text()
    assert f'draft_models_path: "./{draft_directory.name}"' in graph
    config = json.loads(config_path.read_text())
    assert config["model_config_list"] == [
        {"config": {"name": "servable", "base_path": "models/servable"}}
    ]
