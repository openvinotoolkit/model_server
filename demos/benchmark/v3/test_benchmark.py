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

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


@pytest.mark.parametrize("backend", ["speech2text", "translations"])
@pytest.mark.parametrize("statuses", [(200, 200), (200, 503), (503, 503)])
def test_speech_benchmark_reports_request_outcomes(tmp_path, backend, statuses):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    sample = {"audio": {"array": [0.0] * 160, "sampling_rate": 16000}}
    (dataset / "train.jsonl").write_text(
        (json.dumps(sample) + "\n") * len(statuses), encoding="utf-8"
    )

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]"
    ).save_pretrained(tokenizer_dir)

    responses = iter(statuses)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.rfile.read(int(self.headers["Content-Length"])))
            status = next(responses)
            body = (
                b'{"text": "hello world"}'
                if status == 200
                else b'{"error": "unavailable"}'
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("benchmark.py")),
                    "--backend",
                    backend,
                    "--api_url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--dataset",
                    str(dataset),
                    "--tokenizer",
                    str(tokenizer_dir),
                    "--batch_size",
                    "1",
                ],
                env={
                    **os.environ,
                    "HF_HOME": str(tmp_path / "hf"),
                    "HF_HUB_OFFLINE": "1",
                    "OPENAI_API_KEY": "",
                },
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        finally:
            server.shutdown()
            thread.join()

    successful = statuses.count(200)
    assert len(requests) == len(statuses)
    assert all(b"audio/wav" in request and b"RIFF" in request for request in requests)
    assert result.returncode == 0, result.stderr
    assert (
        f"Success rate: {100.0 * successful / len(statuses)}%. ({successful}/{len(statuses)})"
        in result.stdout
    )
    assert f"Tokens: {2 * successful}\n" in result.stdout
