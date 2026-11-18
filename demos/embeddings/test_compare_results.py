# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import runpy
import sys
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("service_path", "request_path"),
    [
        ("/v3/embeddings", "/v3/embeddings"),
        ("/proxy/v3/embeddings/", "/proxy/v3/embeddings"),
        ("/proxy/v3/", "/proxy/v3/embeddings"),
    ],
)
def test_compare_results_service_url(monkeypatch, capsys, service_path, request_path):
    # Use a fixed reference embedding so the CLI can run without a model download.
    hidden_state = np.array([[[1.0, 0.0]]], dtype=np.float32)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=Mock(
                from_pretrained=Mock(return_value=Mock(return_value={}))
            ),
            AutoModel=Mock(
                from_pretrained=Mock(
                    return_value=Mock(
                        return_value=SimpleNamespace(last_hidden_state=hidden_state)
                    )
                )
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "optimum.intel",
        SimpleNamespace(OVModelForFeatureExtraction=None, OVSentenceTransformer=None),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            no_grad=nullcontext,
            nn=SimpleNamespace(
                functional=SimpleNamespace(
                    normalize=lambda embeddings, **kwargs: embeddings
                )
            ),
        ),
    )

    class EmbeddingsHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == request_path:
                self.send_response(200)
                payload = {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}
                    ],
                    "model": request["model"],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            else:
                self.send_response(404)
                payload = {
                    "error": {"message": f"Unexpected request path: {self.path}"}
                }
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

    script = Path(__file__).with_name("compare_results.py")
    with HTTPServer(("127.0.0.1", 0), EmbeddingsHandler) as server:
        worker = Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        worker.start()
        try:
            service_url = f"http://127.0.0.1:{server.server_port}{service_path}"
            monkeypatch.setattr(
                sys,
                "argv",
                [str(script), "--service_url", service_url, "--input", "hello"],
            )
            runpy.run_path(str(script), run_name="__main__")
        finally:
            server.shutdown()
            worker.join()

    assert "[SUCCESS]" in capsys.readouterr().out
