# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Process-level KServe checks with every OpenCV library blocked by the OS loader."""

import json
import os
from pathlib import Path
import socket
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET


def port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def check(root, auditor, executable):
    http_port = port()
    grpc_port = port()
    while grpc_port == http_port:
        grpc_port = port()
    environment = dict(os.environ, LD_AUDIT=str(auditor))
    with tempfile.TemporaryDirectory() as models, tempfile.TemporaryFile(mode="w+") as log:
        version = Path(models) / "1"
        version.mkdir()
        model = ET.parse(root / "src/test/dummy/1/dummy.xml")
        for data in model.iter("data"):
            if data.get("shape") == "1,10":
                data.set("shape", "1,2,2,3")
        for tensor_port in model.iter("port"):
            if [dim.text for dim in tensor_port.findall("dim")] == ["1", "10"]:
                for dim in list(tensor_port):
                    tensor_port.remove(dim)
                for size in (1, 2, 2, 3):
                    ET.SubElement(tensor_port, "dim").text = str(size)
        model.write(version / "dummy.xml")
        shutil.copyfile(root / "src/test/dummy/1/dummy.bin", version / "dummy.bin")
        process = subprocess.Popen(
            [
                str(executable),
                "--model_name", "image",
                "--model_path", models,
                "--layout", "NHWC",
                "--rest_port", str(http_port),
                "--port", str(grpc_port),
                "--allow_unestimatable_image_formats",
                "--log_level", "ERROR",
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        address = f"http://127.0.0.1:{http_port}"
        client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            for _ in range(300):
                if process.poll() is not None:
                    raise AssertionError(f"OVMS exited during startup: {process.returncode}")
                try:
                    with client.open(address + "/v2/health/ready", timeout=1):
                        break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise AssertionError("OVMS did not become ready without OpenCV")

            def infer(encoded):
                data = struct.pack("<I", len(encoded)) + encoded
                header = json.dumps({
                    "inputs": [{
                        "name": "b", "datatype": "BYTES", "shape": [1],
                        "parameters": {"binary_data_size": len(data)},
                    }],
                }).encode()
                return urllib.request.Request(
                    address + "/v2/models/image/infer",
                    data=header + data,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Inference-Header-Content-Length": str(len(header)),
                    },
                )

            image = (root / "src/test/images/rgb.jpg").read_bytes()
            with client.open(infer(image), timeout=10) as response:
                result = json.load(response)
                assert result["outputs"][0]["shape"] == [1, 2, 2, 3], result
                assert len(result["outputs"][0]["data"]) == 12, result

            unsupported = b"Pf\n1 1\n-1.0\n" + struct.pack("<f", 0.5)
            try:
                client.open(infer(unsupported), timeout=10)
            except urllib.error.HTTPError as error:
                detail = error.read().decode()
                assert error.code == 500, (error.code, detail)
                assert "OpenCV" in detail and "missing" in detail, detail
                assert "Install OpenCV" in detail, detail
            else:
                raise AssertionError("An unsupported image unexpectedly succeeded without OpenCV")

            numeric = urllib.request.Request(
                address + "/v2/models/image/infer",
                data=json.dumps({
                    "inputs": [{
                        "name": "b", "datatype": "FP32", "shape": [1, 2, 2, 3],
                        "data": [1.0] * 12,
                    }],
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            with client.open(numeric, timeout=10) as response:
                result = json.load(response)
                assert result["outputs"][0]["data"] == [2.0] * 12, result
            print("PASS: OpenCV-free startup, encoded JPEG inference, explicit missing-OpenCV HTTP error, numeric inference")
        except Exception:
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
