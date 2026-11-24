#!/usr/bin/env python3
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
"""Compose a custom OVMS docker image from a full OVMS image.

By default the resulting image supports only CPU, IR models and no Python.
Optional features copy the relevant artifacts from the full image and install
the required system dependencies.

Example:
    python3 tools/docker_image_composer/compose.py --onnx --opencv -o ovms-custom:onnx
"""

import argparse
import os
import re
import shutil
import shlex
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCKERFILE_NAME = "Dockerfile.compose"
GPU_DRIVERS_SCRIPT = "install_ubuntu_gpu_drivers.sh"
LEVEL_ZERO_LOADER_SCRIPT = "install_ubuntu_level_zero_loader.sh"
DEFAULT_FULL_IMAGE = "openvino/model_server:weekly"

SUPPORTED_UBUNTU = {"22.04": "ubuntu22", "24.04": "ubuntu24", "26.04": "ubuntu26"}

# Artifacts in /ovms/lib removed unless the corresponding feature is requested.
GPU_LIBS = ["libopenvino_intel_gpu_plugin.so*"]
OPENCV_LIBS = ["libopencv*.so*"]
NPU_LIBS = [
    "libopenvino_intel_npu_plugin.so*",
    "libopenvino_intel_npu_compiler.so*",
    "libopenvino_intel_npu_compiler_loader.so*",
    "libopenvino_intel_npu_vm_runtime.so*",
]
TENSORFLOW_LIBS = ["libopenvino_tensorflow_frontend.so*", "libopenvino_tensorflow_lite_frontend.so*"]
PADDLE_LIBS = ["libopenvino_paddle_frontend.so*"]
ONNX_LIBS = ["libopenvino_onnx_frontend.so*"]
PYTHON_LIBS = ["python", "pyovms.so", "libovmspython.so*", "libpython_calculators.so*"]
PYTHON_DIRS = ["/ovms/python_deps"]
ESPEAK_LIBS = ["libespeak-ng.so*"]
ESPEAK_DIRS = ["/ovms/share/espeak-ng-data"]

ENTRYPOINT = (
    'ENTRYPOINT ["/bin/bash", "-c", "if [ -d /ovms/share/espeak-ng-data ]; then export ESPEAK_DATA_PATH=/ovms/share/espeak-ng-data; fi; '
    'if [ -z \\"$OPENCV_IO_MAX_IMAGE_PIXELS\\" ]; then export OPENCV_IO_MAX_IMAGE_PIXELS=67108864; fi; '
    'exec /ovms/bin/ovms \\"$@\\"", "--"]'
)


def log(msg):
    print(msg, file=sys.stderr)


def fail(msg):
    print("error: {}".format(msg), file=sys.stderr)
    sys.exit(1)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def inspect_full_image(image, skip_pull):
    if not skip_pull:
        log("pulling image: {}".format(image))
        p = subprocess.run(["docker", "pull", image])
        if p.returncode != 0:
            fail("docker pull {} failed".format(image))

    p = run(["docker", "inspect", "-f", '{{ index .Config.Labels "base-image" }}', image])
    if p.returncode != 0:
        fail("docker inspect {} failed: {}".format(image, p.stderr))
    base_image = p.stdout.strip()
    if base_image == "<no value>":
        base_image = ""

    p = run(["docker", "run", "--rm", "--user", "root", "--entrypoint", "bash", image, "-c",
             "cat /etc/os-release; echo '---LDD---'; ldd /ovms/bin/ovms; echo '---LDDPY---'; "
             "if [ -f /ovms/lib/libovmspython.so ]; then ldd /ovms/lib/libovmspython.so; fi"])
    if p.returncode != 0:
        fail("failed to inspect content of {}: {}".format(image, p.stderr))
    os_release, ldd_output = p.stdout.split("---LDD---", 1)
    ldd_ovms, ldd_python = ldd_output.split("---LDDPY---", 1)

    os_id = re.search(r'^ID="?([^"\n]+)"?', os_release, re.MULTILINE)
    os_version = re.search(r'^VERSION_ID="?([^"\n]+)"?', os_release, re.MULTILINE)
    if os_id is None or os_version is None:
        fail("cannot determine OS of the full image")
    python_version = re.search(r"libpython(\d+\.\d+)", ldd_output)
    return {
        "base_image": base_image,
        "os_id": os_id.group(1),
        "os_version": os_version.group(1),
        "python_version": python_version.group(1) if python_version else None,
        # Older builds link libpython into the ovms binary, so Python cannot be dropped.
        "python_required": "libpython" in ldd_ovms,
        "opencv_required": "libopencv" in ldd_ovms,
    }


def filter_stage(full_image, flags):
    removed = []
    if not flags.gpu:
        removed += GPU_LIBS
    if not flags.opencv:
        removed += OPENCV_LIBS
    if not flags.npu:
        removed += NPU_LIBS
    if not flags.tensorflow:
        removed += TENSORFLOW_LIBS
    if not flags.paddle:
        removed += PADDLE_LIBS
    if not flags.onnx:
        removed += ONNX_LIBS
    if not flags.python:
        removed += PYTHON_LIBS
    if not flags.espeak:
        removed += ESPEAK_LIBS

    df = "FROM {} AS full\nUSER root\n".format(full_image)
    paths = ["/ovms/lib/{}".format(name) for name in removed]
    if not flags.python:
        paths += PYTHON_DIRS
    if not flags.espeak:
        paths += ESPEAK_DIRS
    if paths:
        df += "RUN rm -rf {}\n".format(" ".join(paths))
    # Packaging dereferences symlinks, so identical library copies are turned back into symlinks.
    df += """RUN cd /ovms/lib && find . -maxdepth 1 -type f -name '*.so*' -exec sha256sum {} + | sort | \\
    awk '{ if ($1 == h) print f, $2; else { h = $1; f = $2 } }' | \\
    while read -r target dup; do ln -sf "${target#./}" "$dup"; done
"""
    return df


def release_stage(base_image, info, flags):
    os_version = info["os_version"]
    base_os = SUPPORTED_UBUNTU[os_version]
    xml_package = "libxml2-16" if os_version == "26.04" else "libxml2"
    df = """
FROM {base} AS release
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates {xml_package} && \\
    apt-get clean && rm -rf /var/lib/apt/lists/*
""".format(base=base_image, xml_package=xml_package)

    if flags.gpu or flags.npu:
        df += "COPY versions.mk /usr/local/share/ovms-versions.mk\n"

    if flags.gpu:
        df += """
COPY {script} /tmp/install_gpu_drivers.sh
RUN BASE_OS={base_os} INSTALL_DRIVER_VERSION={version} /tmp/install_gpu_drivers.sh && rm -f /tmp/install_gpu_drivers.sh
""".format(script=GPU_DRIVERS_SCRIPT, base_os=base_os, version=shlex.quote(flags.gpu_driver_version or ""))

    if flags.gpu or flags.npu:
        df += """
COPY {script} /tmp/install_level_zero_loader.sh
RUN BASE_OS={base_os} NPU={npu} bash /tmp/install_level_zero_loader.sh && \\
    rm -f /tmp/install_level_zero_loader.sh /usr/local/share/ovms-versions.mk
""".format(script=LEVEL_ZERO_LOADER_SCRIPT, base_os=base_os, npu=int(flags.npu))

    if flags.python:
        df += """
RUN apt-get update && apt-get install -y --no-install-recommends libpython{version} && \\
    apt-get clean && rm -rf /var/lib/apt/lists/*
""".format(version=info["python_version"])

    df += """
RUN rm -rf /tmp/* && groupadd --gid 5000 ovms && \\
    useradd --home-dir /home/ovms --create-home --uid 5000 --gid 5000 --groups 39,44 --shell /bin/bash --skel /dev/null ovms

COPY --from=full /ovms /ovms
ENV LD_LIBRARY_PATH=/ovms/lib
ENV PATH="$PATH:/ovms/bin"
"""
    if flags.python:
        df += "ENV PYTHONPATH=/ovms/lib/python:/ovms/python_deps\n"

    df += """LABEL supported-devices="CPU=1 GPU={gpu} NPU={npu}"
LABEL base-image={base}
LABEL full-image={full}
{entrypoint}
USER ovms
""".format(gpu=int(flags.gpu), npu=int(flags.npu), base=base_image, full=flags.full_image, entrypoint=ENTRYPOINT)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compose a custom OVMS docker image with selected features from a full OVMS image. "
                    "By default all optional features are disabled. Approximate size increases are measured "
                    "against the default image from openvino/model_server-gpu:latest on Ubuntu 24.04.")
    parser.add_argument("--full-image", default=DEFAULT_FULL_IMAGE,
                        help="Source OVMS Docker image (default: %(default)s).")
    parser.add_argument("-o", "--output-name", default="ovms-custom", help="Name of the generated image. Default: ovms-custom")
    parser.add_argument("--base-image", help="Base image for the generated image. Default: base image of the full image.")
    parser.add_argument("--gpu", action="store_true", help="Adds Intel GPU support and drivers (about +500 MB).")
    parser.add_argument("--gpu-driver-version", help="GPU driver version passed to {}.".format(GPU_DRIVERS_SCRIPT))
    parser.add_argument("--npu", action="store_true", help="Adds Intel NPU support, compiler, and drivers (about +300 MB).")
    parser.add_argument("--opencv", action="store_true", help="Keeps upstream OpenCV shared libraries (about +50 MB).")
    parser.add_argument("--python", action="store_true", help="Adds Python runtime and custom-node support (about +95 MB).")
    parser.add_argument("--tensorflow", action="store_true", help="Enables TensorFlow and TensorFlow Lite model loading (about +5 MB).")
    parser.add_argument("--paddle", action="store_true", help="Enables PaddlePaddle model loading (about +2 MB).")
    parser.add_argument("--onnx", action="store_true", help="Enables ONNX model loading (about +5 MB).")
    parser.add_argument("--espeak", action="store_true", help="Adds eSpeak library and data for Kokoro TTS (about +10 MB).")
    parser.add_argument("--work-dir", default=".", help="Directory for the generated Dockerfile and build context. Default: current directory.")
    parser.add_argument("--skip-pull", action="store_true", help="Do not pull the full image.")
    parser.add_argument("--dry-run", action="store_true", help="Only generate {}, do not build the image.".format(DOCKERFILE_NAME))
    flags = parser.parse_args()

    info = inspect_full_image(flags.full_image, flags.skip_pull)
    log("full image OS: {} {}, python: {}".format(info["os_id"], info["os_version"], info["python_version"]))

    if info["os_id"] != "ubuntu" or info["os_version"] not in SUPPORTED_UBUNTU:
        fail("only Ubuntu {} full images are supported".format(", ".join(SUPPORTED_UBUNTU)))
    if info["python_required"] and not flags.python:
        fail("ovms in {} is linked with libpython{}; use --python or a full image built without Python (py_off)".format(
            flags.full_image, info["python_version"]))
    if info["opencv_required"] and not flags.opencv:
        fail("ovms in {} depends on OpenCV libraries; use --opencv".format(flags.full_image))
    if flags.python and not info["python_version"]:
        fail("--python requested but ovms in {} is built without Python support".format(flags.full_image))

    base_image = flags.base_image or info["base_image"] or "ubuntu:{}".format(info["os_version"])

    os.makedirs(flags.work_dir, exist_ok=True)
    scripts = []
    if flags.gpu:
        scripts.append(GPU_DRIVERS_SCRIPT)
    if flags.gpu or flags.npu:
        scripts.extend([LEVEL_ZERO_LOADER_SCRIPT, "versions.mk"])
    for script in scripts:
        src = os.path.join(REPO_ROOT, script)
        dst = os.path.join(flags.work_dir, script)
        if not (os.path.exists(dst) and os.path.samefile(src, dst)):
            shutil.copy(src, dst)

    dockerfile_path = os.path.join(flags.work_dir, DOCKERFILE_NAME)
    with open(dockerfile_path, "w") as dockerfile:
        dockerfile.write(filter_stage(flags.full_image, flags))
        dockerfile.write(release_stage(base_image, info, flags))
    log("generated {}".format(dockerfile_path))

    if flags.dry_run:
        return
    proxy_args = []
    for name in ("http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        if os.environ.get(name):
            proxy_args += ["--build-arg", name]
    p = subprocess.run(["docker", "buildx", "build"] + proxy_args +
                       ["-t", flags.output_name, "-f", dockerfile_path, flags.work_dir])
    if p.returncode != 0:
        fail("docker build {} failed".format(flags.output_name))
    log("built image {}".format(flags.output_name))


if __name__ == "__main__":
    main()
