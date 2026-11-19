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

import logging
import os
import re

from dataclasses import dataclass

from tests.functional.constants.target_device import TargetDevice

logger = logging.getLogger(__name__)

GPU_DEVICES_ENV = "OVMS_TESTS_GPU_DEVICES"
RESOLVED_TARGET_DEVICES_ENV = "OVMS_TESTS_RESOLVED_TARGET_DEVICES"
_GPU_RE = re.compile(rf"\b{TargetDevice.GPU}(?:\.\d+)?\b")
_GPU_WITHOUT_INDEX_RE = re.compile(rf"\b{TargetDevice.GPU}\b(?!\.\d)")


@dataclass(frozen=True)
class GpuDevice:
    name: str
    full_name: str
    is_discrete: bool
    device_id: str

    def describe(self):
        return f"{self.name}: {self.full_name} type={'dGPU' if self.is_discrete else 'iGPU'} device_id={self.device_id}"


def get_gpu_devices():
    """ Return GPUs in OpenVINO order, where an integrated GPU is always the first one. """
    import openvino as ov  # pylint: disable=import-outside-toplevel, import-error
    from openvino.properties.device import Type  # pylint: disable=import-outside-toplevel, import-error

    core = ov.Core()
    gpu_devices = []
    for name in core.available_devices:
        if not _GPU_RE.fullmatch(name):
            continue
        try:
            device_id = str(core.get_property(name, "GPU_DEVICE_ID"))
        except RuntimeError:  # not reported by older OpenVINO, e.g. 2024.5
            device_id = ""
        is_discrete = core.get_property(name, "DEVICE_TYPE") == Type.DISCRETE
        gpu_devices.append(GpuDevice(name, core.get_property(name, "FULL_DEVICE_NAME"), is_discrete, device_id))
    return gpu_devices


def get_gpu_names(target_device):
    return _GPU_RE.findall(target_device)


def find_gpu(gpu_devices, name):
    """ OpenVINO names the only GPU of a host 'GPU', and 'GPU' is 'GPU.0' on hosts with more GPUs. """
    if name == TargetDevice.GPU or (name == f"{TargetDevice.GPU}.0" and len(gpu_devices) == 1):
        return gpu_devices[0] if gpu_devices else None
    return next((gpu for gpu in gpu_devices if gpu.name == name), None)


def resolve_default_gpu(target_devices):
    """ Point GPU without index at the first discrete GPU, e.g. 'AUTO:GPU,CPU' -> 'AUTO:GPU.1,CPU'. """
    if not any(get_gpu_names(_td) for _td in target_devices):
        return list(target_devices)
    try:
        gpu_devices = get_gpu_devices()
    except (ImportError, RuntimeError) as exc:
        logger.warning(f"Cannot list GPUs with OpenVINO, GPU target devices are used as given: {exc}")
        return list(target_devices)

    resolved = list(target_devices)
    discrete_gpu = next((gpu for gpu in gpu_devices if gpu.is_discrete), None)
    if discrete_gpu is not None and discrete_gpu != gpu_devices[0]:
        resolved = [_GPU_WITHOUT_INDEX_RE.sub(discrete_gpu.name, _td) for _td in target_devices]
        assert len(set(resolved)) == len(resolved), f"Target devices {target_devices} resolve to duplicates: {resolved}"
        logger.info(f"GPU without index resolved to {discrete_gpu.describe()}")
    # The test run report stores os.environ, so it shows which GPU was tested.
    os.environ[GPU_DEVICES_ENV] = "; ".join(gpu.describe() for gpu in gpu_devices)
    os.environ[RESOLVED_TARGET_DEVICES_ENV] = ",".join(resolved)
    return resolved
