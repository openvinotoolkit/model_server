#
# Copyright (c) 2023 Intel Corporation
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

from enum import Enum, auto


class OvmsImageType(Enum):
    UBUNTU20 = auto()
    UBUNTU22 = auto()
    UBUNTU22_GPU = auto()
    UBUNTU22_NGINX = auto()
    UBUNTU24 = auto()
    UBUNTU24_GPU = auto()
    UBUNTU24_NGINX = auto()
    REDHAT = auto()
    REDHAT_GPU = auto()
    REDHAT_CUDA = auto()


class OvmsBaseImageType(Enum):
    COMMON = "common"
    UBUNTU = "ubuntu"
    UBUNTU20 = "ubuntu20"
    UBUNTU22 = "ubuntu22"
    UBUNTU24 = "ubuntu24"
    UBUNTU_PYTHON = "ubuntu_python"
    UBUNTU20_PYTHON = "ubuntu20_python"
    UBUNTU22_PYTHON = "ubuntu22_python"
    UBUNTU24_PYTHON = "ubuntu24_python"
    UBUNTU_GPU = "ubuntu_gpu"
    UBUNTU_NGINX = "ubuntu_nginx"
    REDHAT = "redhat"
    REDHAT_PYTHON = "redhat_python"
    REDHAT_GPU = "redhat_gpu"


# Libraries listed by ldd /ovms/bin/ovms
dynamic_libraries = {
    OvmsBaseImageType.COMMON: {
        'libgcc_s.so',
        'liblzma.so',
        'libstdc++.so',
        'libuuid.so',
        'libxml2.so',
    },
    OvmsBaseImageType.UBUNTU: {'libicuuc.so', 'libicudata.so',},
    OvmsBaseImageType.UBUNTU_PYTHON: {'libexpat.so',},
    OvmsBaseImageType.UBUNTU20: {'librt.so',},
    OvmsBaseImageType.UBUNTU20_PYTHON: {'libpython3.8.so', 'libutil.so',},
    OvmsBaseImageType.UBUNTU22: {'libdl.so', 'libm.so', 'libpthread.so',},
    OvmsBaseImageType.UBUNTU22_PYTHON: {'libpython3.10.so',},
    OvmsBaseImageType.UBUNTU24: {'libdl.so', 'libm.so', 'libpthread.so',},
    OvmsBaseImageType.UBUNTU24_PYTHON: {'libpython3.12.so',},
    OvmsBaseImageType.REDHAT: {'libdl.so', 'libm.so', 'libpthread.so', 'libcrypt.so',},
    OvmsBaseImageType.REDHAT_PYTHON:{'libpython3.9.so'},
}

whitelisted_dynamic_libraries = {
    OvmsImageType.UBUNTU20: {"default": dynamic_libraries[OvmsBaseImageType.COMMON] | dynamic_libraries[OvmsBaseImageType.UBUNTU] | dynamic_libraries[OvmsBaseImageType.UBUNTU20],
                             "python": dynamic_libraries[OvmsBaseImageType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseImageType.UBUNTU20_PYTHON]},
    OvmsImageType.UBUNTU22: {"default": dynamic_libraries[OvmsBaseImageType.COMMON] | dynamic_libraries[OvmsBaseImageType.UBUNTU] | dynamic_libraries[OvmsBaseImageType.UBUNTU22],
                             "python": dynamic_libraries[OvmsBaseImageType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseImageType.UBUNTU22_PYTHON]},
    OvmsImageType.UBUNTU22_GPU: {"default": dynamic_libraries[OvmsBaseImageType.COMMON] | dynamic_libraries[OvmsBaseImageType.UBUNTU] | dynamic_libraries[OvmsBaseImageType.UBUNTU22],
                                  "python": dynamic_libraries[OvmsBaseImageType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseImageType.UBUNTU22_PYTHON]},
    OvmsImageType.UBUNTU22_NGINX: {"default": dynamic_libraries[OvmsBaseImageType.COMMON]  | dynamic_libraries[OvmsBaseImageType.UBUNTU] | dynamic_libraries[OvmsBaseImageType.UBUNTU22],
                                   "python": dynamic_libraries[OvmsBaseImageType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseImageType.UBUNTU22_PYTHON]},
    OvmsImageType.UBUNTU24: {"default": dynamic_libraries[OvmsBaseImageType.COMMON] | dynamic_libraries[OvmsBaseImageType.UBUNTU] | dynamic_libraries[OvmsBaseImageType.UBUNTU24],
                             "python": dynamic_libraries[OvmsBaseImageType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseImageType.UBUNTU24_PYTHON]},
    OvmsImageType.UBUNTU24_GPU: {"default": dynamic_libraries[OvmsBaseImageType.COMMON] | dynamic_libraries[OvmsBaseImageType.UBUNTU] | dynamic_libraries[OvmsBaseImageType.UBUNTU24],
                                  "python": dynamic_libraries[OvmsBaseImageType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseImageType.UBUNTU24_PYTHON]},
    OvmsImageType.UBUNTU24_NGINX: {"default": dynamic_libraries[OvmsBaseImageType.COMMON]  | dynamic_libraries[OvmsBaseImageType.UBUNTU] | dynamic_libraries[OvmsBaseImageType.UBUNTU24],
                                   "python": dynamic_libraries[OvmsBaseImageType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseImageType.UBUNTU24_PYTHON]},
    OvmsImageType.REDHAT: {"default": dynamic_libraries[OvmsBaseImageType.COMMON] | dynamic_libraries[OvmsBaseImageType.REDHAT],
                           "python": dynamic_libraries[OvmsBaseImageType.REDHAT_PYTHON]},
    OvmsImageType.REDHAT_GPU: {"default": dynamic_libraries[OvmsBaseImageType.COMMON] | dynamic_libraries[OvmsBaseImageType.REDHAT],
                               "python": dynamic_libraries[OvmsBaseImageType.REDHAT_PYTHON]},
}

# Libraries located in /ovms/lib/
libraries = {
    OvmsBaseImageType.COMMON: {
        'libazurestorage.so',
        'libcpprest.so',
        'libface_detection_cc_proto.so',
        'libface_detection_options_registry.so',
        'libinference_calculator_cc_proto.so',
        'libinference_calculator_options_registry.so',
        'libOpenCL.so',
        'libopencv_calib3d.so',
        'libopencv_core.so',
        'libopencv_features2d.so',
        'libopencv_flann.so',
        'libopencv_highgui.so',
        'libopencv_imgcodecs.so',
        'libopencv_imgproc.so',
        'libopencv_optflow.so',
        'libopencv_video.so',
        'libopencv_videoio.so',
        'libopencv_ximgproc.so',
        'libopenvino.so',
        'libopenvino_auto_batch_plugin.so',
        'libopenvino_auto_plugin.so',
        'libopenvino_c.so',
        'libopenvino_genai.so',
        'libopenvino_hetero_plugin.so',
        'libopenvino_intel_cpu_plugin.so',
        'libopenvino_intel_gpu_plugin.so',
        'libopenvino_ir_frontend.so',
        'libopenvino_onnx_frontend.so',
        'libopenvino_paddle_frontend.so',
        'libopenvino_pytorch_frontend.so',
        'libopenvino_tensorflow_frontend.so',
        'libopenvino_tensorflow_lite_frontend.so',
        'libopenvino_tokenizers.so',
        'libtbb.so',
    },
    OvmsBaseImageType.UBUNTU: set(),
    OvmsBaseImageType.UBUNTU22: {'libopenvino_intel_npu_plugin.so',},
    OvmsBaseImageType.UBUNTU24: {'libopenvino_intel_npu_plugin.so',},
    OvmsBaseImageType.UBUNTU20_PYTHON: set(),
    OvmsBaseImageType.UBUNTU22_PYTHON: set(),
    OvmsBaseImageType.UBUNTU24_PYTHON: set(),
    OvmsBaseImageType.REDHAT: {'libpugixml.so',},
    OvmsBaseImageType.REDHAT_PYTHON: set(),
}

whitelisted_libraries = {
    OvmsImageType.UBUNTU20: {"default": libraries[OvmsBaseImageType.COMMON] | libraries[OvmsBaseImageType.UBUNTU]},
    OvmsImageType.UBUNTU22: {"default": libraries[OvmsBaseImageType.COMMON] | libraries[OvmsBaseImageType.UBUNTU] | libraries[OvmsBaseImageType.UBUNTU22]},
    OvmsImageType.UBUNTU22_GPU: {"default": libraries[OvmsBaseImageType.COMMON] | libraries[OvmsBaseImageType.UBUNTU] | libraries[OvmsBaseImageType.UBUNTU22]},
    OvmsImageType.UBUNTU22_NGINX: {"default": libraries[OvmsBaseImageType.COMMON] | libraries[OvmsBaseImageType.UBUNTU] | libraries[OvmsBaseImageType.UBUNTU22]},
    OvmsImageType.UBUNTU24: {"default": libraries[OvmsBaseImageType.COMMON] | libraries[OvmsBaseImageType.UBUNTU] | libraries[OvmsBaseImageType.UBUNTU24]},
    OvmsImageType.UBUNTU24_GPU: {"default": libraries[OvmsBaseImageType.COMMON] | libraries[OvmsBaseImageType.UBUNTU] | libraries[OvmsBaseImageType.UBUNTU24]},
    OvmsImageType.UBUNTU24_NGINX: {"default": libraries[OvmsBaseImageType.COMMON] | libraries[OvmsBaseImageType.UBUNTU] | libraries[OvmsBaseImageType.UBUNTU24]},
    OvmsImageType.REDHAT: {"default": libraries[OvmsBaseImageType.COMMON] | libraries[OvmsBaseImageType.REDHAT]},
    OvmsImageType.REDHAT_GPU: {"default": libraries[OvmsBaseImageType.COMMON] | libraries[OvmsBaseImageType.REDHAT]},
}

# Apt/yum packages
packages = {
    OvmsBaseImageType.UBUNTU: {
        'ca-certificates',
        'curl',
        'libxml2',
        'openssl',
    },
    OvmsBaseImageType.UBUNTU_PYTHON: {
        'libexpat1',
        'libsqlite3-0',
        'readline-common',
    },
    OvmsBaseImageType.UBUNTU20: {
        'libicu66',
        'libssl1.1',
        'tzdata',
    },
    OvmsBaseImageType.UBUNTU20_PYTHON: {
        'libmpdec2',
        'libpython3.8',
        'libpython3.8-minimal',
        'libpython3.8-stdlib',
        'mime-support',
    },
    OvmsBaseImageType.UBUNTU22: {'libicu70'},
    OvmsBaseImageType.UBUNTU22_PYTHON: {
        'libmpdec3',
        'libreadline8',
        'libpython3.10',
        'libpython3.10-minimal',
        'libpython3.10-stdlib',
        'media-types',
    },
    OvmsBaseImageType.UBUNTU24: {'libicu74',
        'tzdata',
        'netbase',
        'libreadline8t64'},
    OvmsBaseImageType.UBUNTU24_PYTHON: {
        'libpython3.12t64',
        'libpython3.12-minimal',
        'libpython3.12-stdlib',
        'media-types',
    },
    OvmsBaseImageType.UBUNTU_GPU: {
        'intel-driver-compiler-npu',
        'intel-fw-npu',
        'intel-igc-core-2',
        'intel-igc-opencl-2',
        'intel-level-zero-gpu',
        'intel-level-zero-npu',
        'intel-opencl-icd',
        'level-zero',
        'libhwloc15',
        'libigdgmm12',
        'libnuma1',
        'libtbb12',
        'libtbbbind-2-5',
        'libtbbmalloc2',
        'ocl-icd-libopencl1',
    },
    OvmsBaseImageType.UBUNTU_NGINX: {
        'dumb-init',
        'libkrb5support0',
        'libk5crypto3',
        'libkeyutils1',
        'libkrb5-3',
        'libgssapi-krb5-2',
        'nginx',
        'nginx-common',
    },
    OvmsBaseImageType.REDHAT: {
    },
    OvmsBaseImageType.REDHAT_PYTHON: {
        'expat',
        'python3-libs',
        'python3-pip-wheel',
        'python3-setuptools-wheel',
    },
    OvmsBaseImageType.REDHAT_GPU: {
        'intel-gmmlib',
        'intel-igc-core',
        'intel-igc-opencl',
        'intel-level-zero-gpu',
        'intel-opencl',
        'level-zero',
        'libedit',
        'libnl3',
        'ocl-icd',
    },
}

whitelisted_packages = {
    OvmsImageType.UBUNTU20: {"default": packages[OvmsBaseImageType.UBUNTU] | packages[OvmsBaseImageType.UBUNTU20],
                             "python": packages[OvmsBaseImageType.UBUNTU_PYTHON] | packages[OvmsBaseImageType.UBUNTU20_PYTHON]},
    OvmsImageType.UBUNTU22: {"default": packages[OvmsBaseImageType.UBUNTU] | packages[OvmsBaseImageType.UBUNTU22],
                             "python": packages[OvmsBaseImageType.UBUNTU_PYTHON] | packages[OvmsBaseImageType.UBUNTU22_PYTHON]},
    OvmsImageType.UBUNTU22_GPU: {"default": packages[OvmsBaseImageType.UBUNTU] | packages[OvmsBaseImageType.UBUNTU22] | packages[OvmsBaseImageType.UBUNTU_GPU],
                                 "python": packages[OvmsBaseImageType.UBUNTU_PYTHON] | packages[OvmsBaseImageType.UBUNTU22_PYTHON]},
    OvmsImageType.UBUNTU22_NGINX: {"default": packages[OvmsBaseImageType.UBUNTU] | packages[OvmsBaseImageType.UBUNTU22] | packages[OvmsBaseImageType.UBUNTU_NGINX],
                                   "python": packages[OvmsBaseImageType.UBUNTU_PYTHON] | packages[OvmsBaseImageType.UBUNTU22_PYTHON]},
    OvmsImageType.UBUNTU24: {"default": packages[OvmsBaseImageType.UBUNTU] | packages[OvmsBaseImageType.UBUNTU24],
                             "python": packages[OvmsBaseImageType.UBUNTU_PYTHON] | packages[OvmsBaseImageType.UBUNTU24_PYTHON]},
    OvmsImageType.UBUNTU24_GPU: {"default": packages[OvmsBaseImageType.UBUNTU] | packages[OvmsBaseImageType.UBUNTU24] | packages[OvmsBaseImageType.UBUNTU_GPU],
                                 "python": packages[OvmsBaseImageType.UBUNTU_PYTHON] | packages[OvmsBaseImageType.UBUNTU24_PYTHON]},
    OvmsImageType.UBUNTU24_NGINX: {"default": packages[OvmsBaseImageType.UBUNTU] | packages[OvmsBaseImageType.UBUNTU24] | packages[OvmsBaseImageType.UBUNTU_NGINX],
                                   "python": packages[OvmsBaseImageType.UBUNTU_PYTHON] | packages[OvmsBaseImageType.UBUNTU24_PYTHON]},
    OvmsImageType.REDHAT: {"default": packages[OvmsBaseImageType.REDHAT],
                           "python": packages[OvmsBaseImageType.REDHAT_PYTHON]},
    OvmsImageType.REDHAT_GPU: {"default":  packages[OvmsBaseImageType.REDHAT] | packages[OvmsBaseImageType.REDHAT_GPU],
                               "python": packages[OvmsBaseImageType.REDHAT_PYTHON]},
}
