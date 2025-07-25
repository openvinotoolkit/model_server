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


class OvmsType(Enum):
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
    WINDOWS = auto()


class OvmsBaseType(Enum):
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
    WINDOWS = "windows"
    WINDOWS_PYTHON = "windows_python"


# Libraries listed by ldd /ovms/bin/ovms
dynamic_libraries = {
    OvmsBaseType.COMMON: {
        'libgcc_s.so',
        'liblzma.so',
        'libstdc++.so',
        'libuuid.so',
        'libxml2.so',
    },
    OvmsBaseType.UBUNTU: {'libicuuc.so', 'libicudata.so',},
    OvmsBaseType.UBUNTU_PYTHON: {'libexpat.so',},
    OvmsBaseType.UBUNTU20: {'librt.so',},
    OvmsBaseType.UBUNTU20_PYTHON: {'libpython3.8.so', 'libutil.so',},
    OvmsBaseType.UBUNTU22: {'libdl.so', 'libm.so', 'libpthread.so',},
    OvmsBaseType.UBUNTU22_PYTHON: {'libpython3.10.so',},
    OvmsBaseType.UBUNTU24: {'libdl.so', 'libm.so', 'libpthread.so',},
    OvmsBaseType.UBUNTU24_PYTHON: {'libpython3.12.so',},
    OvmsBaseType.REDHAT: {'libdl.so', 'libm.so', 'libpthread.so', 'libcrypt.so',},
    OvmsBaseType.REDHAT_PYTHON:{'libpython3.9.so'},
}

whitelisted_dynamic_libraries = {
    OvmsType.UBUNTU20: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.UBUNTU] | dynamic_libraries[OvmsBaseType.UBUNTU20],
                             "python": dynamic_libraries[OvmsBaseType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseType.UBUNTU20_PYTHON]},
    OvmsType.UBUNTU22: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.UBUNTU] | dynamic_libraries[OvmsBaseType.UBUNTU22],
                             "python": dynamic_libraries[OvmsBaseType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseType.UBUNTU22_PYTHON]},
    OvmsType.UBUNTU22_GPU: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.UBUNTU] | dynamic_libraries[OvmsBaseType.UBUNTU22],
                                  "python": dynamic_libraries[OvmsBaseType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseType.UBUNTU22_PYTHON]},
    OvmsType.UBUNTU22_NGINX: {"default": dynamic_libraries[OvmsBaseType.COMMON]  | dynamic_libraries[OvmsBaseType.UBUNTU] | dynamic_libraries[OvmsBaseType.UBUNTU22],
                                   "python": dynamic_libraries[OvmsBaseType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseType.UBUNTU22_PYTHON]},
    OvmsType.UBUNTU24: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.UBUNTU] | dynamic_libraries[OvmsBaseType.UBUNTU24],
                             "python": dynamic_libraries[OvmsBaseType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseType.UBUNTU24_PYTHON]},
    OvmsType.UBUNTU24_GPU: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.UBUNTU] | dynamic_libraries[OvmsBaseType.UBUNTU24],
                                  "python": dynamic_libraries[OvmsBaseType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseType.UBUNTU24_PYTHON]},
    OvmsType.UBUNTU24_NGINX: {"default": dynamic_libraries[OvmsBaseType.COMMON]  | dynamic_libraries[OvmsBaseType.UBUNTU] | dynamic_libraries[OvmsBaseType.UBUNTU24],
                                   "python": dynamic_libraries[OvmsBaseType.UBUNTU_PYTHON] | dynamic_libraries[OvmsBaseType.UBUNTU24_PYTHON]},
    OvmsType.REDHAT: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.REDHAT],
                           "python": dynamic_libraries[OvmsBaseType.REDHAT_PYTHON]},
    OvmsType.REDHAT_GPU: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.REDHAT],
                               "python": dynamic_libraries[OvmsBaseType.REDHAT_PYTHON]},
}

# Libraries located in /ovms/lib/
libraries = {
    OvmsBaseType.COMMON: {
        'libazurestorage.so',
        'libcpprest.so',
        'libgit2.so',
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
    OvmsBaseType.UBUNTU: set(),
    OvmsBaseType.UBUNTU22: {'libopenvino_intel_npu_plugin.so',},
    OvmsBaseType.UBUNTU24: {'libopenvino_intel_npu_plugin.so',},
    OvmsBaseType.UBUNTU20_PYTHON: set(),
    OvmsBaseType.UBUNTU22_PYTHON: set(),
    OvmsBaseType.UBUNTU24_PYTHON: set(),
    OvmsBaseType.REDHAT: set(),
    OvmsBaseType.REDHAT_PYTHON: set(),
    OvmsBaseType.WINDOWS: {
        'git.exe',
        'git2.dll',
        'git-lfs.exe',
        'icudt70.dll',
        'icuuc70.dll',
        'libcurl-x64.dll',
        'libiconv-2.dll',
        'libintl-8.dll',
        'libpcre2-8-0.dll',
        'libwinpthread-1.dll',
        'opencv_world4100.dll',
        'openvino.dll',
        'openvino_auto_batch_plugin.dll',
        'openvino_auto_plugin.dll',
        'openvino_c.dll',
        'openvino_genai.dll',
        'openvino_genai_c.dll',
        'openvino_hetero_plugin.dll',
        'openvino_intel_cpu_plugin.dll',
        'openvino_intel_gpu_plugin.dll',
        'openvino_intel_npu_plugin.dll',
        'openvino_ir_frontend.dll',
        'openvino_onnx_frontend.dll',
        'openvino_paddle_frontend.dll',
        'openvino_pytorch_frontend.dll',
        'openvino_tensorflow_frontend.dll',
        'openvino_tensorflow_lite_frontend.dll',
        'openvino_tokenizers.dll',
        'tbb12.dll',
        'zlib1.dll',
    },
    OvmsBaseType.WINDOWS_PYTHON: {
        'libcrypto-3.dll',
        'libffi-8.dll',
        'libssl-3.dll',
        'pip.exe',
        'pip3.exe',
        'pip3.12.exe',
        'python.exe',
        'python3.dll',
        'python312.dll',
        'pythonw.exe',
        'sqlite3.dll',
        't32.exe',
        't64.exe',
        't64-arm.exe',
        'w32.exe',
        'w64.exe',
        'w64-arm.exe',
        'vcruntime140.dll',
        'vcruntime140_1.dll',
    },
}

whitelisted_libraries = {
    OvmsType.UBUNTU20: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.UBUNTU]},
    OvmsType.UBUNTU22: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.UBUNTU] | libraries[OvmsBaseType.UBUNTU22]},
    OvmsType.UBUNTU22_GPU: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.UBUNTU] | libraries[OvmsBaseType.UBUNTU22]},
    OvmsType.UBUNTU22_NGINX: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.UBUNTU] | libraries[OvmsBaseType.UBUNTU22]},
    OvmsType.UBUNTU24: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.UBUNTU] | libraries[OvmsBaseType.UBUNTU24]},
    OvmsType.UBUNTU24_GPU: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.UBUNTU] | libraries[OvmsBaseType.UBUNTU24]},
    OvmsType.UBUNTU24_NGINX: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.UBUNTU] | libraries[OvmsBaseType.UBUNTU24]},
    OvmsType.REDHAT: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.REDHAT]},
    OvmsType.REDHAT_GPU: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.REDHAT]},
    OvmsType.WINDOWS: {"default": libraries[OvmsBaseType.WINDOWS], "python": libraries[OvmsBaseType.WINDOWS_PYTHON]},
}

# Apt/yum packages
packages = {
    OvmsBaseType.UBUNTU: {
        'ca-certificates',
        'curl',
        'libxml2',
        'openssl',
    },
    OvmsBaseType.UBUNTU_PYTHON: {
        'libexpat1',
        'libsqlite3-0',
        'readline-common',
    },
    OvmsBaseType.UBUNTU20: {
        'libicu66',
        'libssl1.1',
        'tzdata',
    },
    OvmsBaseType.UBUNTU20_PYTHON: {
        'libmpdec2',
        'libpython3.8',
        'libpython3.8-minimal',
        'libpython3.8-stdlib',
        'mime-support',
    },
    OvmsBaseType.UBUNTU22: {'libicu70'},
    OvmsBaseType.UBUNTU22_PYTHON: {
        'libmpdec3',
        'libreadline8',
        'libpython3.10',
        'libpython3.10-minimal',
        'libpython3.10-stdlib',
        'media-types',
    },
    OvmsBaseType.UBUNTU24: {'libicu74',
        'tzdata',
        'netbase',
        'libreadline8t64'},
    OvmsBaseType.UBUNTU24_PYTHON: {
        'libpython3.12t64',
        'libpython3.12-minimal',
        'libpython3.12-stdlib',
        'media-types',
    },
    OvmsBaseType.UBUNTU_GPU: {
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
    OvmsBaseType.UBUNTU_NGINX: {
        'dumb-init',
        'libkrb5support0',
        'libk5crypto3',
        'libkeyutils1',
        'libkrb5-3',
        'libgssapi-krb5-2',
        'nginx',
    },
    OvmsBaseType.REDHAT: set(),
    OvmsBaseType.REDHAT_PYTHON: {
        'expat',
        'python3-libs',
        'python3-pip-wheel',
        'python3-setuptools-wheel',
    },
    OvmsBaseType.REDHAT_GPU: {
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
    OvmsType.UBUNTU20: {"default": packages[OvmsBaseType.UBUNTU] | packages[OvmsBaseType.UBUNTU20],
                             "python": packages[OvmsBaseType.UBUNTU_PYTHON] | packages[OvmsBaseType.UBUNTU20_PYTHON]},
    OvmsType.UBUNTU22: {"default": packages[OvmsBaseType.UBUNTU] | packages[OvmsBaseType.UBUNTU22],
                             "python": packages[OvmsBaseType.UBUNTU_PYTHON] | packages[OvmsBaseType.UBUNTU22_PYTHON]},
    OvmsType.UBUNTU22_GPU: {"default": packages[OvmsBaseType.UBUNTU] | packages[OvmsBaseType.UBUNTU22] | packages[OvmsBaseType.UBUNTU_GPU],
                                 "python": packages[OvmsBaseType.UBUNTU_PYTHON] | packages[OvmsBaseType.UBUNTU22_PYTHON]},
    OvmsType.UBUNTU22_NGINX: {"default": packages[OvmsBaseType.UBUNTU] | packages[OvmsBaseType.UBUNTU22] | packages[OvmsBaseType.UBUNTU_NGINX],
                                   "python": packages[OvmsBaseType.UBUNTU_PYTHON] | packages[OvmsBaseType.UBUNTU22_PYTHON]},
    OvmsType.UBUNTU24: {"default": packages[OvmsBaseType.UBUNTU] | packages[OvmsBaseType.UBUNTU24],
                             "python": packages[OvmsBaseType.UBUNTU_PYTHON] | packages[OvmsBaseType.UBUNTU24_PYTHON]},
    OvmsType.UBUNTU24_GPU: {"default": packages[OvmsBaseType.UBUNTU] | packages[OvmsBaseType.UBUNTU24] | packages[OvmsBaseType.UBUNTU_GPU],
                                 "python": packages[OvmsBaseType.UBUNTU_PYTHON] | packages[OvmsBaseType.UBUNTU24_PYTHON]},
    OvmsType.UBUNTU24_NGINX: {"default": packages[OvmsBaseType.UBUNTU] | packages[OvmsBaseType.UBUNTU24] | packages[OvmsBaseType.UBUNTU_NGINX],
                                   "python": packages[OvmsBaseType.UBUNTU_PYTHON] | packages[OvmsBaseType.UBUNTU24_PYTHON]},
    OvmsType.REDHAT: {"default": packages[OvmsBaseType.REDHAT],
                           "python": packages[OvmsBaseType.REDHAT_PYTHON]},
    OvmsType.REDHAT_GPU: {"default":  packages[OvmsBaseType.REDHAT] | packages[OvmsBaseType.REDHAT_GPU],
                               "python": packages[OvmsBaseType.REDHAT_PYTHON]},
}
