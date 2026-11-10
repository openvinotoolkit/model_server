#
# Copyright (c) 2023-2026 Intel Corporation
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
    UBUNTU22 = auto()
    UBUNTU22_GPU = auto()
    UBUNTU22_NGINX = auto()
    UBUNTU24 = auto()
    UBUNTU24_GPU = auto()
    UBUNTU24_NGINX = auto()
    UBUNTU26 = auto()
    UBUNTU26_GPU = auto()
    UBUNTU26_NGINX = auto()
    REDHAT = auto()
    REDHAT_GPU = auto()
    REDHAT_CUDA = auto()
    WINDOWS = auto()


class OvmsBaseType(Enum):
    COMMON = "common"
    UBUNTU = "ubuntu"
    UBUNTU22 = "ubuntu22"
    UBUNTU24 = "ubuntu24"
    UBUNTU26 = "ubuntu26"
    UBUNTU_GPU = "ubuntu_gpu"
    UBUNTU_NGINX = "ubuntu_nginx"
    REDHAT = "redhat"
    REDHAT_GPU = "redhat_gpu"
    WINDOWS = "windows"


# Libraries listed by ldd /ovms/bin/ovms
dynamic_libraries = {
    OvmsBaseType.COMMON: {
        'libgcc_s.so',
        'libstdc++.so',
        'libuuid.so',
        'libxml2.so',
    },
    OvmsBaseType.UBUNTU: set(),
    OvmsBaseType.UBUNTU24: {'libdl.so', 'libm.so', 'libpthread.so', 'liblzma.so', 'libicuuc.so', 'libicudata.so',},
    OvmsBaseType.UBUNTU26: {'libdl.so', 'libm.so', 'libpthread.so'},
    OvmsBaseType.REDHAT: {'libdl.so', 'libm.so', 'libpthread.so', 'liblzma.so'},
}

whitelisted_dynamic_libraries = {
    OvmsType.UBUNTU24: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.UBUNTU] | dynamic_libraries[OvmsBaseType.UBUNTU24]},
    OvmsType.UBUNTU24_GPU: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.UBUNTU] | dynamic_libraries[OvmsBaseType.UBUNTU24]},
    OvmsType.UBUNTU24_NGINX: {"default": dynamic_libraries[OvmsBaseType.COMMON]  | dynamic_libraries[OvmsBaseType.UBUNTU] | dynamic_libraries[OvmsBaseType.UBUNTU24]},
    OvmsType.UBUNTU26: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.UBUNTU] | dynamic_libraries[OvmsBaseType.UBUNTU26]},
    OvmsType.UBUNTU26_GPU: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.UBUNTU] | dynamic_libraries[OvmsBaseType.UBUNTU26]},
    OvmsType.UBUNTU26_NGINX: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.UBUNTU] | dynamic_libraries[OvmsBaseType.UBUNTU26]},
    OvmsType.REDHAT: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.REDHAT]},
    OvmsType.REDHAT_GPU: {"default": dynamic_libraries[OvmsBaseType.COMMON] | dynamic_libraries[OvmsBaseType.REDHAT]},
}

# Libraries located in /ovms/lib/
libraries = {
    OvmsBaseType.COMMON: {
        'libazure-core.so',
        'libazure-storage-blobs.so',
        'libazure-storage-common.so',
        'libazure-storage-files-shares.so',
        'libespeak-ng.so',
        'libgit2.so',
        'libjsoncpp.so',
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
        'libopenvino_gguf_frontend.so',
        'libopenvino_hetero_plugin.so',
        'libopenvino_intel_cpu_plugin.so',
        'libopenvino_intel_gpu_plugin.so',
        'libopenvino_intel_npu_plugin.so',
        'libopenvino_ir_frontend.so',
        'libopenvino_onnx_frontend.so',
        'libopenvino_paddle_frontend.so',
        'libopenvino_pytorch_frontend.so',
        'libopenvino_tensorflow_frontend.so',
        'libopenvino_tensorflow_lite_frontend.so',
        'libopenvino_tokenizers.so',
        'libovms_llm_calculators.so',
        'libovmspython.so',
        'libpython_calculators.so',
        'libtbb.so',
    },
    OvmsBaseType.UBUNTU: {
        'libopenvino_intel_npu_compiler.so',
        'libopenvino_intel_npu_compiler_loader.so',
        'libopenvino_intel_npu_vm_runtime.so',
    },
    OvmsBaseType.UBUNTU24: set(),
    OvmsBaseType.UBUNTU26: set(),
    OvmsBaseType.REDHAT: set(),
    OvmsBaseType.WINDOWS: {
        'espeak-ng.dll',
        'git2.dll',
        'libcurl-x64.dll',
        'libovmspython.dll',
        'libpython_calculators.dll',
        'opencv_world4140.dll',
        'openvino.dll',
        'openvino_auto_batch_plugin.dll',
        'openvino_auto_plugin.dll',
        'openvino_c.dll',
        'openvino_genai.dll',
        'openvino_genai_c.dll',
        'openvino_gguf_frontend.dll',
        'openvino_hetero_plugin.dll',
        'openvino_intel_cpu_plugin.dll',
        'openvino_intel_gpu_plugin.dll',
        'openvino_intel_npu_compiler.dll',
        'openvino_intel_npu_compiler_loader.dll',
        'openvino_intel_npu_plugin.dll',
        'openvino_intel_npu_vm_runtime.dll',
        'openvino_ir_frontend.dll',
        'openvino_onnx_frontend.dll',
        'openvino_paddle_frontend.dll',
        'openvino_pytorch_frontend.dll',
        'openvino_tensorflow_frontend.dll',
        'openvino_tensorflow_lite_frontend.dll',
        'openvino_tokenizers.dll',
        'tbb12.dll',
        'cli.exe',
        'cli-32.exe',
        'cli-64.exe',
        'cli-arm64.exe',
        'gui.exe',
        'gui-32.exe',
        'gui-64.exe',
        'gui-arm64.exe',
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
    OvmsType.UBUNTU24: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.UBUNTU] | libraries[OvmsBaseType.UBUNTU24]},
    OvmsType.UBUNTU24_GPU: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.UBUNTU] | libraries[OvmsBaseType.UBUNTU24]},
    OvmsType.UBUNTU24_NGINX: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.UBUNTU] | libraries[OvmsBaseType.UBUNTU24]},
    OvmsType.UBUNTU26: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.UBUNTU] | libraries[OvmsBaseType.UBUNTU26]},
    OvmsType.UBUNTU26_GPU: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.UBUNTU] | libraries[OvmsBaseType.UBUNTU26]},
    OvmsType.UBUNTU26_NGINX: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.UBUNTU] | libraries[OvmsBaseType.UBUNTU26]},
    OvmsType.REDHAT: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.REDHAT]},
    OvmsType.REDHAT_GPU: {"default": libraries[OvmsBaseType.COMMON] | libraries[OvmsBaseType.REDHAT]},
    OvmsType.WINDOWS: {"default": libraries[OvmsBaseType.WINDOWS]},
}

# Apt/yum packages
packages = {
    OvmsBaseType.UBUNTU: {
        'ca-certificates',
        'curl',
        'openssl',
        'libexpat1',
        'libsqlite3-0',
        'readline-common',
        'tzdata',
        'netbase',
        'libreadline8t64',
        'media-types',
    },
    OvmsBaseType.UBUNTU24: {
        'libxml2',
        'libicu74',
        'libpython3.12t64',
        'libpython3.12-minimal',
        'libpython3.12-stdlib',
    },
    OvmsBaseType.UBUNTU26: {
        'libpython3.14',
        'libpython3.14-minimal',
        'libpython3.14-stdlib',
        'libnettle8t64',
        'libsasl2-2',
        'libgnutls30t64',
        'libkeyutils1',
        'libkrb5-3',
        'libk5crypto3',
        'libxml2-16',
        'libldap2',
        'libidn2-0',
        'libsasl2-modules-db',
        'libp11-kit0',
        'libnghttp2-14',
        'libunistring5',
        'libhogweed6t64',
        'libssh2-1t64',
        'libcurl4t64',
        'libldap-common',
        'libkrb5support0',
        'librtmp1',
        'libtasn1-6',
        'libffi8',
        'libpsl5t64',
        'libgssapi-krb5-2',
        'libbrotli1',
    },
    OvmsBaseType.UBUNTU_GPU: {
        'intel-driver-compiler-npu',
        'intel-fw-npu',
        'intel-igc-core-2',
        'intel-igc-opencl-2',
        'intel-level-zero-npu',
        'intel-ocloc',
        'intel-opencl-icd',
        'libze1',
        'libhwloc15',
        'libigdgmm12',
        'libnuma1',
        'libtbb12',
        'libtbbbind-2-5',
        'libtbbmalloc2',
        'libze-intel-gpu1',
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
    OvmsBaseType.REDHAT: {
        "findutils",
        'libtirpc',
        'mpdecimal',
        'libnsl2',
        'expat',
        'python3.12-libs',
        'python3.12-pip-wheel',
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
        'libpsl',
        'publicsuffix-list-dafsa',
        'wget',
    },
}

whitelisted_packages = {
    OvmsType.UBUNTU24: {"default": packages[OvmsBaseType.UBUNTU] | packages[OvmsBaseType.UBUNTU24]},
    OvmsType.UBUNTU24_GPU: {"default": packages[OvmsBaseType.UBUNTU] | packages[OvmsBaseType.UBUNTU24] | packages[OvmsBaseType.UBUNTU_GPU]},
    OvmsType.UBUNTU24_NGINX: {"default": packages[OvmsBaseType.UBUNTU] | packages[OvmsBaseType.UBUNTU24] | packages[OvmsBaseType.UBUNTU_NGINX]},
    OvmsType.UBUNTU26: {"default": packages[OvmsBaseType.UBUNTU] | packages[OvmsBaseType.UBUNTU26]},
    OvmsType.UBUNTU26_GPU: {"default": packages[OvmsBaseType.UBUNTU] | packages[OvmsBaseType.UBUNTU26] | packages[OvmsBaseType.UBUNTU_GPU]},
    OvmsType.UBUNTU26_NGINX: {"default": packages[OvmsBaseType.UBUNTU] | packages[OvmsBaseType.UBUNTU26] | packages[OvmsBaseType.UBUNTU_NGINX]},
    OvmsType.REDHAT: {"default": packages[OvmsBaseType.REDHAT]},
    OvmsType.REDHAT_GPU: {"default":  packages[OvmsBaseType.REDHAT] | packages[OvmsBaseType.REDHAT_GPU]},
}
