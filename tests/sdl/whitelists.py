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
    UBUNTU_GENERIC = auto()
    UBUNTU_22_GENERIC = auto()
    UBUNTU_22_GPU = auto()
    UBUNTU_CUDA = auto()
    UBUNTU_NGINX = auto()
    REDHAT_GENERIC = auto()
    REDHAT_GPU = auto()
    REDHAT_CUDA     = auto()


# DYNAMIC LIBRARIES
dynamic_libraries_common = {
    'libgcc_s.so', 'liblzma.so', 'libstdc++.so', 'libuuid.so', 'libxml2.so'
}
 
dynamic_libraries_ubuntu = {'libicuuc.so', 'libicudata.so',}

dynamic_libraries_ubuntu_python = {'libexpat.so',}

dynamic_libraries_ubuntu_20_04 = dynamic_libraries_ubuntu | {'librt.so', 'libtbb.so',}

dynamic_libraries_ubuntu_20_04_python = dynamic_libraries_ubuntu_python | {'libpython3.8.so', 'libutil.so',}

dynamic_libraries_ubuntu_22_04 = dynamic_libraries_ubuntu | {'libm.so', 'libdl.so', 'libpthread.so',}

dynamic_libraries_ubuntu_22_04_pythnon = dynamic_libraries_ubuntu_python | {'libpython3.10.so',}

dynamic_libraries_ubuntu_22_04_no_python = {'libpugixml.so',}

dynamic_libraries_redhat = {'libtbb.so',}

dynamic_libraries_redhat_python = {'libpython3.9.so', 'libutil.so',}

whitelisted_dynamic_libraries = {
    OvmsImageType.UBUNTU_GENERIC: {"default": dynamic_libraries_common | dynamic_libraries_ubuntu_20_04,
                                   "python": dynamic_libraries_ubuntu_20_04_python},
    OvmsImageType.UBUNTU_NGINX: {"default": dynamic_libraries_common  | dynamic_libraries_ubuntu_20_04,
                                 "python": dynamic_libraries_ubuntu_20_04_python},
    OvmsImageType.UBUNTU_22_GENERIC: {"default": dynamic_libraries_common | dynamic_libraries_ubuntu_22_04,
                                      "python": dynamic_libraries_ubuntu_22_04_pythnon,
                                      "no_python": dynamic_libraries_ubuntu_22_04_no_python},
    OvmsImageType.UBUNTU_22_GPU: {"default": dynamic_libraries_common | dynamic_libraries_ubuntu_22_04,
                                  "python": dynamic_libraries_ubuntu_22_04_pythnon,
                                  "no_python": dynamic_libraries_ubuntu_22_04_no_python},
    OvmsImageType.REDHAT_GENERIC: {"default": dynamic_libraries_common | dynamic_libraries_redhat,
                                   "python": dynamic_libraries_redhat_python},
    OvmsImageType.REDHAT_GPU: {"default": dynamic_libraries_common | dynamic_libraries_redhat,
                               "python": dynamic_libraries_redhat_python},
}

# LIBRARIES
libraries_common = {
    'libazurestorage.so', 'libcpprest.so', 'libface_detection_cc_proto.so', 'libface_detection_options_registry.so',
    'libgna.so', 'libinference_calculator_cc_proto.so', 'libinference_calculator_options_registry.so',
    'libopencv_calib3d.so', 'libopencv_core.so', 'libopencv_features2d.so', 'libopencv_flann.so',
    'libopencv_highgui.so', 'libopencv_imgcodecs.so', 'libopencv_imgproc.so', 'libopencv_optflow.so',
    'libopencv_video.so', 'libopencv_videoio.so', 'libopencv_ximgproc.so', 'libopenvino.so',
    'libopenvino_auto_batch_plugin.so', 'libopenvino_auto_plugin.so', 'libopenvino_c.so', 'libopenvino_gapi_preproc.so',
    'libopenvino_hetero_plugin.so', 'libopenvino_intel_cpu_plugin.so', 'libopenvino_intel_gna_plugin.so',
    'libopenvino_intel_gpu_plugin.so', 'libopenvino_ir_frontend.so', 'libopenvino_onnx_frontend.so',
    'libopenvino_paddle_frontend.so', 'libopenvino_pytorch_frontend.so', 'libopenvino_tensorflow_frontend.so',
    'libopenvino_tensorflow_lite_frontend.so', 'libuser_ov_extensions.so'
}

libraries_ubuntu = {'libtbb.so',}

libraries_ubuntu_20_04_python = {'libpython3.8.so',}

libraries_ubuntu_22_04_python = {'libpython3.10.so',}

libraries_redhat = {'libpugixml.so',}

libraries_redhat_python = {'libpython3.9.so',}

whitelisted_libraries = {
    OvmsImageType.UBUNTU_GENERIC: {"default": libraries_common | libraries_ubuntu,
                                   "python": libraries_ubuntu_20_04_python},
    OvmsImageType.UBUNTU_NGINX: {"default": libraries_common | libraries_ubuntu,
                                 "python": libraries_ubuntu_20_04_python},
    OvmsImageType.UBUNTU_22_GENERIC: {"default": libraries_common | libraries_ubuntu,
                                      "python": libraries_ubuntu_22_04_python},
    OvmsImageType.UBUNTU_22_GPU: {"default": libraries_common | libraries_ubuntu,
                                  "python": libraries_ubuntu_22_04_python},
    OvmsImageType.REDHAT_GENERIC: {"default": libraries_common | libraries_redhat,
                                   "python": libraries_redhat_python},
    OvmsImageType.REDHAT_GPU: {"default": libraries_common | libraries_redhat,
                               "python": libraries_redhat_python},
}

# PACKAGES
apt_packages_common = {
    'ca-certificates',
    'curl',
    'libpugixml1v5',
    'libtbb2',
    'libxml2',
    'openssl',
}

apt_packages_common_python = {
    'libexpat1',
    'libreadline8', 
    'libsqlite3-0',
    'readline-common',
}

apt_packages_ubuntu_20_04 = {
    'libicu66',
    'libssl1.1',
    'tzdata',
}

apt_packages_ubuntu_20_04_python = {
    'libmpdec2',
    'libpython3.8',
    'libpython3.8-minimal',
    'libpython3.8-stdlib',
    'mime-support',
}

apt_packages_ubuntu_22_04 = {
    'libicu70',
    'libtbbmalloc2',
}

apt_packages_ubuntu_22_04_python = {
    'libmpdec3',
    'libpython3.10',
    'libpython3.10-minimal',
    'libpython3.10-stdlib',
    'media-types',
}

apt_packages_gpu = {
    'intel-igc-core',
    'intel-igc-opencl',
    'intel-level-zero-gpu',
    'intel-opencl-icd',
    'libigdgmm12',
    'libnuma1',
    'ocl-icd-libopencl1',
}

apt_packages_cuda = {
    'libcudnn8',
    'libcutensor1',
    'libpng16-16',
}

apt_packages_nginx = {'dumb-init', 'nginx',}

yum_packages_common = {
    'libpkgconf',
    'libsemanage',
    'numactl',
    'numactl-libs',
    'ocl-icd',
    'pkgconf',
    'pkgconf-m4',
    'pkgconf-pkg-config',
    'shadow-utils',
    'tbb',  
}

yum_packages_common_python = {
    'expat',
    'gdbm-libs',
    'libnsl2',
    'libtirpc',
    'numactl-debuginfo',
    'numactl-debugsource',
    'numactl-devel',
    'numactl-libs-debuginfo',
    'opencl-headers',
    'python39-libs',
    'python39-pip-wheel',
    'python39-setuptools-wheel',
}

yum_packages_gpu = {
    'intel-gmmlib',
    'intel-igc-core',
    'intel-igc-opencl',
    'intel-opencl',
    'level-zero',
    'libedit',
}

whitelisted_packages = {
    OvmsImageType.UBUNTU_GENERIC: {"default": apt_packages_common | apt_packages_ubuntu_20_04,
                                   "python": apt_packages_common_python | apt_packages_ubuntu_20_04_python},
    OvmsImageType.UBUNTU_NGINX: {"default": apt_packages_common | apt_packages_ubuntu_20_04 | apt_packages_nginx,
                                 "python": apt_packages_common_python | apt_packages_ubuntu_20_04_python},
    OvmsImageType.UBUNTU_22_GENERIC: {"default": apt_packages_common | apt_packages_ubuntu_22_04,
                                      "python": apt_packages_common_python | apt_packages_ubuntu_22_04_python},
    OvmsImageType.UBUNTU_22_GPU: {"default": apt_packages_common | apt_packages_ubuntu_22_04 | apt_packages_gpu, 
                                  "python": apt_packages_common_python | apt_packages_ubuntu_22_04_python},
    OvmsImageType.REDHAT_GENERIC: {"default":  yum_packages_common,
                                   "python": yum_packages_common_python},
    OvmsImageType.REDHAT_GPU: {"default":  yum_packages_common | yum_packages_gpu,
                               "python": yum_packages_common_python},
}

