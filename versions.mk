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

# Shared dependency versions for OpenVINO, GenAI, and Tokenizers.
# Consumed by Makefile (Linux/macOS) via include and by
# windows_install_build_dependencies.bat via a for /f parser.
# Any variable can be overridden by the environment or command-line.

# Source repository git commits / branches (used for source builds)
OV_BRANCH ?= b96fa5829c9a345da4b809a91a3428a4a9f69bb5
OV_TOKENIZERS_BRANCH ?= f7999c4078db739a920eba72c6049892e7aabfbd
OV_GENAI_BRANCH ?= 3b877d75fd1c6e3436b547b8c5e4a0b1696cc0ab

# Source repositories (GitHub owner/repository)
OV_REPO ?= openvinotoolkit/openvino
OV_GENAI_REPO ?= openvinotoolkit/openvino.genai
OV_TOKENIZERS_REPO ?= openvinotoolkit/openvino_tokenizers

# Binary package URLs for each supported platform.
DLDT_PACKAGE_URL_UBUNTU26 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2027.0.0.0.dev20261127/openvino_genai_ubuntu26_2027.0.0.0.dev20261127_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU26 ?= 8b9c49133386be3f76a1f5e2a8e9761afc3634b022f14946fca221819ba040b0
DLDT_PACKAGE_URL_UBUNTU24 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2027.0.0.0.dev20261127/openvino_genai_ubuntu24_2027.0.0.0.dev20261127_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU24 ?= ec157cd09a0b22603ff0f1f903adcd457749027ccfd194584b76956138dd681a
DLDT_PACKAGE_URL_UBUNTU22 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2027.0.0.0.dev20261127/openvino_genai_ubuntu22_2027.0.0.0.dev20261127_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU22 ?= a41b4192e49cf1c331a5db5484ef57f6ac50a81f6ee293d2eb07620afe470930
DLDT_PACKAGE_URL_RHEL ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2027.0.0.0.dev20261127/openvino_genai_rhel9_2027.0.0.0.dev20261127_x86_64.tar.gz
DLDT_PACKAGE_SHA256_RHEL ?= 205601a5fdf4cc960954c24658fed6ebcf7dc4b32d30ade8228b9c122205d4cc
GENAI_PACKAGE_URL_WINDOWS ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2027.0.0.0.dev20261127/openvino_genai_windows_2027.0.0.0.dev20261127_x86_64.zip
GENAI_PACKAGE_SHA256_WINDOWS ?= 3259008f8f66af5ac6db5c9e6b834e91c47f02c61ff642ca132438ed08ab09a7
# Third-party dependency versions (used for binary builds)
OPENCV_VERSION ?= 4.14.0
CURL_VERSION ?= 8.22.0_2

# Ubuntu accelerator defaults and verified runtime packages.
GPU_DRIVER_VERSION_UBUNTU22 ?= 24.52.32224
GPU_DRIVER_VERSION_UBUNTU24 ?= 26.31.39395
GPU_DRIVER_VERSION_UBUNTU26 ?= 26.31.39395
NPU_DRIVER_URL_UBUNTU22 ?= https://github.com/intel/linux-npu-driver/releases/download/v1.26.0/linux-npu-driver-v1.26.0.20251125-19665715237-ubuntu2204.tar.gz
NPU_DRIVER_SHA256_UBUNTU22 ?= cfdbcc9adc1ea20d498ebd9cbdb5c212f6fc940e1034ef7a72e239a8636f653a
NPU_DRIVER_URL_UBUNTU24 ?= https://github.com/intel/linux-npu-driver/releases/download/v1.39.1/linux-npu-driver-v1.39.1.20261119-44517102376-ubuntu2404.tar.gz
NPU_DRIVER_SHA256_UBUNTU24 ?= 884a461b152281f259cd6759a8ae17a77968712b939543607360ec562d4741fc
NPU_DRIVER_URL_UBUNTU26 ?= https://github.com/intel/linux-npu-driver/releases/download/v1.39.1/linux-npu-driver-v1.39.1.20261119-44517102376-ubuntu2604.tar.gz
NPU_DRIVER_SHA256_UBUNTU26 ?= 8003e3edc70ba5e4015df3c526a263fc950ee451d05d10c32813debad8957e96
LEVEL_ZERO_LOADER_URL_UBUNTU22 ?= https://github.com/oneapi-src/level-zero/releases/download/v1.24.2/level-zero_1.24.2+u22.04_amd64.deb
LEVEL_ZERO_LOADER_SHA256_UBUNTU22 ?= 7c304e93835d96025c90f6a3d8f2ce5edf142c24da9f4871113a4f0225fef22e
LEVEL_ZERO_LOADER_URL_UBUNTU24 ?= https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20261110T100000Z/pool/main/l/level-zero-loader/libze1_1.34.0-1~24.04~ppa1_amd64.deb
LEVEL_ZERO_LOADER_SHA256_UBUNTU24 ?= 4ea0d892f5e2b82db3c15c9e9a97327b6323189972b75fe4fc8b029a3aa3d26d
LEVEL_ZERO_LOADER_URL_UBUNTU26 ?= https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20261110T100000Z/pool/main/l/level-zero-loader/libze1_1.34.0-1~26.04~ppa1_amd64.deb
LEVEL_ZERO_LOADER_SHA256_UBUNTU26 ?= d2e3a395ff0b40add337330dcd9e86857963bc173c84a532b5c1cda0dbe7a184
# Windows Python runtime and build interpreters.
WINDOWS_PYTHON312_VERSION ?= 3.12.10
WINDOWS_PYTHON314_VERSION ?= 3.14.10
