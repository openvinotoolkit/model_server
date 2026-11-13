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
OV_BRANCH ?= e31618d01c26ea28b35f98ded16e37dd2e35133c
OV_TOKENIZERS_BRANCH ?= 5abf77d06a48058196d60450f5a1c1835d35affd
OV_GENAI_BRANCH ?= ca69756b797619e6a03c2b8f88b2defe01b7512b

# Source repositories (GitHub owner/repository)
OV_REPO ?= openvinotoolkit/openvino
OV_GENAI_REPO ?= openvinotoolkit/openvino.genai
OV_TOKENIZERS_REPO ?= openvinotoolkit/openvino_tokenizers

# Binary package URLs for each supported platform.
DLDT_PACKAGE_URL_UBUNTU26 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.1.0.dev20261113/openvino_genai_ubuntu26_2026.5.1.0.dev20261113_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU26 ?= 009dfb2135386cb6f31a2edf313fb1c7073f50ddda4c67c4664e2126672ec600
DLDT_PACKAGE_URL_UBUNTU24 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.1.0.dev20261113/openvino_genai_ubuntu24_2026.5.1.0.dev20261113_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU24 ?= 05ba1bf143b9ddf6d9ac32601c3123123432bc54706d8d6d1180cc2da41d323a
DLDT_PACKAGE_URL_UBUNTU22 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.1.0.dev20261113/openvino_genai_ubuntu22_2026.5.1.0.dev20261113_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU22 ?= b4b04041ffd8120d4ac21390cc901fd2691fd62540689bd40b19c99e718228d3
DLDT_PACKAGE_URL_RHEL ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.1.0.dev20261113/openvino_genai_rhel9_2026.5.1.0.dev20261113_x86_64.tar.gz
DLDT_PACKAGE_SHA256_RHEL ?= b2cf1c117cd95b63cf822efbe124e3a4d67ef0b0dd4a99c5a7e42edd017a4d18
GENAI_PACKAGE_URL_WINDOWS ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.1.0.dev20261113/openvino_genai_windows_2026.5.1.0.dev20261113_x86_64.zip
GENAI_PACKAGE_SHA256_WINDOWS ?= bd2217a34548dc2e1faed0f2d3dbe965d2b72f5e26ba833c56f8a7b8c5d804ec
# Third-party dependency versions (used for binary builds)
OPENCV_VERSION ?= 4.14.0
CURL_VERSION ?= 8.22.0_2
