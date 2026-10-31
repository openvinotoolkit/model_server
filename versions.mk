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
OV_BRANCH ?= 9d72fa66c720d1ce9bb00da6b7cd7949e6795eea
OV_TOKENIZERS_BRANCH ?= 47eb7817ce620c9265ff369f01409c052985d2df
OV_GENAI_BRANCH ?= 62bdf13d920b9ce3096bb94eeac48bbc2afc1217

# Source repositories (GitHub owner/repository)
OV_REPO ?= openvinotoolkit/openvino
OV_GENAI_REPO ?= openvinotoolkit/openvino.genai
OV_TOKENIZERS_REPO ?= openvinotoolkit/openvino_tokenizers

# Binary package URLs for each supported platform.
DLDT_PACKAGE_URL_UBUNTU26 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.0.0.dev20261029/openvino_genai_ubuntu26_2026.5.0.0.dev20261029_x86_64.tar.gz
DLDT_PACKAGE_URL_UBUNTU24 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.0.0.dev20261029/openvino_genai_ubuntu24_2026.5.0.0.dev20261029_x86_64.tar.gz
DLDT_PACKAGE_URL_UBUNTU22 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.0.0.dev20261029/openvino_genai_ubuntu22_2026.5.0.0.dev20261029_x86_64.tar.gz
DLDT_PACKAGE_URL_RHEL ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.0.0.dev20261029/openvino_genai_rhel9_2026.5.0.0.dev20261029_x86_64.tar.gz
GENAI_PACKAGE_URL_WINDOWS ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.0.0.dev20261029/openvino_genai_windows_2026.5.0.0.dev20261029_x86_64.zip
# Third-party dependency versions (used for binary builds)
OPENCV_VERSION ?= 4.14.0
CURL_VERSION ?= 8.22.0_2

# llmman CLI used for OCI ModelPack pulls in build images.
LLMMAN_VERSION ?= 0.1.534
LLMMAN_SHA256_X86_64 ?= 33801d30c3a6f4d6ac0b8b2bb18f63b7af6b67c0b3ef04ea443eeba4df0879b5
LLMMAN_SHA256_X86_64_WINDOWS ?= 23a502ccf8b4facb37a1063cd13c6f15efe988386d39cc356ff9b5ebed4d0288
