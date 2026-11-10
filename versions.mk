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
OV_BRANCH ?= 92096e1c255cbd816480be7e4ead89da3f0189d8
OV_TOKENIZERS_BRANCH ?= 47eb7817ce620c9265ff369f01409c052985d2df
OV_GENAI_BRANCH ?= b2594e591b84f3aaf3cf35377440af5f5ebc17e5

# Source repositories (GitHub owner/repository)
OV_REPO ?= openvinotoolkit/openvino
OV_GENAI_REPO ?= openvinotoolkit/openvino.genai
OV_TOKENIZERS_REPO ?= openvinotoolkit/openvino_tokenizers

# Binary package URLs for each supported platform.
DLDT_PACKAGE_URL_UBUNTU26 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2026.5.0.0rc1/openvino_genai_ubuntu26_2026.5.0.0rc1_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU26 ?= 1f160c8a7cf0610a245c3ffcef10295175519944444ab1f0c0279423dce294ff
DLDT_PACKAGE_URL_UBUNTU24 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2026.5.0.0rc1/openvino_genai_ubuntu24_2026.5.0.0rc1_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU24 ?= f9be78f70b29120fdf88389dad4eeed21469040317bbf504d7a6ac93088623e2
DLDT_PACKAGE_URL_UBUNTU22 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2026.5.0.0rc1/openvino_genai_ubuntu22_2026.5.0.0rc1_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU22 ?= 39184b4d5dcfbfdb99c41fba27fbe13e8c29038f8be0dc2e9f736658f5b7891f
DLDT_PACKAGE_URL_RHEL ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2026.5.0.0rc1/openvino_genai_rhel9_2026.5.0.0rc1_x86_64.tar.gz
DLDT_PACKAGE_SHA256_RHEL ?= 06c7dde1db2c82d0752a0796375ecf087390b8d6eccb167f315b628ad3ab0e15
GENAI_PACKAGE_URL_WINDOWS ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2026.5.0.0rc1/openvino_genai_windows_2026.5.0.0rc1_x86_64.zip
GENAI_PACKAGE_SHA256_WINDOWS ?= 349f99b68a6b37631d027787dcc83df4728c4325e592fd799e149248a70e2571
# Third-party dependency versions (used for binary builds)
OPENCV_VERSION ?= 4.14.0
CURL_VERSION ?= 8.22.0_2
