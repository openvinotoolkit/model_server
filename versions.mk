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
OV_SOURCE_BRANCH ?= ede283a88e35465f0d680dabbf1f44080f8fc387
OV_TOKENIZERS_BRANCH ?= 6da74a793f0fa5ca2f47a15af674cf715fccbf41
OV_GENAI_BRANCH ?= 7dea0459b2ac7d8dfd877fd9df6737674fd8371d

# Source repository organizations
OV_SOURCE_ORG ?= openvinotoolkit
OV_GENAI_ORG ?= openvinotoolkit
OV_TOKENIZERS_ORG ?= openvinotoolkit

# Binary package URLs for each supported platform.
DLDT_PACKAGE_URL_UBUNTU24 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2026.2.1.0rc1/openvino_genai_ubuntu24_2026.2.1.0rc1_x86_64.tar.gz
DLDT_PACKAGE_URL_UBUNTU22 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2026.2.1.0rc1/openvino_genai_ubuntu22_2026.2.1.0rc1_x86_64.tar.gz
DLDT_PACKAGE_URL_RHEL ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2026.2.1.0rc1/openvino_genai_rhel8_2026.2.1.0rc1_x86_64.tar.gz
GENAI_PACKAGE_URL_WINDOWS ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2026.2.1.0rc1/openvino_genai_windows_2026.2.1.0rc1_x86_64.zip

# Third-party dependency versions (used for binary builds)
OPENCV_VERSION ?= 4.13.0
CURL_VERSION ?= 8.20.0_2
