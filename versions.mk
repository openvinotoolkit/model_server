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
OV_BRANCH ?= 46d2833b3d0aedee391be940d88509761c5e0000
OV_TOKENIZERS_BRANCH ?= f7999c4078db739a920eba72c6049892e7aabfbd
OV_GENAI_BRANCH ?= 5ae19c7988114f395a699a8192daca2d094a6f28

# Source repositories (GitHub owner/repository)
OV_REPO ?= openvinotoolkit/openvino
OV_GENAI_REPO ?= openvinotoolkit/openvino.genai
OV_TOKENIZERS_REPO ?= openvinotoolkit/openvino_tokenizers

# Binary package URLs for each supported platform.
DLDT_PACKAGE_URL_UBUNTU26 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2027.0.0.0.dev20261119/openvino_genai_ubuntu26_2027.0.0.0.dev20261119_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU26 ?= 5708f590494fed8a27a88d4ce93364b8044b24f397be492a1b5df3839dfc5bb7
DLDT_PACKAGE_URL_UBUNTU24 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2027.0.0.0.dev20261119/openvino_genai_ubuntu24_2027.0.0.0.dev20261119_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU24 ?= 23fb36238042e1d629f9c90022acd4522e61eebfff8ca77e26c78c253328a282
DLDT_PACKAGE_URL_UBUNTU22 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2027.0.0.0.dev20261119/openvino_genai_ubuntu22_2027.0.0.0.dev20261119_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU22 ?= 47e2e15978693d811a832b8b709b2f449638c276d9c6927c595d2609fcd78557
DLDT_PACKAGE_URL_RHEL ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2027.0.0.0.dev20261119/openvino_genai_rhel9_2027.0.0.0.dev20261119_x86_64.tar.gz
DLDT_PACKAGE_SHA256_RHEL ?= 4157a27dbd5a57d1858b38659a97b6a86483fe4d35cc03cc4a1ea2a66c93a363
GENAI_PACKAGE_URL_WINDOWS ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2027.0.0.0.dev20261119/openvino_genai_windows_2027.0.0.0.dev20261119_x86_64.zip
GENAI_PACKAGE_SHA256_WINDOWS ?= 9151fa34b607000820491186ec4a7a62812ffbadcb00c3c60a209dba23168e6b
# Third-party dependency versions (used for binary builds)
OPENCV_VERSION ?= 4.14.0
CURL_VERSION ?= 8.22.0_2
