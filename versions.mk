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
OV_SOURCE_BRANCH ?= 2492324617a7bf9b4b14b5547e75545449bec1f0
OV_TOKENIZERS_BRANCH ?= 02bcda8f91fb5093f4028bb44980d03f44dce790
OV_GENAI_BRANCH ?= 1e7a63d14a18a0d227e5a2413763ee45395af58a

# Source repository organizations
OV_SOURCE_ORG ?= openvinotoolkit
OV_GENAI_ORG ?= openvinotoolkit
OV_TOKENIZERS_ORG ?= openvinotoolkit

# Binary package URLs for each supported platform.
DLDT_PACKAGE_URL_UBUNTU24 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.2.0.0.dev20260509/openvino_genai_ubuntu24_2026.2.0.0.dev20260509_x86_64.tar.gz
DLDT_PACKAGE_URL_UBUNTU22 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.2.0.0.dev20260509/openvino_genai_ubuntu22_2026.2.0.0.dev20260509_x86_64.tar.gz
DLDT_PACKAGE_URL_RHEL ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.2.0.0.dev20260509/openvino_genai_rhel8_2026.2.0.0.dev20260509_x86_64.tar.gz
GENAI_PACKAGE_URL_WINDOWS ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.2.0.0.dev20260509/openvino_genai_windows_2026.2.0.0.dev20260509_x86_64.zip
