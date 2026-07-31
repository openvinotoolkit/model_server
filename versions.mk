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
# NOTE: OV_SOURCE_BRANCH and OV_GENAI_BRANCH are pinned to the commits required
# by the onyx-support patches in ./patches (see patches/openvino/readme.md and
# patches/openvino.genai/readme.md). This is a temporary, one-off pin - restore
# the previous commits below once the patches are no longer needed:
#   OV_SOURCE_BRANCH ?= d08e55c64c37fde1f4f6157cc5f5e07dd36ce5e8  (pre-patch branch tip)
#   OV_GENAI_BRANCH ?= 8981d6f848f17985979be0a9224251d181f68c56  (pre-patch branch tip)
# OV_TOKENIZERS_BRANCH tracks upstream and is safe to advance ahead of the
# commit referenced by the genai patch's submodule bump: OVMS builds tokenizers
# as a separate component (BUILD_TOKENIZERS=OFF in the genai cmake invocation),
# so its pin does not affect the OVMS build.
OV_SOURCE_BRANCH ?= 5b6997da03a7a0713fb4376f9109b4832383cc24
OV_TOKENIZERS_BRANCH ?= 183c6f25cda2a469cba5eff8b72022d2d51ba0ca
OV_GENAI_BRANCH ?= c637ed85efebf1a44d5f0433845849a2d80b353c

# Source repository organizations
OV_SOURCE_ORG ?= openvinotoolkit
OV_GENAI_ORG ?= openvinotoolkit
OV_TOKENIZERS_ORG ?= openvinotoolkit

# Binary package URLs for each supported platform.
DLDT_PACKAGE_URL_UBUNTU24 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2026.3.0.0rc2/openvino_genai_ubuntu24_2026.3.0.0rc2_x86_64.tar.gz
DLDT_PACKAGE_URL_UBUNTU22 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2026.3.0.0rc2/openvino_genai_ubuntu22_2026.3.0.0rc2_x86_64.tar.gz
DLDT_PACKAGE_URL_RHEL ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2026.3.0.0rc2/openvino_genai_rhel8_2026.3.0.0rc2_x86_64.tar.gz
GENAI_PACKAGE_URL_WINDOWS ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2026.3.0.0rc2/openvino_genai_windows_2026.3.0.0rc2_x86_64.zip

# Third-party dependency versions (used for binary builds)
OPENCV_VERSION ?= 4.13.0
CURL_VERSION ?= 8.21.0_4
