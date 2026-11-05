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
OV_BRANCH ?= 9079d13308fc249bea8cae21ead4f15ecb269df0
OV_TOKENIZERS_BRANCH ?= 47eb7817ce620c9265ff369f01409c052985d2df
OV_GENAI_BRANCH ?= 8f6b609bcbc7e6d189fc2e1d146d66ab07763406

# Source repositories (GitHub owner/repository)
OV_REPO ?= openvinotoolkit/openvino
OV_GENAI_REPO ?= openvinotoolkit/openvino.genai
OV_TOKENIZERS_REPO ?= openvinotoolkit/openvino_tokenizers

# Binary package URLs for each supported platform.
DLDT_PACKAGE_URL_UBUNTU26 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.0.0.dev20261104/openvino_genai_ubuntu26_2026.5.0.0.dev20261104_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU26 ?= 76e8c97559d3ccbe96c3f4caa4ad185009c3f47d3e237b3de0c2687bc21b38b8
DLDT_PACKAGE_URL_UBUNTU24 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.0.0.dev20261104/openvino_genai_ubuntu24_2026.5.0.0.dev20261104_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU24 ?= 5b66d1442eebb2a5d0ae3aecd6c3253210e8f569acfe2be69b0236db4e1fa120
DLDT_PACKAGE_URL_UBUNTU22 ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.0.0.dev20261104/openvino_genai_ubuntu22_2026.5.0.0.dev20261104_x86_64.tar.gz
DLDT_PACKAGE_SHA256_UBUNTU22 ?= 98e3f7de5303085bde084de0f0180596c2ba46ef62ed076c4bcb58621f800681
DLDT_PACKAGE_URL_RHEL ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.0.0.dev20261104/openvino_genai_rhel9_2026.5.0.0.dev20261104_x86_64.tar.gz
DLDT_PACKAGE_SHA256_RHEL ?= 8a572d5ab543dd97aa1b69feb918fc49e9a17cdb3f388ad14bb30c4e69ee6bc4
GENAI_PACKAGE_URL_WINDOWS ?= https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2026.5.0.0.dev20261104/openvino_genai_windows_2026.5.0.0.dev20261104_x86_64.zip
GENAI_PACKAGE_SHA256_WINDOWS ?= 46185fb2183564fdc01a3174afa37eeb70930af6ccc264f11172745b9f4d1c2b
# Third-party dependency versions (used for binary builds)
OPENCV_VERSION ?= 4.14.0
CURL_VERSION ?= 8.22.0_2
