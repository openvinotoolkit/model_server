#
# Copyright (c) 2022 Intel Corporation
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

cmake_minimum_required(VERSION 3.17)

project(kserve-samples LANGUAGES C CXX)

set(CMAKE_INSTALL_PREFIX install)

set(TRITON_ENABLE_CC_HTTP ON)
set(TRITON_ENABLE_CC_GRPC ON)

include(FetchContent)
FetchContent_Declare(
  triton-client
  GIT_REPOSITORY https://github.com/triton-inference-server/client.git
  GIT_TAG main
)
FetchContent_MakeAvailable(triton-client)
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.0.0  
)
FetchContent_MakeAvailable(cxxopts)
link_directories("${triton-client_BINARY_DIR}/cc-clients/install/lib")
include_directories("${triton-client_BINARY_DIR}/cc-clients/install/include")
include_directories("${cxxopts_SOURCE_DIR}/include")

add_subdirectory(samples)
