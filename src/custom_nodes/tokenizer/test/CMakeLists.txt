#
# Copyright (c) 2023 Intel Corporation
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
cmake_minimum_required(VERSION 3.10)
project(test)

set(CMAKE_CXX_STANDARD 17)

# Download and install Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)

set(UTIL_DIRS
    ../src/             # for model.hpp
    ../../../)          # for custom_node_interface.h

file(DOWNLOAD
    https://github.com/microsoft/BlingFire/raw/${BLINGFIRE_SHA}/ldbsrc/ldb/gpt2.bin
    ${CMAKE_BINARY_DIR}/gpt2.bin)

file(DOWNLOAD
    https://github.com/microsoft/BlingFire/raw/${BLINGFIRE_SHA}/ldbsrc/ldb/gpt2.i2w
    ${CMAKE_BINARY_DIR}/gpt2.i2w)

add_executable(tokenization_test tokenization_test.cpp)
target_include_directories(tokenization_test PRIVATE ${UTIL_DIRS})
target_link_libraries(tokenization_test gtest_main tokenizer)

add_executable(detokenization_test detokenization_test.cpp)
target_include_directories(detokenization_test PRIVATE ${UTIL_DIRS})
target_link_libraries(detokenization_test gtest_main detokenizer)
