#
# Copyright (c) 2020 Intel Corporation
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
cmake_minimum_required(VERSION 2.8)
project(ovms)
set(CMAKE_CXX_STANDARD 17)

file(GLOB ovms_SRC
    "*.hpp"
    "*.cpp"
)

add_executable(ovms ${ovms_SRC})

set(BAZEL_BUILD "$ENV{BAZEL_BUILD}")
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/external/spdlog/include/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/external/com_github_grpc_grpc/include/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/external/tensorflow_serving/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/external/cxxopts/include/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/external/eigen_archive/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/external/org_tensorflow/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/external/com_google_absl/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/execroot/ovms/bazel-out/k8-opt/bin/external/org_tensorflow/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/external/com_google_protobuf/src/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/execroot/ovms/bazel-out/k8-opt/bin/external/tensorflow_serving/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/external/rapidjson/include/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/external/com_github_googleapis_google_cloud_cpp/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/external/com_github_googleapis_google_cloud_cpp_common/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/execroot/ovms/bazel-out/k8-opt/bin/external/com_github_googleapis_google_cloud_cpp/google/cloud/storage/)
include_directories(/root/.cache/bazel/_bazel_root/${BAZEL_BUILD}/execroot/ovms/bazel-out/k8-opt/bin/external/awssdk/_virtual_includes/core/)
include_directories(/awssdk/aws-cpp-sdk-s3/include)

find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${InferenceEngine_DIR}/../include
    ${InferenceEngine_DIR}/../include/cpp
    ${NGraph_DIR}/../include
)

message(STATUS ${NGRAPH_LIBRARIES})

file(STRINGS ovms_libs.in OVMS_LIBS)
target_link_libraries(${PROJECT_NAME} PRIVATE ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} ${NGRAPH_LIBRARIES} )
target_link_libraries(${PROJECT_NAME} azurestorage cpprest)
target_link_libraries(${PROJECT_NAME} ${OVMS_LIBS})
target_link_libraries(${PROJECT_NAME} stdc++fs ssl crypto uuid xml2 dl)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")
