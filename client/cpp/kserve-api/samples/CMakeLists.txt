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

function(sample sample_name)
    add_executable(grpc_${sample_name} grpc_${sample_name}.cpp)
    add_dependencies(grpc_${sample_name} cc-clients)
    target_link_libraries(grpc_${sample_name} grpcclient libprotobuf.a)

    add_executable(http_${sample_name} http_${sample_name}.cpp)
    add_dependencies(http_${sample_name} cc-clients)
    target_link_libraries(http_${sample_name} httpclient)
endfunction()

sample(infer_dummy)
sample(infer_resnet)
sample(async_infer_resnet)
sample(server_live)
sample(server_ready)
sample(server_metadata)
sample(model_ready)
sample(model_metadata)
