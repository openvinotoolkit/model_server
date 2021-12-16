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

ARG BUILD_IMAGE=openvino/model_server-build:latest
FROM $BUILD_IMAGE

ARG ENV_KLOCWORK_PROJECT
ARG ENV_KLOCWORK_URL
ARG ENV_KLOCWORK_TOKEN
ARG KLOCWORK_LTOKEN=/ovms/ltoken

RUN http_proxy=${HTTP_PROXY} yum install -y glibc.i686 libgcc.x86_64 libgcc.i686 redhat-lsb-core.i686
ADD ./kwbuildtools /tmp/kwbuildtools

WORKDIR /example_cpp_client/cpp
RUN /tmp/kwbuildtools/bin/kwinject --output /tmp/out.out bazel build //src:all

WORKDIR /ovms/src
RUN /tmp/kwbuildtools/bin/kwinject --output /tmp/out.out bazel build //src:static_analysis

RUN echo $ENV_KLOCWORK_TOKEN > /ovms/ltoken
RUN /tmp/kwbuildtools/bin/kwbuildproject --force --url ${ENV_KLOCWORK_URL}"${ENV_KLOCWORK_PROJECT}" --tables-directory kwtables /tmp/out.out ; exit 0
RUN /tmp/kwbuildtools/bin/kwadmin --url ${ENV_KLOCWORK_URL} load "${ENV_KLOCWORK_PROJECT}" kwtables
