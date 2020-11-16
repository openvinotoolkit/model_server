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

RUN rm /etc/yum.repos.d/CentOS-Base.repo
RUN rm /etc/yum.repos.d/epel.repo
ADD CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo
ADD epel.repo /etc/yum.repos.d/epel.repo

RUN http_proxy=${HTTP_PROXY} yum install -y glibc.i686 libgcc.x86_64 libgcc.i686 numactl redhat-lsb-core.i686
ADD ./kwbuildtools /tmp/kwbuildtools
WORKDIR /ovms/src
ENV InferenceEngine_DIR=/opt/intel/openvino/inference_engine/share/
ENV ngraph_DIR=/opt/intel/openvino/deployment_tools/ngraph/cmake/
ENV LD_LIBRARY_PATH=LD_LIBRARY_PATH:/opt/intel/openvino/inference_engine/external/tbb/lib
ADD CMakeLists.txt /ovms/src
RUN export BAZEL_BUILD=$(ls /root/.cache/bazel/_bazel_root/ | egrep  '.{32}') && \
    grep '\.a\|\.o' /root/.cache/bazel/_bazel_root/$BAZEL_BUILD/execroot/ovms/bazel-out/k8-opt/bin/src/ovms-2.params | grep -Ev '(ovms_lib|libstdc\+\+|main.pic.o)' | \
    sed "s/^bazel-out\/k8-opt\/bin\/external\//\/root\/\.cache\/bazel\/_bazel_root\/$BAZEL_BUILD\/execroot\/ovms\/bazel-out\/k8-opt\/bin\/external\//g" | \
    sed "s/^external\//\/root\/\.cache\/bazel\/_bazel_root\/$BAZEL_BUILD\/external\//g" > ovms_libs.in && \
    cmake3 .
RUN /tmp/kwbuildtools/bin/kwinject --output ./out.out make
RUN echo $ENV_KLOCWORK_TOKEN > /ovms/ltoken
RUN /tmp/kwbuildtools/bin/kwbuildproject --force --url ${ENV_KLOCWORK_URL}"${ENV_KLOCWORK_PROJECT}" --tables-directory kwtables out.out ; exit 0
RUN /tmp/kwbuildtools/bin/kwadmin --url ${ENV_KLOCWORK_URL} load "${ENV_KLOCWORK_PROJECT}" kwtables
