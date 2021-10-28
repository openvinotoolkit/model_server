#
# Copyright (c) 2021 Intel Corporation
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

FROM ubuntu:20.04 as base_build

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt update && apt install -y --no-install-recommends \
        automake \
        build-essential \
        ca-certificates \
        curl \
        cmake \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng-dev \
        libtool \
        libzmq3-dev \
        mlocate \
        ninja-build \
        openjdk-8-jdk\
        openjdk-8-jre-headless \
        pkg-config \
        python-dev \
        software-properties-common \
        swig \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        python3-distutils \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 --no-cache-dir install \
    future>=0.17.1 \
    grpcio \
    h5py \
    keras_applications>=1.0.8 \
    keras_preprocessing>=1.1.0 \
    mock \
    numpy \
    requests \
    --ignore-installed setuptools \
    --ignore-installed six

RUN apt update && apt install -y --no-install-recommends \
        libopencv-dev \
        libtbb-dev

# Set up Bazel
ENV BAZEL_VERSION 3.7.2
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

WORKDIR /input/images
RUN cd /input/ && wget https://raw.githubusercontent.com/openvinotoolkit/model_server/v2021.4.1/example_client/input_images.txt && \
    cd /input/images && curl -sS https://github.com/openvinotoolkit/model_server/tree/v2021.4.1/example_client/images | \
    grep -oh "\/images.*jpeg" | cut -c 9- | cut -f1 -d"\"" | \
    xargs -I %s wget https://raw.githubusercontent.com/openvinotoolkit/model_server/v2021.4.1/example_client/images/%s

WORKDIR /build
COPY .bazelrc WORKSPACE /build/
COPY external /build/external/
COPY third_party /build/third_party/

RUN mkdir -p /opt/intel/openvino/opencv/ && ln -sf /usr/lib/x86_64-linux-gnu /opt/intel/openvino/opencv/lib
RUN mkdir -p /opt/intel/openvino/opencv/include && ln -sf /usr/include/opencv4/opencv2 /opt/intel/openvino/opencv/include

RUN bazel build \
    @tensorflow_serving//tensorflow_serving/apis:prediction_service_cc_proto \
    @com_github_grpc_grpc//:grpc++ \
    @com_google_protobuf//:protobuf_lite \
    @org_tensorflow//tensorflow/core:framework \
    @org_tensorflow//tensorflow/core:lib \
    @opencv//:opencv

COPY src/ /build/src/
RUN bazel build //src:all

FROM ubuntu:20.04 as release
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update && apt install -y --no-install-recommends libopencv-core4.2 libopencv-imgcodecs4.2 libtbb-dev && \
    apt clean && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*

WORKDIR /clients/
RUN mkdir -p /clients/libs/ && mkdir -p /clients/images
COPY --from=base_build \
    /build/bazel-bin/src/classification_client_sync \
    /build/bazel-bin/src/classification_client_async_benchmark \
    /build/bazel-bin/src/synthetic_client_async_benchmark \
    /clients/
COPY --from=base_build /input/images/*.jpeg /clients/images/
COPY --from=base_build /input/input_images.txt /clients/

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/clients/libs
