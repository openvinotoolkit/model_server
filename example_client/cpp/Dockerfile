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

FROM ubuntu:20.04

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

WORKDIR /opencv
RUN git clone https://github.com/opencv/opencv --depth 1 --branch 4.5.3 . && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make && make install

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

WORKDIR /build
COPY .bazelrc WORKSPACE /build/
COPY external /build/external/
COPY third_party /build/third_party/

RUN bazel build \
    @tensorflow_serving//tensorflow_serving/apis:prediction_service_cc_proto \
    @com_github_grpc_grpc//:grpc++ \
    @com_google_protobuf//:protobuf_lite \
    @org_tensorflow//tensorflow/core:framework \
    @org_tensorflow//tensorflow/core:lib \
    @opencv//:opencv

COPY src/ /build/src/

RUN bazel build //src:resnet_client

ENV LD_LIBRARY_PATH=/usr/local/lib
ENTRYPOINT ["/build/bazel-bin/src/resnet_client"]
