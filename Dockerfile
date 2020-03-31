FROM ubuntu:18.04 as base_build

ARG TF_SERVING_VERSION_GIT_BRANCH=master
ARG TF_SERVING_VERSION_GIT_COMMIT=head

LABEL maintainer=gvasudevan@google.com
LABEL tensorflow_serving_github_branchtag=${TF_SERVING_VERSION_GIT_BRANCH}
LABEL tensorflow_serving_github_commit=${TF_SERVING_VERSION_GIT_COMMIT}

RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        gcc-multilib \
        g++-multilib \
        git \
        gstreamer1.0-plugins-base \
        libavcodec-dev \
        libboost-regex-dev \
        libcairo2-dev \
        libcurl3-dev \
        libfreetype6-dev \
        libglib2.0-dev \
        libgstreamer1.0-0 \
        libgtk2.0-dev \
        libopenblas-dev \
        libpango1.0-dev \
        libpng-dev \
        libssl-dev \
        libswscale-dev \
        libtool \
        libusb-1.0-0-dev \
        libzmq3-dev \
        mlocate \
        nano \
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
        gnupg2 \
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

# Set up Bazel
ENV BAZEL_VERSION 2.0.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download TF Serving sources (optionally at specific commit).
WORKDIR /tensorflow-serving
RUN git clone --branch=${TF_SERVING_VERSION_GIT_BRANCH} https://github.com/tensorflow/serving . && \
    git remote add upstream https://github.com/tensorflow/serving.git && \
    if [ "${TF_SERVING_VERSION_GIT_COMMIT}" != "head" ]; then git checkout ${TF_SERVING_VERSION_GIT_COMMIT} ; fi

# Build OpenVINO and nGraph (OV dependency) with D_GLIBCXX_USE_CXX11_ABI=0
WORKDIR /openvino
RUN git clone https://github.com/opencv/dldt --branch 2020.1 --single-branch --depth 1 .
RUN git submodule update --init --recursive
COPY openvino.diff .
RUN git apply openvino.diff && rm openvino.diff
WORKDIR /openvino/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DNGRAPH_USE_CXX_ABI=0 ..
RUN make --jobs=$(nproc --all)

## Compile server
WORKDIR /tensorflow-serving
COPY src/ tensorflow_serving/ovms/
COPY WORKSPACE .
RUN bazel build //tensorflow_serving/ovms:server_cc

RUN cp /openvino/bin/intel64/Release/lib/plugins.xml /root/.cache/bazel/_bazel_root/*/execroot/tf_serving/bazel-out/k8-opt/bin/_solib_k8/*/

ENTRYPOINT ./bazel-bin/tensorflow_serving/ovms/server_cc
