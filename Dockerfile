FROM ubuntu:18.04 as base_build

LABEL version="1.0.0"
LABEL description="OpenVINO Model Server"

ARG ovms_metadata_file

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        gcc-8 \
        g++-8 \
        git \
        libusb-1.0-0-dev \
        python-dev \
        python3-distutils \
        unzip \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set gcc8 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 700 --slave /usr/bin/g++ g++ /usr/bin/g++-7 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8

# Set up Bazel
ENV BAZEL_VERSION 2.0.0
WORKDIR /bazel
RUN curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Build OpenVINO and nGraph (OV dependency) with D_GLIBCXX_USE_CXX11_ABI=0
RUN git clone --recurse-submodules -j4 https://github.com/opencv/dldt --branch 2020.1 --single-branch --depth 1 /openvino
WORKDIR /openvino/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_SAMPLES=0 -DNGRAPH_USE_CXX_ABI=0 -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 ..
RUN make --jobs=$(nproc --all)

# Build OpenVINO Model Server
WORKDIR /ovms
COPY .bazelrc WORKSPACE /ovms/
RUN bazel build @org_tensorflow//tensorflow/core:framework
RUN bazel build @tensorflow_serving//tensorflow_serving/apis:prediction_service_cc_proto

COPY src/ /ovms/src/

RUN bazel build //src:ovms
RUN cp /openvino/bin/intel64/Release/lib/plugins.xml /root/.cache/bazel/_bazel_root/*/execroot/ovms/bazel-out/k8-opt/bin/_solib_k8/*/
RUN bazel test --test_summary=detailed --test_output=all //src:ovms_test

ADD ${ovms_metadata_file} metadata.json

ENTRYPOINT ["./bazel-bin/src/ovms"]
