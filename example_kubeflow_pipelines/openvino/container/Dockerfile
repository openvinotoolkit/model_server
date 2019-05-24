FROM intelpython/intelpython3_core as TENSORFLOW

RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev && \
        apt-get clean

RUN git clone -b r1.14 --depth 1 https://github.com/tensorflow/tensorflow


RUN conda create --name myenv -y
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.24.1
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

RUN cd tensorflow && bazel build tensorflow/tools/graph_transforms:summarize_graph

FROM intelpython/intelpython3_core as IE
RUN apt-get update && apt-get install -y \
            autoconf \
            automake \
            build-essential \
            ca-certificates \
            curl \
            git \
            gstreamer1.0-plugins-base \
            libavcodec-dev \
            libavformat-dev \
            libboost-regex-dev \
            libcairo2-dev \
            libgfortran3 \
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
            pkg-config \
            unzip \
            vim \
            wget

RUN wget https://cmake.org/files/v3.14/cmake-3.14.3.tar.gz && \
    tar -xvzf cmake-3.14.3.tar.gz && \
    cd cmake-3.14.3/  && \
    ./configure && \
    make -j$(nproc) && \
    make install

RUN echo "deb http://ftp.us.debian.org/debian/ jessie main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb-src http://ftp.us.debian.org/debian/ jessie main contrib non-free" >> /etc/apt/sources.list && \
    apt update && \
    apt-get install -y g++-4.9
ENV CXX=/usr/bin/g++-4.9
RUN pip install cython numpy
ARG DLDT_DIR=/dldt-2019_R1.0.1
RUN git clone --depth=1 -b 2019_R1.0.1 https://github.com/opencv/dldt.git ${DLDT_DIR} && \
    cd ${DLDT_DIR} && git submodule init && git submodule update --recursive && \
    rm -Rf .git && rm -Rf model-optimizer

WORKDIR ${DLDT_DIR}
RUN curl -L https://github.com/intel/mkl-dnn/releases/download/v0.18/mklml_lnx_2019.0.3.20190220.tgz | tar -xz
WORKDIR ${DLDT_DIR}/inference-engine/build
RUN cmake -DGEMM=MKL  -DMKLROOT=${DLDT_DIR}/mklml_lnx_2019.0.3.20190220 -DENABLE_MKL_DNN=ON -DTHREADING=OMP -DCMAKE_BUILD_TYPE=Release ..
RUN make -j$(nproc)
WORKDIR ${DLDT_DIR}/inference-engine/ie_bridges/python/build
RUN cmake -DInferenceEngine_DIR=${DLDT_DIR}/inference-engine/build -DPYTHON_EXECUTABLE=$(which python) -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.6m ${DLDT_DIR}/inference-engine/ie_bridges/python && \
    make -j$(nproc)

FROM intelpython/intelpython3_core as FINAL
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
        python3-pip \
        gcc \
        python-setuptools \
        python3-setuptools \
        libgfortran3 \
        unzip \
        vim && \
        apt-get clean
RUN curl -L -o 2019_R1.0.1.tar.gz https://github.com/opencv/dldt/archive/2019_R1.0.1.tar.gz && \
    tar -zxf 2019_R1.0.1.tar.gz && \
    rm 2019_R1.0.1.tar.gz && \
    rm -Rf dldt-2019_R1.0.1/inference-engine
WORKDIR dldt-2019_R1.0.1/model-optimizer

RUN conda create --name myenv -y
ENV PATH /opt/conda/envs/myenv/bin:$PATH

RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt
RUN curl -L -o google-cloud-sdk.zip https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.zip && \
    unzip -qq google-cloud-sdk.zip -d tools && \
    rm google-cloud-sdk.zip && \
    tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=true --bash-completion=false \
        --disable-installation-options && \
    tools/google-cloud-sdk/bin/gcloud -q components update \
        gcloud core gsutil && \
    tools/google-cloud-sdk/bin/gcloud config set component_manager/disable_update_check true && \
    touch tools/google-cloud-sdk/lib/third_party/google.py && \
    pip install -U crcmod
ENV PATH ${PATH}:/dldt-2019_R1.0.1/model-optimizer:/dldt-2019_R1.0.1/model-optimizer/tools/google-cloud-sdk/bin

COPY --from=IE /dldt-2019_R1.0.1/inference-engine/bin/intel64/Release/lib/*.so /usr/local/lib/
COPY --from=IE /dldt-2019_R1.0.1/inference-engine/ie_bridges/python/bin/intel64/Release/python_api/python3.6/openvino/ /usr/local/lib/openvino/
COPY --from=IE /dldt-2019_R1.0.1/mklml_lnx_2019.0.3.20190220/lib/lib*.so /usr/local/lib/
ENV LD_LIBRARY_PATH=/usr/local/lib

WORKDIR /slim

RUN git clone --depth 1 https://github.com/tensorflow/models && rm -Rf models/.git && \
    git clone --depth 1  -b r1.14 https://github.com/tensorflow/tensorflow && rm -Rf tensorflow/.git

ENV PYTHONPATH=/usr/local/lib:/slim/models/research/slim:/slim/tensorflow/python/tools

COPY --from=TENSORFLOW /tensorflow/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph /usr/bin/summarize_graph
COPY --from=TENSORFLOW /root/.cache/bazel/_bazel_root/*/execroot/org_tensorflow/bazel-out/k8-opt/bin/_solib_k8/_U_S_Stensorflow_Stools_Sgraph_Utransforms_Csummarize_Ugraph___Utensorflow/libtensorflow_framework.so.1 /usr/local/lib/libtensorflow_framework.so.1

WORKDIR /scripts

COPY classes.py convert_model.py predict.py slim_model.py requirements.txt ./
RUN pip install -r requirements.txt

