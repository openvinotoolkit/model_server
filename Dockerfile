FROM ubuntu:16.04 as DEV
RUN apt-get update && apt-get install -y \
            autoconf \
            automake \
            build-essential \
            ca-certificates \
            curl \
            gcc-multilib \
            git \
            g++-multilib \
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
            libpng12-dev \
            libssl-dev \
            libswscale-dev \
            libtool \
            libusb-1.0-0-dev \
            pkg-config \
            python3-pip \
            python-dev \
            unzip \
            vim \
            wget

RUN wget https://cmake.org/files/v3.14/cmake-3.14.3.tar.gz && \
    tar -xvzf cmake-3.14.3.tar.gz && \
    cd cmake-3.14.3/  && \
    ./configure && \
    make -j$(nproc) && \
    make install
RUN pip3 install cython numpy
ARG DLDT_DIR=/2019_R1.0.1
RUN git clone --depth=1 -b 2019_R1.0.1 https://github.com/opencv/dldt.git ${DLDT_DIR} && \
    cd ${DLDT_DIR} && git submodule init && git submodule update --recursive && \
    rm -Rf .git && rm -Rf model-optimizer

WORKDIR ${DLDT_DIR}
RUN curl -L https://github.com/intel/mkl-dnn/releases/download/v0.18/mklml_lnx_2019.0.3.20190220.tgz | tar -xz
WORKDIR ${DLDT_DIR}/inference-engine/build
RUN cmake -DGEMM=MKL  -DMKLROOT=${DLDT_DIR}/mklml_lnx_2019.0.3.20190220 -DENABLE_MKL_DNN=ON -DTHREADING=OMP -DCMAKE_BUILD_TYPE=Release ..
RUN make -j$(nproc)
WORKDIR ${DLDT_DIR}/inference-engine/ie_bridges/python/build
RUN cmake -DInferenceEngine_DIR=${DLDT_DIR}/inference-engine/build -DPYTHON_EXECUTABLE=$(which python3) -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m ${DLDT_DIR}/inference-engine/ie_bridges/python && \
    make -j$(nproc)

FROM ubuntu:16.04 as PROD

RUN apt-get update && apt-get install -y --no-install-recommends \
            ca-certificates \
            curl \
            libgomp1 \
            python3-dev \
            python3-pip \
            virtualenv
WORKDIR /ie-serving-py

COPY requirements.txt /ie-serving-py/
RUN virtualenv -p python3 .venv && \
    . .venv/bin/activate && pip3 install -r requirements.txt

COPY start_server.sh setup.py requirements.txt version /ie-serving-py/
COPY ie_serving /ie-serving-py/ie_serving

RUN . .venv/bin/activate && pip3 install .

COPY --from=DEV /2019_R1.0.1/inference-engine/bin/intel64/Release/lib/*.so /usr/local/lib/
COPY --from=DEV /2019_R1.0.1/inference-engine/ie_bridges/python/bin/intel64/Release/python_api/python3.5/openvino/ /usr/local/lib/openvino/
COPY --from=DEV /2019_R1.0.1/mklml_lnx_2019.0.3.20190220/lib/lib*.so /usr/local/lib/
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONPATH=/usr/local/lib

