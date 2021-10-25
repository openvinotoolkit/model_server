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

FROM docker.io/library/ubuntu:20.04
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

COPY *.deb  /

SHELL ["/bin/bash", "-c"]

RUN set -exuo pipefail ; \
	rm -f /etc/apt/apt.conf.d/docker-clean ; \
	echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache ; \
	apt update ; \
	apt install -y \
		 gnupg2 \
		 software-properties-common
RUN	apt-key adv --keyserver-options http-proxy=http://proxy-us.intel.com:911 --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN     add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN     apt update
RUN	apt-get update 
RUN	DEBIAN_FRONTEND=noninteractive apt install -y \
		./libcudnn*.deb \
		autoconf \
		automake \
		build-essential \
		ca-certificates \
		clang-format-9 \
		cuda \
		curl \
		g++-multilib \
		gcc-multilib \
		git \
		git-lfs \
		gstreamer1.0-plugins-base \
		libavcodec-dev \
		libavformat-dev \
		libboost-regex-dev \
		libcairo2-dev \
		libcutensor1 \
		libcutensor-dev \
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
		libzstd-dev \
		pkg-config \
		python \
		python3-pip \
		shellcheck \
		time \
		unzip \
		wget \
	; \
	pip3 install cython ; \
	curl -L https://github.com/Kitware/ninja/releases/download/v1.10.0.gfb670.kitware.jobserver-1/ninja-1.10.0.gfb670.kitware.jobserver-1_x86_64-linux-gnu.tar.gz | tar xzv --strip-components=1 -C /usr/local/bin ; \
	curl https://github.com/mozilla/sccache/releases/download/v0.2.15/sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz -L | tar xvzC /usr/local/bin --strip-components=1 --wildcards '*/sccache' ; \
	chmod a+x /usr/local/bin/sccache ; \
	curl https://github.com/Kitware/CMake/releases/download/v3.15.0/cmake-3.15.0-Linux-x86_64.tar.gz -L | tar xzvC /usr/local --exclude={doc,man} --strip-components=1 ; \
	curl -L https://github.com/ccache/ccache/releases/download/v4.3/ccache-4.3.tar.xz | tar xJv ; \
	cd ccache-4.3 ; \
	mkdir build ; \
	cd build ; \
	cmake -DCMAKE_BUILD_TYPE=Release -G Ninja .. ; \
	ninja -v install ; \
	cd / ; \
	rm -r *.deb ccache*

ENV CUDA_PATH=/usr/local/cuda
ENV PATH=${CUDA_PATH}/bin:$PATH

RUN git clone https://github.com/openvinotoolkit/openvino /openvino ; cd /openvino ; git checkout 2021.4.1; git submodule update --init --recursive

ENV OPENVINO_BUILD_PATH=/cuda_plugin_build
ENV OPENVINO_HOME=/openvino
ENV OPENVINO_CONTRIB=/openvino_contrib

RUN mkdir /openvino_contrib/
COPY ./openvino_contrib/ /openvino_contrib/
RUN /openvino_contrib/modules/cuda_plugin/build.sh --build

RUN mkdir /auto_plugin
COPY ./iotg-odt-prc-openvino/ /auto_plugin/
RUN cmake "${OPENVINO_HOME}" \  
        -DVERBOSE_BUILD=ON \
        -DENABLE_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_TESTS=ON \
        -DENABLE_FASTER_BUILD=ON \
        -DIE_EXTRA_MODULES=${OPENVINO_CONTRIB}/modules \
        -DBUILD_java_api=OFF \
        -DBUILD_arm_plugin=OFF \
        -DENABLE_STRICT_DEPENDENCIES=OFF 

RUN make -j 48
RUN cmake -DCMAKE_INSTALL_PREFIX=/auto_plugin/build -P cmake_install.cmake

WORKDIR /plugins/
RUN cp /openvino/bin/intel64/Release/lib/libCUDAPlugin.so /plugins/
RUN cp /openvino/bin/intel64/Release/lib/libngraph_backend.so /plugins/
RUN cp /openvino/bin/intel64/Release/lib/libinterpreter_backend.so /plugins/
RUN cp /auto_plugin/build/deployment_tools/inference_engine/lib/intel64/libAutoPlugin.so /plugins/

