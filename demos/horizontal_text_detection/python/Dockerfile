#
# Copyright (c) 2023 Intel Corporation
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
FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python3 python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev yasm
RUN apt-get install -y libva-dev libdrm-dev wayland-protocols libx11-dev libx11-xcb-dev libxcb-present-dev libxcb-dri3-dev libvpl-dev
RUN apt-get -y install \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
  libass-dev \
  libfreetype6-dev \
  libgnutls28-dev \
  libmp3lame-dev \
  libsdl2-dev \
  libtool \
  libva-dev \
  libvdpau-dev \
  libvorbis-dev \
  libxcb1-dev \
  libxcb-shm0-dev \
  libxcb-xfixes0-dev \
  meson \
  ninja-build \
  pkg-config \
  texinfo \
  wget \
  yasm \
  zlib1g-dev \
  nasm \
  libx264-dev \
  libx265-dev \
  libnuma-dev \
  libvpx-dev \
  libfdk-aac-dev \
  libopus-dev \
  libswresample-dev \
  libdav1d-dev
RUN apt-get install -y python3-pip && pip3 install tritonclient[all] numpy==1.24.3
RUN git clone https://github.com/oneapi-src/oneVPL.git onevpl && \
  cd onevpl && \
  export VPL_INSTALL_DIR=`pwd`/../_vplinstall && \
  mkdir _build && \
  mkdir $VPL_INSTALL_DIR && \
  cd _build && \
  cmake .. -DCMAKE_INSTALL_PREFIX=$VPL_INSTALL_DIR && \
  cmake --build . --config Release && \
  cmake --build . --config Release --target install
 RUN git clone https://github.com/oneapi-src/oneVPL-intel-gpu onevpl-gpu && \
   cd onevpl-gpu  && \
   mkdir build && cd build && \
   cmake .. && \
   make -j $(nproc) && \
   make install
RUN cp _vplinstall/lib/libvpl.so.2.9 /lib/libvpl.so
RUN ln -s /lib/libvpl.so /lib/libvpl.so.2
RUN git clone https://github.com/FFmpeg/FFmpeg.git /ffmpeg && \
    cd ffmpeg && \
    ./configure --enable-libvpl --toolchain=gcc-tsan --enable-gpl --enable-libx264 --enable-libx265 --enable-libvpx --enable-zlib  --prefix=/usr --libdir=/usr/lib64 --pkgconfigdir=/usr/lib64/pkgconfig --incdir=/usr/include && \
    make -j $(nproc) && \
    make install
RUN git clone https://github.com/opencv/opencv.git /opencv && \
    cd /opencv && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_FFMPEG=ON \
  -DPYTHON_EXECUTABLE=$(which python) -DBUILD_opencv_python2=OFF \
  -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
  -DPYTHON3_EXECUTABLE=$(which python3) \
  -DPYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. && \
  make -j $(nproc) && \
  make install
RUN python3 -c "import cv2 ; print(cv2.getBuildInformation()) "

COPY ./rtsp_client.py ./rtsp_client.py 
ENTRYPOINT ["python3", "rtsp_client.py"]
