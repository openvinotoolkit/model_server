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
FROM ubuntu:22.04 as BASE
ARG DEBIAN_FRONTEND=noninteractive
ARG JOBS
RUN apt-get update -qq && apt-get --no-install-recommends -y install \
  autoconf \
  automake \
  build-essential \
  cmake \
  python3-dev \
  git-core \
  pkg-config \
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
  libnuma-dev \
  libvpx-dev \
  libvpl-dev \
  libfdk-aac-dev \
  libopus-dev \
  libswscale-dev \
  libavcodec-dev \
  libavformat-dev \
  libswresample-dev \
  libdav1d-dev \
  python3-pip \ 
  gstreamer1.0* \
  libx264-dev \ 
  libx265-dev \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  && rm -rf /var/lib/apt/lists/*
RUN pip3 install "numpy<2.0.0"
RUN git clone https://github.com/intel/libvpl.git onevpl --branch v2023.3.1 && \
  cd onevpl && \
  export VPL_INSTALL_DIR=`pwd`/../_vplinstall && \
  mkdir _build && \
  mkdir $VPL_INSTALL_DIR && \
  cd _build && \
  cmake .. -DCMAKE_INSTALL_PREFIX=$VPL_INSTALL_DIR && \
  cmake --build . --config Release && \
  cmake --build . --config Release --target install
 RUN git clone https://github.com/intel/vpl-gpu-rt.git onevpl-gpu --branch intel-onevpl-23.3 && \
   cd onevpl-gpu  && \
   mkdir build && cd build && \
   cmake .. && \
   make -j $JOBS && \
   make install
RUN cp _vplinstall/lib/libvpl.so.2.9 /lib/libvpl.so
RUN ln -s /lib/libvpl.so /lib/libvpl.so.2
RUN git clone https://github.com/FFmpeg/FFmpeg.git /ffmpeg && \
    cd ffmpeg && \
    ./configure --enable-libvpl --toolchain=gcc-tsan --enable-gpl --enable-libvpx --enable-zlib  --prefix=/usr --libdir=/usr/lib64 --pkgconfigdir=/usr/lib64/pkgconfig --enable-libx264 --enable-libx265 --enable-cross-compile --incdir=/usr/include && \
    make -j $JOBS && \
    make install
RUN git clone https://github.com/opencv/opencv.git /opencv --branch 4.8.1 && \
    cd /opencv && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_FFMPEG=ON -D BUILD_SHARED_LIBS=OFF -D WITH_GSTREAMER=ON \
  -DPYTHON_EXECUTABLE=$(which python) -DBUILD_opencv_python2=OFF \
  -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
  -DPYTHON3_EXECUTABLE=$(which python3) \
  -DPYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. && \
  make -j $JOBS && \
  make install


RUN mkdir /cvlib && find /lib/x86_64-linux-gnu/ -name 'libopencv*' -print | xargs -I % cp % /cvlib
RUN mkdir /fflib && ldd /bin/ffmpeg | awk '{print $3}' | sed '/^\s*$/d' | xargs -I % cp % /fflib
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
COPY --from=BASE /lib/python3/dist-packages/cv2 /lib/python3/dist-packages/cv2
COPY --from=BASE /bin/ffmpeg /bin/ffmpeg
COPY --from=BASE _vplinstall/lib/libvpl.so.2.9 /lib/libvpl.so
RUN ln -s /lib/libvpl.so /lib/libvpl.so.2
COPY --from=BASE /lib64 /lib64
COPY --from=BASE /cvlib /lib/x86_64-linux-gnu/
COPY --from=BASE /fflib /lib/x86_64-linux-gnu/
RUN apt-get update -qq && apt-get --no-install-recommends -y install python3 python3-pip libavcodec58 libavformat58 libswscale5 libgstreamer-opencv1.0-0 gstreamer1.0* && rm -rf /var/lib/apt/lists/*
COPY ./requirements.txt ./requirements.txt 
RUN sed -i '/opencv-python/d' requirements.txt && pip3 install -r requirements.txt
RUN python3 -c "import cv2 ; print(cv2.getBuildInformation()) "
COPY ./stream_client.py ./stream_client.py 
ENTRYPOINT ["python3", "/workspace/client.py"]
