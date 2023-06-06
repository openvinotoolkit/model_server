FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3
RUN  apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python3 python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev yasm
RUN apt-get install -y python3-pip && pip3 install 'numpy<1.24'
RUN apt-get install -y libva-dev libdrm-dev wayland-protocols libx11-dev libx11-xcb-dev libxcb-present-dev libxcb-dri3-dev libvpl-dev
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
   make && \
   make install
RUN cp _vplinstall/lib/libvpl.so.2.9 /lib/libvpl.so
RUN git clone https://github.com/FFmpeg/FFmpeg.git /ffmpeg && \
    cd ffmpeg && \
    ./configure --enable-libvpl --toolchain=gcc-tsan && \
    make -j60 && \
    make install 
RUN git clone https://github.com/opencv/opencv.git /opencv && \
    cd /opencv && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_GSTREAMER=ON -DWITH_FFMPEG=ON \
  -DPYTHON_EXECUTABLE=$(which python) -DBUILD_opencv_python2=OFF \
  -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
  -DPYTHON3_EXECUTABLE=$(which python3) \
  -DPYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. && \
  make -j $(nproc) && \
  make install
RUN python3 -c "import cv2 ; print(cv2.getBuildInformation()) "

RUN pip3 install tritonclient[all]
COPY ./horizontal_text_detection.py ./horizontal_text_detection.py 
