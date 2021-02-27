FROM centos:7
ARG NODE_NAME=east-ocr
RUN yum install centos-release-scl -y && yum install -y \
    devtoolset-8-gcc*

RUN curl -s https://download.01.org/opencv/master/openvinotoolkit/thirdparty/linux/opencv/opencv_4.5.1-044_centos7.txz | tar --use-compress-program=xz -xvf -

COPY . /custom_node/
WORKDIR /custom_node/
RUN mkdir -p lib
RUN /opt/rh/devtoolset-8/root/bin/g++  -c -std=c++17 ${NODE_NAME}.cpp -fpic  -I/opencv/include/
RUN /opt/rh/devtoolset-8/root/bin/g++ -shared -o lib/libcustom-node-${NODE_NAME}.so ${NODE_NAME}.o -L/opencv/lib/ -I/opencv/include/ -lopencv_core -lopencv_imgproc -lopencv_imgcodecs



